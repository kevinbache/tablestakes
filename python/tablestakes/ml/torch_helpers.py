import abc
import time
from argparse import Namespace
from typing import *

import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from ray import tune
from ray.tune.logger import Logger as TuneLogger
from ray.tune import result as tune_result

from chillpill import params

from tablestakes import constants, utils
from tablestakes.ml import hyperparams, ablation

PS = TypeVar('PS', bound=params.ParameterSet)


def artless_smash(input: torch.Tensor) -> torch.Tensor:
    """
    Smash all vectors in time dimension into a series of summary layers.  cat them and work with that.
    This is about the simplest way you can move from O(n) time to O(1) time.
    If you want, make it better by starting with some conv layers.
    """
    means = torch.mean(input, dim=1, keepdim=True)
    vars = torch.var(input, dim=1, keepdim=True)
    l1s = torch.norm(input, p=1, dim=1, keepdim=True)
    l2s = torch.norm(input, p=2, dim=1, keepdim=True)
    lse = torch.logsumexp(input, dim=1, keepdim=True)

    maxs = torch.max(input, dim=1, keepdim=True)[0]
    mins = torch.min(input, dim=1, keepdim=True)[0]
    medians = torch.median(input, dim=1, keepdim=True)[0]

    return torch.cat((means, vars, l1s, l2s, lse, maxs, mins, medians), dim=1).view(input.shape[0], 1, -1)


class ArtlessSmasher(nn.Module):
    def __init__(self, num_input_channels: int):
        super().__init__()
        self.num_input_channels = num_input_channels

    @staticmethod
    def forward(x):
        return artless_smash(x)

    def get_num_output_features(self):
        return 8 * self.num_input_channels


# class ReachAroundEncoder(nn.Module):
#     """x_base and x_vocab are inputs to the transformer and a raw x_base is also appended to the output"""
#     def __init__(self, blah):
#         super().__init__()

class TransformerModuleParams(params.ParameterSet):
    num_layers = 4
    num_heads = 8
    fc_dim_mult = 2


def get_fast_linear_attention_encoder(hp: TransformerModuleParams, num_trans_input_dims: int, feature_map=None):
    """https://github.com/idiap/fast-transformers"""
    from fast_transformers import builders, feature_maps

    if feature_map == 'favor':
        feature_map = feature_maps.Favor.factory(n_dims=num_trans_input_dims)
    elif feature_map == 'grf':
        feature_map = feature_maps.GeneralizedRandomFeatures.factory(n_dims=num_trans_input_dims)
    else:
        feature_map = None

    # Create the builder for our transformers
    builder = builders.TransformerEncoderBuilder.from_kwargs(
        n_layers=hp.num_layers,
        n_heads=hp.num_heads,
        query_dimensions=num_trans_input_dims // hp.num_heads,
        value_dimensions=num_trans_input_dims // hp.num_heads,
        feed_forward_dimensions=num_trans_input_dims * hp.fc_dim_mult,
        attention_type='linear',
        activation='gelu',
        feature_map=feature_map,
    )

    # Build a transformer with linear attention
    return builder.get()


def get_performer_encoder(hp: TransformerModuleParams, num_trans_input_dims: int):
    """https://github.com/lucidrains/performer-pytorch"""
    from performer_pytorch import Performer
    return Performer(
        dim=num_trans_input_dims,
        depth=hp.num_layers,
        heads=hp.num_heads,
        ff_mult=hp.fc_dim_mult,
    )


def get_pytorch_transformer_encoder(hp: TransformerModuleParams, num_trans_input_dims: int):
    enc_layer = nn.TransformerEncoderLayer(
        d_model=num_trans_input_dims,
        nhead=hp.num_heads,
        dim_feedforward=hp.fc_dim_mult * num_trans_input_dims,
        activation='gelu'
    )

    return nn.TransformerEncoder(
        encoder_layer=enc_layer,
        num_layers=hp.num_layers,
    )


def get_simple_ablatable_transformer_encoder(
        hp: TransformerModuleParams,
        num_trans_input_dims: int,
        do_drop_k=True,
):
    return ablation.Encoder(
        d_model=num_trans_input_dims,
        num_layers=hp.num_layers,
        num_heads=hp.num_heads,
        d_ff_mult=hp.fc_dim_mult,
        do_drop_k=do_drop_k,
    )


CURRENT_EPOCH_NAME = 'current_epoch'
PARAM_COUNT_NAME = 'param_count'
TRAINABLE_PARAM_COUNT_NAME = 'trainable_param_count'
TIME_PERF_NAME = 'train_time_perf'
TIME_PROCESS_NAME = 'train_time_process'


class LogCopierCallback(pl.Callback):
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        d = {k: v.item() for k, v in pl_module.metrics_to_log.items()}
        d[CURRENT_EPOCH_NAME] = trainer.current_epoch
        tune.report(**d)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        d = {k: v.item() for k, v in pl_module.metrics_to_log.items()}
        d[CURRENT_EPOCH_NAME] = trainer.current_epoch
        tune.report(**d)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        d = {k: v.item() for k, v in pl_module.metrics_to_log.items()}
        d[CURRENT_EPOCH_NAME] = trainer.current_epoch
        tune.report(**d)


class CounterTimerCallback(pl.Callback):
    def __init__(self):
        self._train_start_perf = None
        self._train_start_process = None

    @staticmethod
    def _count_params(pl_module):
        return sum(p.numel() for p in pl_module.parameters())

    @staticmethod
    def _count_trainable_params(pl_module):
        return sum(p.numel() for p in pl_module.parameters() if p.requires_grad)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, ):
        if trainer.logger is None:
            return
        d = {
            PARAM_COUNT_NAME: self._count_params(pl_module),
            TRAINABLE_PARAM_COUNT_NAME: self._count_trainable_params(pl_module),
        }
        # noinspection PyTypeChecker
        trainer.logger.log_hyperparams(params=d)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        self._train_start_perf = time.perf_counter()
        self._train_start_process = time.process_time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        if trainer.logger is None:
            return
        d = {
            TIME_PERF_NAME: time.perf_counter() - self._train_start_perf,
            TIME_PROCESS_NAME: time.process_time() - self._train_start_process,
        }
        trainer.logger.log_metrics(d, step=trainer.global_step)


class BetterAccuracy(pl.metrics.Accuracy):
    """PyTorch Lightning's += lines cause warnings about transferring lots of scalars between cpu / gpu"""
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape,  f"preds.shape: {preds.shape}, target.shape: {target.shape}"
        self.correct = self.correct + torch.sum(preds.eq(target))
        self.total = self.total + target.numel()


# ref: https://community.neptune.ai/t/neptune-and-hyperparameter-search-with-tune/567/3
class TuneNeptuneLogger(TuneLogger):
    """Neptune logger.
    """

    def _init(self):
        from neptune.sessions import Session

        hp = hyperparams.LearningParams.from_dict(self.config)
        project_name = utils.get_neptune_fully_qualified_project_name(hp.project_name)
        experiment_name = f'tune_logger-{hp.get_exp_group_name()}-{tune.get_trial_id()}'

        project = Session().get_project(project_name)

        self.exp = project.create_experiment(
            name=experiment_name,
            params=self.config,
            tags=hp.experiment_tags,
            upload_source_files=constants.SOURCES_GLOB_STR,
        )

    def on_result(self, result):
        for name, value in result.items():
            if isinstance(value, float):
                self.exp.log_metric(name, x=result.get(tune_result.TRAINING_ITERATION), y=value)
            elif isinstance(value, int):
                self.exp.log_metric(name, x=result.get(tune_result.TRAINING_ITERATION), y=value)
            elif isinstance(value, str):
                self.exp.log_text(name, x=result.get(tune_result.TRAINING_ITERATION), y=value)
            else:
                continue

        # from ray.tune.result import (NODE_IP, TRAINING_ITERATION, TIME_TOTAL_S,
        #                              TIMESTEPS_TOTAL, EXPR_PARAM_FILE,
        #                              EXPR_PARAM_PICKLE_FILE, EXPR_PROGRESS_FILE,
        #                              EXPR_RESULT_FILE)

    def close(self):
        self.exp.stop()


if __name__ == '__main__':
    # num_batch, num_words, num_dims = 32, 93, 37
    # input = torch.rand(num_batch, num_words, num_dims)
    #
    # means = torch.mean(input, dim=1, keepdim=True)
    # vars = torch.var(input, dim=1, keepdim=True)
    # l1s = torch.abs(input - means).mean(dim=1, keepdim=True)
    #
    # maxs = torch.max(input, dim=1, keepdim=True)[0]
    # mins = torch.min(input, dim=1, keepdim=True)[0]
    # medians = torch.median(input, dim=1, keepdim=True)[0]
    #
    # torch.cat((means, vars, l1s, maxs, mins, medians), dim=1)
    pass