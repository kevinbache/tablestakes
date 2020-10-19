import time
from argparse import Namespace
from pathlib import Path
from typing import *

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
from ray import tune
from ray.tune.logger import Logger as TuneLogger
from ray.tune import result as tune_result
from tablestakes import constants, utils

from tablestakes.ml import hyperparams, ablation
from torch import nn as nn
from torch.nn import functional as F


class SizedSequential(nn.Sequential):
    def __init__(self, *args, num_output_features):
        super().__init__(*args)

        self._num_output_features = num_output_features

    def get_num_output_features(self):
        return self._num_output_features


def resnet_conv1_block(
        num_input_features: int,
        num_hidden_features: int,
        num_groups: int,
        activation=nn.LeakyReLU,
        do_include_norm=True,
):
    layers = [
        activation(),
        nn.Conv1d(num_input_features, num_hidden_features, 1),
    ]
    if do_include_norm:
        layers = [nn.GroupNorm(num_groups=num_groups, num_channels=num_input_features)] + layers

    return SizedSequential(*layers, num_output_features=num_hidden_features)


class FullyConv1Resnet(nn.Module):
    """
    Would be a normal resnet but I don't want to fix the size.

    https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
    """
    def __init__(
            self,
            num_input_features: int,
            neuron_counts: List[int],
            num_groups=32,
            num_blocks_per_residual=2,
            activation=nn.LeakyReLU,
            do_error_if_group_div_off=False,
            do_exclude_first_norm=True,
    ):
        super().__init__()
        self.neuron_counts = neuron_counts
        self.num_blocks_per_residual = num_blocks_per_residual
        self.activation = activation

        all_counts = [num_input_features] + neuron_counts

        # round up counts so they're group divisible
        remainders = [count % num_groups for count in all_counts]
        if any(remainders) and do_error_if_group_div_off:
            raise ValueError(f"Number of neurons ({all_counts}) must be divisible by number of groups ({num_groups})")

        # neurons which are added to each layer to ensure that layer sizes are divisible by num_groups
        extra_counts = [num_groups - r if r else 0 for r in remainders]
        self._extra_counts = extra_counts

        all_counts = [c + e for c, e in zip(all_counts, extra_counts)]

        self._num_output_features = all_counts[-1]

        blocks = []
        for block_ind, (num_in, num_hidden) in enumerate(zip(all_counts, all_counts[1:])):
            do_include_norm = block_ind > 0 or do_exclude_first_norm
            blocks.append(resnet_conv1_block(num_in, num_hidden, num_groups, self.activation, do_include_norm))
        self.blocks = nn.ModuleList(blocks)

        resizers = []
        prev_size = all_counts[0]
        for block_ind, block in enumerate(self.blocks):
            if block_ind > 0 and block_ind % self.num_blocks_per_residual == 0:
                new_size = block.get_num_output_features()
                if new_size == prev_size:
                    resizers.append(None)
                    continue
                resizers.append(nn.Conv1d(prev_size, new_size, 1))
                prev_size = new_size
            else:
                resizers.append(None)
        self.resizers = nn.ModuleList(resizers)

    def forward(self, x):
        # the remainders fixup to all_counts in __init__  doesn't account for the fact that the input x may
        # still be the wrong size.  pad it up if needed
        if self._extra_counts[0]:
            x = F.pad(
                x,
                pad=(0, 0, 0, self._extra_counts[0]),
                mode='constant',
                value=0,
            )
        old_x = x

        for block_index, (block, resizer, extra) in enumerate(zip(self.blocks, self.resizers, self._extra_counts)):
            x = block(x)
            if block_index and block_index % self.num_blocks_per_residual == 0:
                if resizer is not None:
                    x += resizer(old_x)
                old_x = x
        return x

    def get_num_outputs(self):
        return self._num_output_features


class SlabNet(FullyConv1Resnet):
    """A fully convolutional resnet with constant size"""
    def __init__(
            self,
            num_input_features: int,
            num_neurons: int,
            num_layers: int,
            num_groups: int,
            num_blocks_per_residual=2,
            activation=nn.LeakyReLU,
            do_exclude_first_norm=True,
    ):
        super().__init__(
            num_input_features=num_input_features,
            neuron_counts=[num_neurons] * num_layers,
            num_groups=num_groups,
            num_blocks_per_residual=num_blocks_per_residual,
            activation=activation,
            do_exclude_first_norm=do_exclude_first_norm,
        )


class HeadedSlabNet(SlabNet):
    def __init__(
            self,
            num_input_features: int,
            num_output_features: int,
            num_neurons: int,
            num_layers: int,
            num_groups=32,
            num_blocks_per_residual=2,
            activation=nn.LeakyReLU,
            do_exclude_first_norm=True,
    ):
        super().__init__(
            num_input_features=num_input_features,
            num_neurons=num_neurons,
            num_layers=num_layers,
            num_groups=num_groups,
            num_blocks_per_residual=num_blocks_per_residual,
            activation=activation,
            do_exclude_first_norm=do_exclude_first_norm,
        )
        # could be changed by GroupNorm fixup
        num_slab_outputs = self.get_num_outputs()

        self.head = nn.Conv1d(num_slab_outputs, num_output_features, 1)
        self._num_output_features = num_output_features

    def forward(self, x):
        x = super().forward(x)
        x = self.head(x)
        return x


class ReachAroundTransformer(nn.Module):
    """x_base and x_vocab are inputs to the transformer and a raw x_base is also appended to the output"""
    def __init__(self, blah):
        super().__init__()


def get_fast_linear_attention_encoder(hp: hyperparams.LearningParams, num_trans_input_dims: int, feature_map=None):
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
        n_layers=hp.num_trans_enc_layers,
        n_heads=hp.num_trans_heads,
        query_dimensions=num_trans_input_dims // hp.num_trans_heads,
        value_dimensions=num_trans_input_dims // hp.num_trans_heads,
        feed_forward_dimensions=num_trans_input_dims * hp.num_trans_fc_dim_mult,
        attention_type='linear',
        activation='gelu',
        feature_map=feature_map,
    )

    # Build a transformer with linear attention
    return builder.get()


def get_performer_encoder(hp: hyperparams.LearningParams, num_trans_input_dims: int):
    """https://github.com/lucidrains/performer-pytorch"""
    from performer_pytorch import Performer
    return Performer(
        dim=num_trans_input_dims,
        depth=hp.num_trans_enc_layers,
        heads=hp.num_trans_heads,
        ff_mult=hp.num_trans_fc_dim_mult,
    )


def get_pytorch_transformer_encoder(hp: hyperparams.LearningParams, num_trans_input_dims: int):
    enc_layer = nn.TransformerEncoderLayer(
        d_model=num_trans_input_dims,
        nhead=hp.num_trans_heads,
        dim_feedforward=hp.num_trans_fc_dim_mult * num_trans_input_dims,
        activation='gelu'
    )

    return nn.TransformerEncoder(
        encoder_layer=enc_layer,
        num_layers=hp.num_trans_enc_layers,
    )


def get_simple_ablatable_transformer_encoder(
        hp: hyperparams.LearningParams,
        num_trans_input_dims: int,
        do_drop_k=True,
):
    return ablation.Encoder(
        d_model=num_trans_input_dims,
        num_layers=hp.num_trans_enc_layers,
        num_heads=hp.num_trans_heads,
        d_ff_mult=hp.num_trans_fc_dim_mult,
        do_drop_k=do_drop_k,
    )


CURRENT_EPOCH_NAME = 'current_epoch'
PARAM_COUNT_NAME = 'param_count'
TIME_PERF_NAME = 'time_perf'
TIME_PROCESS_NAME = 'time_process'


class LogCopierCallback(pl.Callback):
    @staticmethod
    def _count_params(pl_module):
        return sum(p.numel() for p in pl_module.parameters() if p.requires_grad)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        d = {k: v.item() for k, v in pl_module.metrics_to_log.items()}
        d[CURRENT_EPOCH_NAME] = trainer.current_epoch
        d[PARAM_COUNT_NAME] = self._count_params(pl_module)
        tune.report(**d)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        d = {k: v.item() for k, v in pl_module.metrics_to_log.items()}
        d[CURRENT_EPOCH_NAME] = trainer.current_epoch
        d[PARAM_COUNT_NAME] = self._count_params(pl_module)
        tune.report(**d)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        d = {k: v.item() for k, v in pl_module.metrics_to_log.items()}
        d[CURRENT_EPOCH_NAME] = trainer.current_epoch
        d[PARAM_COUNT_NAME] = self._count_params(pl_module)
        tune.report(**d)


class CounterTimerCallback(pl.Callback):
    def __init__(self):
        self._train_start_perf = None
        self._train_start_process = None

    @staticmethod
    def _count_params(pl_module):
        return sum(p.numel() for p in pl_module.parameters() if p.requires_grad)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        pl_module.log(name=PARAM_COUNT_NAME, value=self._count_params(pl_module), prog_bar=True, logger=True)
        self._train_start_perf = time.perf_counter()
        self._train_start_process = time.process_time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        pl_module.log(name=PARAM_COUNT_NAME, value=self._count_params(pl_module), prog_bar=True, logger=True)
        pl_module.log(
            name=TIME_PERF_NAME,
            value=time.perf_counter() - self._train_start_perf,
            prog_bar=True,
            logger=True,
        )
        pl_module.log(
            name=TIME_PROCESS_NAME,
            value=time.process_time() - self._train_start_process,
            prog_bar=True,
            logger=True,
        )

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        pl_module.log(name=PARAM_COUNT_NAME, value=self._count_params(pl_module), prog_bar=False, logger=True)

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        pl_module.log(name=PARAM_COUNT_NAME, value=self._count_params(pl_module), prog_bar=False, logger=True)


class MyLightningNeptuneLogger(pl_loggers.NeptuneLogger):
    def __init__(self, hp: hyperparams.LearningParams, version: str = ''):
        super().__init__(
            api_key=utils.get_logger_api_key(),
            project_name=utils.get_neptune_fully_qualified_project_name(hp.project_name),
            close_after_fit=True,
            offline_mode=False,
            experiment_name=f'pl_logger-{hp.get_exp_group_name()}-{version}',
            params=hp.to_dict(),
            tags=hp.experiment_tags,
            upload_source_files=constants.SOURCES_GLOB_STR,
        )
        self.append_tags(hp.experiment_tags)

    @pl.utilities.rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        # params = self._convert_params(params)
        # params = self._flatten_dict(params)
        for key, val in params.items():
            # self.experiment.set_property(f'param__{key}', val)
            self.experiment.set_property(key, val)


def get_pl_logger(hp: hyperparams.LearningParams, tune=None):
    version = 'local' if tune is None else tune.get_trial_id()
    logger = MyLightningNeptuneLogger(hp, version),

    return logger


class BetterAccuracy(pl.metrics.Accuracy):
    """PyTorch Lightning's += lines cause warnings about transferring lots of scalars between cpu / gpu"""
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape,  f"preds.shape: {preds.shape}, target.shape: {target.shape}"
        self.correct = self.correct + torch.sum(preds == target)
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

