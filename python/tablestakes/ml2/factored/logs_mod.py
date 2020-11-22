from argparse import Namespace
import glob
import time
from typing import *


import torch
from pytorch_lightning.metrics.utils import _input_format_classification
from tablestakes.ml2.data import datapoints
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from ray import tune

from tablestakes import constants, utils

from chillpill import params

CURRENT_EPOCH_NAME = 'current_epoch'
PARAM_COUNT_NAME = 'param_count'
TRAINABLE_PARAM_COUNT_NAME = 'trainable_param_count'
TIME_PERF_NAME = 'train_time_perf'
TIME_PROCESS_NAME = 'train_time_process'


class LoggingParams(params.ParameterSet):
    num_steps_per_histogram_log: int = 10
    num_steps_per_metric_log: int = 10
    output_dir: str = 'output'


class TuneLogCopierCallback(pl.Callback):
    @staticmethod
    def _get_metrics_dict(trainer, pl_module):
        d = trainer.logged_metrics
        d.update(trainer.callback_metrics)
        d.update(trainer.progress_bar_metrics)
        # d.update(pl_module.metrics_tracker.metrics_to_log_for_tune)

        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.item()
        return d

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        d = self._get_metrics_dict(trainer, pl_module)
        d[CURRENT_EPOCH_NAME] = trainer.current_epoch
        tune.report(**d)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        d = self._get_metrics_dict(trainer, pl_module)
        d[CURRENT_EPOCH_NAME] = trainer.current_epoch
        tune.report(**d)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        d = self._get_metrics_dict(trainer, pl_module)
        d[CURRENT_EPOCH_NAME] = trainer.current_epoch
        tune.report(**d)


class VocabLengthCallback(pl.Callback):
    VOCAB_PAD_VALUE = utils.VOCAB_PAD_VALUE

    def __init__(self, doc_dim=1):
        self._valid_counts = []
        self._invalid_counts = []

        self.doc_dim = doc_dim

    def _reset(self):
        self._valid_counts = []
        self._invalid_counts = []

    def on_train_epoch_start(self, trainer, pl_module):
        self._reset()

    def on_train_batch_start(self, trainer, pl_module: pl.LightningModule, batch: datapoints.XYMetaDatapoint, batch_idx, dataloader_idx):
        vocab = batch.x.vocab

        is_invalid = (vocab == self.VOCAB_PAD_VALUE).int()
        is_valid = 1 - is_invalid

        valid_counts = is_valid.sum(dim=self.doc_dim)
        invalid_counts = is_invalid.sum(dim=self.doc_dim)
        self._valid_counts.append(valid_counts)
        self._invalid_counts.append(invalid_counts)

    def on_train_epoch_end(self, trainer, pl_module, outputs):

        valid_counts = torch.cat(self._valid_counts)
        invalid_counts = torch.cat(self._invalid_counts)

        batch_lens = valid_counts + invalid_counts
        num_total = batch_lens.sum()

        p_valids = valid_counts / batch_lens

        trainer.logger.log_metrics(
            metrics={
                'p_tokens_is_valid': valid_counts.sum() / num_total,
                'num_tokens_total': num_total,

                'p_valids_min': p_valids.min(),
                'p_valids_median': p_valids.median(),
                'p_valids_max': p_valids.max(),

                'min_num_valid_tokens_per_doc': valid_counts.min(),
                'median_num_valid_tokens_per_doc': valid_counts.median(),
                'max_num_valid_tokens_per_doc': valid_counts.max(),

                'min_num_wasted_tokens_per_doc': invalid_counts.min(),
                'median_num_wasted_tokens_per_doc': invalid_counts.median(),
                'max_num_wasted_tokens_per_doc': invalid_counts.max(),
            },
            step=trainer.global_step,
        )


class CounterTimerLrCallback(pl.Callback):
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

    def _get_lr_dict(self, pl_module) -> Dict[str, float]:
        lrs = [float(param_group['lr']) for param_group in pl_module.optimizers().param_groups]
        assert (all([lr == lrs[0] for lr in lrs]))
        lr = lrs[0]

        return {
            'lrs_opt': lr,
        }

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        d = self._get_lr_dict(pl_module)
        trainer.logger.log_metrics(d, step=trainer.global_step)

        self._train_start_perf = time.perf_counter()
        self._train_start_process = time.process_time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        if trainer.logger is None:
            return

        d = self._get_lr_dict(pl_module)
        d[TIME_PERF_NAME] = time.process_time() - self._train_start_process
        d[TIME_PROCESS_NAME] = time.process_time() - self._train_start_process

        trainer.logger.log_metrics(d, step=trainer.global_step)


class BetterAccuracy(pl.metrics.Accuracy):
    Y_VALUE_TO_IGNORE = constants.Y_VALUE_TO_IGNORE

    """PyTorch Lightning's += lines cause warnings about transferring lots of scalars between cpu / gpu"""

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = _input_format_classification(preds, target, self.threshold)
        assert preds.shape == target.shape

        try:
            self.correct = self.correct + torch.sum(preds.eq(target))
        except RuntimeError as e:
            print(f'preds.shape: {preds.shape}, target.shape: {target.shape}')
            raise e

        self.total = self.total + target.numel() - target.eq(self.Y_VALUE_TO_IGNORE).sum()


class SigmoidBetterAccuracy(pl.metrics.Accuracy):
    Y_VALUE_TO_IGNORE = constants.Y_VALUE_TO_IGNORE

    def __init__(self):
        super().__init__(
            threshold=0.5,
            compute_on_step=True,
            dist_sync_on_step=False,
            process_group=None,
            dist_sync_fn=None,
        )
        self.s = nn.Sigmoid()

    """PyTorch Lightning's += lines cause warnings about transferring lots of scalars between cpu / gpu"""

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds = preds.argmax(dim=1)
        # target = target.squeeze(1)
        preds = self.s(preds)
        assert preds.shape == target.shape, f"preds.shape: {preds.shape}, target.shape: {target.shape}"
        self.correct = self.correct + torch.sum(preds.eq(target))
        self.total = self.total + target.numel() - target.eq(self.Y_VALUE_TO_IGNORE).sum()


class ExperimentParams(params.ParameterSet):
    project_name: str = 'my_project'
    experiment_name: str = 'my_experiment'
    experiment_tags: str = ('testing', )
    sources_glob_str: str = '*.py'
    offline_mode: bool = False

    def get_project_exp_name(self):
        return f'{self.project_name}-{self.experiment_name}'


# noinspection PyProtectedMember
class MyLightningNeptuneLogger(pl_loggers.NeptuneLogger):
    def __init__(self, hp: ExperimentParams, version: str = '', offline_mode=False):
        source_files = glob.glob(str(hp.sources_glob_str), recursive=True)

        self.offline_mode = offline_mode

        super().__init__(
            api_key=utils.get_logger_api_key(),
            project_name=utils.get_neptune_fully_qualified_project_name(hp.project_name),
            close_after_fit=True,
            offline_mode=offline_mode,
            experiment_name=f'pl_log-{version}',
            params=hp.to_dict(),
            tags=hp.experiment_tags,
            upload_source_files=source_files,
        )
        self.append_tags(hp.experiment_tags)

    @pl.utilities.rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        params = self._sanitize_params(params)
        return self.set_properties(params)

    def set_properties(self, new_properties: Dict):
        if self.offline_mode:
            import warnings
            warnings.warn('log_mods.MyNeptuneLogger skipping set_properties')
            return
        else:
            properties = self.experiment._backend.get_experiment(self.experiment.internal_id).properties
            properties = {p.key: p.value for p in properties}
            properties.update({k: str(v) for k, v in new_properties.items()})
            return self.experiment._backend.update_experiment(
                experiment=self.experiment,
                properties=properties,
            )


def get_pl_logger(hp: ExperimentParams, tune=None):
    version = 'local' if tune is None else tune.get_trial_id()
    logger = MyLightningNeptuneLogger(hp=hp, version=version, offline_mode=hp.offline_mode),

    return logger
