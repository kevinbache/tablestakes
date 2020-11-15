from argparse import Namespace
from dataclasses import dataclass, field
import glob
import time
from typing import *

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from ray import tune

from tablestakes import constants, utils

CURRENT_EPOCH_NAME = 'current_epoch'
PARAM_COUNT_NAME = 'param_count'
TRAINABLE_PARAM_COUNT_NAME = 'trainable_param_count'
TIME_PERF_NAME = 'train_time_perf'
TIME_PROCESS_NAME = 'train_time_process'


@dataclass
class LoggingParams(utils.DataclassPlus):
    num_steps_per_histogram_log: int = 10
    num_steps_per_metric_log: int = 10
    output_dir: str = 'output'


class LogCopierCallback(pl.Callback):
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
        assert(all([lr == lrs[0] for lr in lrs]))
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
        # preds = preds.argmax(dim=1)
        # target = target.squeeze(1)
        assert preds.shape == target.shape,  f"preds.shape: {preds.shape}, target.shape: {target.shape}"
        self.correct = self.correct + torch.sum(preds.eq(target))
        self.total = self.total + target.numel() - target.eq(self.Y_VALUE_TO_IGNORE).sum()


@dataclass
class ExperimentParams(utils.DataclassPlus):
    project_name: str = 'my_project'
    experiment_name: str = 'my_experiment'
    experiment_tags: str = field(default_factory=lambda: ['testing'])
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
