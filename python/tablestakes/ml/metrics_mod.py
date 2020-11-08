import abc
import time
from typing import *

import torch
import pytorch_lightning as pl
from chillpill import params

from ray import tune
from ray.tune.logger import Logger as TuneLogger
from ray.tune import result as tune_result

from tablestakes import constants, utils
from tablestakes.ml import hyperparams, torch_mod
from torch.nn import functional as F

CURRENT_EPOCH_NAME = 'current_epoch'
PARAM_COUNT_NAME = 'param_count'
TRAINABLE_PARAM_COUNT_NAME = 'trainable_param_count'
TIME_PERF_NAME = 'train_time_perf'
TIME_PROCESS_NAME = 'train_time_process'


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
        print('about to report')
        tune.report(**d)
        print('done reporting')

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        d = self._get_metrics_dict(trainer, pl_module)
        d[CURRENT_EPOCH_NAME] = trainer.current_epoch
        print('about to report')
        tune.report(**d)
        print('done reporting')

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        d = self._get_metrics_dict(trainer, pl_module)
        d[CURRENT_EPOCH_NAME] = trainer.current_epoch
        print('about to report')
        tune.report(**d)
        print('done reporting')


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
    Y_VALUE_TO_IGNORE = constants.Y_VALUE_TO_IGNORE

    """PyTorch Lightning's += lines cause warnings about transferring lots of scalars between cpu / gpu"""
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape,  f"preds.shape: {preds.shape}, target.shape: {target.shape}"
        self.correct = self.correct + torch.sum(preds.eq(target))
        self.total = self.total + target.numel() - target.eq(self.Y_VALUE_TO_IGNORE).sum()


# # ref: https://community.neptune.ai/t/neptune-and-hyperparameter-search-with-tune/567/3
# class TuneNeptuneLogger(TuneLogger):
#     """Neptune logger.
#     """
#
#     def _init(self):
#         from neptune.sessions import Session
#
#         hp = hyperparams.LearningParams.from_dict(self.config)
#         project_name = utils.get_neptune_fully_qualified_project_name(hp.project_name)
#         experiment_name = f'tune_logger-{hp.get_exp_group_name()}-{tune.get_trial_id()}'
#
#         project = Session().get_project(project_name)
#
#         self.exp = project.create_experiment(
#             name=experiment_name,
#             params=self.config,
#             tags=hp.experiment_tags,
#             upload_source_files=constants.SOURCES_GLOB_STR,
#         )
#
#     def on_result(self, result):
#         for name, value in result.items():
#             if isinstance(value, float):
#                 self.exp.log_metric(name, x=result.get(tune_result.TRAINING_ITERATION), y=value)
#             elif isinstance(value, int):
#                 self.exp.log_metric(name, x=result.get(tune_result.TRAINING_ITERATION), y=value)
#             elif isinstance(value, str):
#                 self.exp.log_text(name, x=result.get(tune_result.TRAINING_ITERATION), y=value)
#             else:
#                 continue
#
#     def close(self):
#         self.exp.stop()


Y_VALUE_TO_IGNORE = constants.Y_VALUE_TO_IGNORE


class ClassificationMetricsTracker(torch_mod.Parameterized, abc.ABC):
    TRAIN_PHASE_NAME = 'train'
    VALID_PHASE_NAME = 'valid'
    TEST_PHASE_NAME = 'test'

    PHASE_NAMES = [
        TRAIN_PHASE_NAME,
        VALID_PHASE_NAME,
        TEST_PHASE_NAME,
    ]

    LOSS_VAL_NAME = 'loss'
    TOTAL_NAME = 'total'

    METRICS = {
        'acc': BetterAccuracy(),
    }

    class MetricParams(params.ParameterSet):
        num_steps_per_histogram_log = 10
        num_steps_per_metric_log = 10
        output_dir = 'output'

    def get_params(self) -> MetricParams:
        return self.MetricParams()

    def __init__(self, hp: MetricParams):
        super().__init__()
        self.hp = hp
        self.pl_module = None

        self.metrics_to_log_for_tune = {}

    def set_pl_module(self, pl_module: pl.LightningModule):
        self.pl_module = pl_module

    @staticmethod
    def _get_phase_name(phase_name: str, metric_name: str, output_name: Optional[str] = None):
        out = f'{phase_name}'
        out += f'_{metric_name}'
        if output_name is not None:
            out += f'_{output_name}'
        return out

    # @classmethod
    # def get_valid_metric_name(cls, metric_name: str, output_name: Optional[str] = None):
    #     return cls._get_phase_name(cls.VALID_PHASE_NAME, metric_name, output_name)
    #
    # @classmethod
    # def get_train_metric_name(cls, metric_name: str, output_name: Optional[str] = None):
    #     return cls._get_phase_name(cls.TRAIN_PHASE_NAME, metric_name, output_name)

    @classmethod
    def get_all_metric_names_for_phase(cls, phase_name: str):
        metrics_names = [m for m in cls.METRICS.keys()] + [cls.LOSS_VAL_NAME]
        output_names = constants.Y_BASE_NAMES

        out = []
        for metric_name in metrics_names:
            for output_name in output_names:
                out.append(cls._get_phase_name(phase_name, metric_name, output_name))
        out.append(cls._get_phase_name(phase_name, cls.LOSS_VAL_NAME, cls.TOTAL_NAME))

        return out

    def log_losses_and_metrics(self, phase_name, loss, losses, y_hats_dict, ys_dict, prog_bar=False):
        output_names = ys_dict.keys()

        on_epoch = None

        d = {}

        total_loss_name = self._get_phase_name(phase_name, 'loss', 'total')

        # LOG
        self.pl_module.log(total_loss_name, loss, prog_bar=False, on_epoch=on_epoch)
        d[total_loss_name] = loss

        for output_name, current_loss in zip(output_names, losses):
            full_metric_name = self._get_phase_name(phase_name, 'loss', output_name)
            # LOG
            self.pl_module.log(full_metric_name, current_loss, prog_bar=False, on_epoch=on_epoch)

            d[full_metric_name] = current_loss

        for metric_name, metric in self.METRICS.items():
            for output_name in output_names:
                y_hat = y_hats_dict[output_name]
                y = ys_dict[output_name]
                full_metric_name = self._get_phase_name(phase_name, metric_name, output_name)
                y_hat = y_hat.argmax(dim=2)
                metric_value = metric(y_hat, y)

                # LOG
                self.pl_module.log(full_metric_name, metric_value, prog_bar=prog_bar, on_epoch=on_epoch)
                d[full_metric_name] = metric_value

        opt = self.pl_module.trainer.optimizers[0]
        lrs = [float(param_group['lr']) for param_group in opt.param_groups]
        assert(all([lr == lrs[0] for lr in lrs]))
        lr = lrs[0]
        # LOG
        self.pl_module.log('lrs_opt', lr, prog_bar=False, on_epoch=on_epoch)
        d['lrs_op'] = lr

        self.metrics_to_log_for_tune = d

    def on_pretrain_routine_start(self) -> None:
        if self.pl_module.logger is None:
            return
        self.pl_module.logger.log_hyperparams(self.pl_module.hp.to_dict())

    def on_after_backward(self):
        if self.pl_module.logger is None:
            return

        if self.hp.num_steps_per_histogram_log \
                and (self.pl_module.global_step + 1) % self.hp.num_steps_per_histogram_log == 0:
            d = {
                f'gn/{k.replace("grad_1.0_norm_", "")}': v
                for k, v in self.pl_module.logger._flatten_dict(self.pl_module.grad_norm(1)).items()
            }

            self.pl_module.logger.log_metrics(
                metrics=d,
                step=self.pl_module.global_step,
            )

    def inner_forward_step(self, batch, batch_idx):
        xs_dict, ys_dict, meta = batch
        y_hats_dict = self.pl_module(batch)

        losses = torch.stack([
            F.cross_entropy(
                input=y_hat.permute(0, 2, 1),
                target=y,
                ignore_index=Y_VALUE_TO_IGNORE,
            )
            for y, y_hat in zip(ys_dict.values(), y_hats_dict.values())
        ])

        loss = losses.sum()

        return xs_dict, ys_dict, y_hats_dict, losses, loss

    def training_step(self, batch, batch_idx):
        xs_dict, ys_dict, y_hats_dict, losses, loss = self.inner_forward_step(batch, batch_idx)
        self.log_losses_and_metrics(self.TRAIN_PHASE_NAME, loss, losses, y_hats_dict, ys_dict, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xs_dict, ys_dict, y_hats_dict, losses, loss = self.inner_forward_step(batch, batch_idx)
        self.log_losses_and_metrics(self.VALID_PHASE_NAME, loss, losses, y_hats_dict, ys_dict)

    def test_step(self, batch, batch_idx):
        xs_dict, ys_dict, y_hats_dict, losses, loss = self.inner_forward_step(batch, batch_idx)
        self.log_losses_and_metrics(self.TEST_PHASE_NAME, loss, losses, y_hats_dict, ys_dict)
