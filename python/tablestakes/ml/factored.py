import abc
from typing import *

import torch
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl

from chillpill import params

from tablestakes import constants
from tablestakes.ml import torch_helpers, param_torch_mods


# noinspection PyProtectedMember
from torch.utils.data import DataLoader


Y_VALUE_TO_IGNORE = constants.Y_VALUE_TO_IGNORE


class OptimizersMaker(param_torch_mods.Parametrized['OptimizationMaker.OptParams']):
    class OptParams(params.ParameterSet):
        lr = 0.001
        min_lr = 1e-6
        patience = 10
        search_mode = 'max'
        lr_reduction_factor = 0.5
        batch_size = 32

    def __init__(self, hp: OptParams):
        super().__init__(hp)
        self.pl_module = None

    def set_pl_module(self, pl_module: pl.LightningModule):
        self.pl_module = pl_module

    def get_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler._LRScheduler]]:
        optimizer = optim.AdamW(self.pl_module.parameters(), lr=self.hp.lr)

        # coser = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.hp.patience // 2,
        #     eta_min=self.hp.min_lr,
        #     verbose=True,
        # )

        reducer = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode=self.hp.search_mode,
                factor=self.hp.lr_reduction_factor,
                patience=self.hp.patience,
                min_lr=self.hp.min_lr,
                verbose=True,
            ),
            'monitor': self.hp.search_metric,
            'interval': 'epoch',
            'frequency': 1
        }

        optimizers = [optimizer]
        # schedulers = [reducer, coser]
        schedulers = [reducer]

        return optimizers, schedulers


class MetricTracker(param_torch_mods.Parametrized):

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
        'acc': torch_helpers.BetterAccuracy(),
    }

    class Params(params.ParameterSet):
        num_steps_per_histogram_log = 10
        num_steps_per_metric_log = 10
        output_dir = 'output'

    def __init__(self, hp: Params):
        super().__init__(hp)
        self.hp = hp
        self.pl_module = None

    def set_pl_module(self, pl_module: pl.LightningModule):
        self.pl_module = pl_module

    @staticmethod
    def _get_phase_name(phase_name: str, metric_name: str, output_name: Optional[str] = None):
        out = f'{phase_name}'
        out += f'_{metric_name}'
        if output_name is not None:
            out += f'_{output_name}'
        return out

    @classmethod
    def get_valid_metric_name(cls, metric_name: str, output_name: Optional[str] = None):
        return cls._get_phase_name(cls.VALID_PHASE_NAME, metric_name, output_name)

    @classmethod
    def get_train_metric_name(cls, metric_name: str, output_name: Optional[str] = None):
        return cls._get_phase_name(cls.TRAIN_PHASE_NAME, metric_name, output_name)

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

        self.pl_module.log(total_loss_name, loss, prog_bar=False, on_epoch=on_epoch)
        d[total_loss_name] = loss

        for output_name, current_loss in zip(output_names, losses):
            full_metric_name = self._get_phase_name(phase_name, 'loss', output_name)
            self.pl_module.log(full_metric_name, current_loss, prog_bar=False, on_epoch=on_epoch)

            d[full_metric_name] = current_loss

        for metric_name, metric in self.METRICS.items():
            for output_name in output_names:
                y_hat = y_hats_dict[output_name]
                y = ys_dict[output_name]
                full_metric_name = self._get_phase_name(phase_name, metric_name, output_name)
                y_hat = y_hat.argmax(dim=2)
                metric_value = metric(y_hat, y)

                self.pl_module.log(full_metric_name, metric_value, prog_bar=prog_bar, on_epoch=on_epoch)
                d[full_metric_name] = metric_value

        opt = self.pl_module.trainer.optimizers[0]
        lrs = [float(param_group['lr']) for param_group in opt.param_groups]
        assert(all([lr == lrs[0] for lr in lrs]))
        lr = lrs[0]
        self.pl_module.log('lrs_opt', lr, prog_bar=False, on_epoch=on_epoch)

        # d['lrs_opt'] = lr
        # self.metrics_to_log_for_tune = d

    def on_pretrain_routine_start(self) -> None:
        self.pl_module.logger.log_hyperparams(self.pl_module.hp.to_dict())

    def on_after_backward(self):
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

    def inner_forward_step(self, batch):
        xs_dict, ys_dict = batch
        y_hats_dict = self.pl_module(**xs_dict)

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
        xs_dict, ys_dict, y_hats_dict, losses, loss = self.inner_forward_step(batch)
        self.log_losses_and_metrics(self.TRAIN_PHASE_NAME, loss, losses, y_hats_dict, ys_dict, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xs_dict, ys_dict, y_hats_dict, losses, loss = self.inner_forward_step(batch)
        self.log_losses_and_metrics(self.VALID_PHASE_NAME, loss, losses, y_hats_dict, ys_dict)

    def test_step(self, batch, batch_idx):
        xs_dict, ys_dict, y_hats_dict, losses, loss = self.inner_forward_step(batch)
        self.log_losses_and_metrics(self.TEST_PHASE_NAME, loss, losses, y_hats_dict, ys_dict)


class FactoredLightningModule(pl.LightningModule, param_torch_mods.Parametrized[param_torch_mods.PS]):
    """A Lightning Module which has been factored into distinct parts."""

    def __init__(
            self,
            hp: param_torch_mods.PS,
            # dm: pl.LightningDataModule,
            metrics_tracker: MetricTracker,
            opt: OptimizersMaker,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hp = hp
        # self.dm = dm
        self.metrics_tracker = metrics_tracker
        self.opt = opt

        self.metrics_tracker.set_pl_module(pl_module=self)
        self.opt.set_pl_module(pl_module=self)

    def training_step(self, batch, batch_idx):
        return self.metrics_tracker.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.metrics_tracker.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.metrics_tracker.test_step(batch, batch_idx)

    def configure_optimizers(self):
        return self.opt.get_optimizers()

    def on_pretrain_routine_start(self) -> None:
        self.metrics_tracker.on_pretrain_routine_start()

    def on_after_backward(self):
        self.metrics_tracker.on_after_backward()
