from typing import *

from tablestakes.ml import metrics_mod
from torch import optim
import pytorch_lightning as pl

from chillpill import params

from tablestakes import constants
from tablestakes.ml import torch_mod, data


Y_VALUE_TO_IGNORE = constants.Y_VALUE_TO_IGNORE


class OptimizersMaker(torch_mod.Parameterized['OptimizationMaker.OptParams']):
    class OptParams(params.ParameterSet):
        num_epochs = 10
        batch_size = 32
        lr = 0.001
        min_lr = 1e-6
        patience = 10
        search_metric = 'valid_loss_total'
        search_mode = 'max'
        lr_reduction_factor = 0.5

    def get_params(self) -> OptParams:
        return self.OptParams()

    def __init__(self, hp: OptParams):
        super().__init__()
        self.hp = hp
        # self.pl_module = None

    def set_pl_module(self, pl_module: pl.LightningModule):
        self.pl_module = pl_module

    # noinspection PyProtectedMember
    def get_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler._LRScheduler]]:
        optimizer = optim.AdamW(self.pl_module.parameters(), lr=self.hp.lr)

        # coser = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.hp.opt.patience // 2,
        #     eta_min=self.hp.opt.min_lr,
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
        # schedulers = [x_reducer_name, coser]
        schedulers = [reducer]

        return optimizers, schedulers


class FactoredLightningModule(pl.LightningModule, torch_mod.Parameterized[torch_mod.PS]):
    """A Lightning Module which has been factored into distinct parts."""
    class FactoredParams(params.ParameterSet):
        data = data.TablestakesDataModule.DataParams()
        opt = OptimizersMaker.OptParams()
        metrics = metrics_mod.ClassificationMetricsTracker.MetricParams()
        exp = torch_mod.ExperimentParams()
        Tune = None

    # TODO: gets params?
    def __init__(
            self,
            hp: FactoredParams,
            metrics_tracker: metrics_mod.ClassificationMetricsTracker,
            opt: OptimizersMaker,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hp = hp
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

    @staticmethod
    def get_metrics_tracker_class():
        return metrics_mod.ClassificationMetricsTracker

    @classmethod
    def from_hp(cls, hp: FactoredParams):
        return cls(
            hp=hp,
            data_module= data.TablestakesDataModule(hp.data),
            metrics_tracker=cls.get_metrics_tracker_class()(hp.metrics),
            opt=OptimizersMaker(hp.opt),
        )
