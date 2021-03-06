from typing import *

import pytorch_lightning as pl
from torch import optim

from tablestakes.ml2.factored import trunks_mod

from chillpill import params


class OptParams(trunks_mod.BuilderParams):
    num_epochs: int = 10
    lr: float = 0.001
    min_lr: float = 1e-6
    patience: int = 10
    search_metric: str = 'valid/loss'
    search_mode: str = 'min'
    lr_reduction_factor: float = 0.5

    def build(self) -> 'OptimizersMaker':
        return OptimizersMaker(hp=self)


class OptimizersMaker:
    def __init__(self, hp: OptParams):
        super().__init__()
        self.hp = hp
        self.pl_module = None

    def set_pl_module(self, pl_module: pl.LightningModule):
        self.pl_module = pl_module

    # noinspection PyProtectedMember
    def get_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler._LRScheduler]]:
        optimizer = optim.AdamW(self.pl_module.parameters(), lr=self.hp.lr)

        # coser = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.neck_hp.opt.patience // 2,
        #     eta_min=self.neck_hp.opt.min_lr,
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


