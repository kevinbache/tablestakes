import copy
from dataclasses import dataclass
from typing import *

import pytorch_lightning as pl
import torch

from tablestakes import constants, utils
from tablestakes.ml2.data import datapoints, data_module
from tablestakes.ml2.factored import opt_mod, head_mod, logs_mod

Y_VALUE_TO_IGNORE = constants.Y_VALUE_TO_IGNORE


@dataclass
class FactoredParams(utils.DataclassPlus):
    opt: opt_mod.OptParams
    logs: logs_mod.LoggingParams
    exp: logs_mod.ExperimentParams

    head: Union[head_mod.HeadParams, head_mod.WeightedHeadParams]

    data: data_module.DataParams


class FactoredLightningModule(pl.LightningModule, head_mod.LossMetrics):
    """A Lightning Module which has been factored into distinct parts."""

    def __init__(
            self,
            hp: FactoredParams,
            opt: opt_mod.OptimizersMaker,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hp = copy.deepcopy(hp)
        self.opt = opt
        self.opt.set_pl_module(pl_module=self)

        # set me in subclasses
        self.head = None

    ############
    # TRAINING #
    ############
    def forward_plus_lossmetrics(
            self,
            batch: datapoints.XYMetaDatapoint,
            batch_idx: int,
    ) -> Dict[str, Any]:
        y_hats = self(batch.x)
        return self.head.loss_metrics(y_hats, batch.y, batch.meta)

    def training_step(self, batch, batch_idx):
        d = self.forward_plus_lossmetrics(batch, batch_idx)
        loss = d[self.LOSS_NAME]
        self.log_lossmetrics_dict(utils.Phase.train, d)
        return loss

    def validation_step(self, batch, batch_idx):
        d = self.forward_plus_lossmetrics(batch, batch_idx)
        self.log_lossmetrics_dict(utils.Phase.valid, d)

    def test_step(self, batch, batch_idx):
        d = self.forward_plus_lossmetrics(batch, batch_idx)
        self.log_lossmetrics_dict(utils.Phase.test, d)

    def log_lossmetrics_dict(self, phase: utils.Phase, d: Dict[str, Any]):
        # if phase == utils.Phase.train:
        #     self.log(name=self.LOSS_NAME, value=d[self.LOSS_NAME], prog_bar=True)
        d = {phase.name: d}
        d = utils.sanitize_tensors(d)
        d = utils.flatten_dict(d, delimiter='/')
        prog_bar = phase == utils.Phase.train
        return self.log_dict(d, prog_bar=prog_bar)

    #######
    # OPT #
    #######
    def configure_optimizers(self):
        return self.opt.get_optimizers()

    def on_pretrain_routine_start(self) -> None:
        if self.logger is None:
            return
        self.logger.log_hyperparams(self.hp.to_dict())

    def on_after_backward(self):
        if self.logger is None:
            return

        if self.hp.logs.num_steps_per_histogram_log \
                and (self.global_step + 1) % self.hp.logs.num_steps_per_histogram_log == 0:
            d = {
                f'gn/{k.replace("grad_1.0_norm_", "")}': v
                for k, v in self.logger._flatten_dict(self.grad_norm(1)).items()
            }

            self.logger.log_metrics(metrics=d, step=self.global_step)


class TablestakesBertTransConvTClass(FactoredLightningModule):
    def __init__(self, hp: FactoredParams, opt: opt_mod.OptimizersMaker):
        super().__init__(hp, opt)

    @classmethod
    def from_hp(cls, hp: FactoredParams):
        return cls(
            hp=hp,
            opt_maker=opt_mod.OptimizersMaker(hp.opt),
        )