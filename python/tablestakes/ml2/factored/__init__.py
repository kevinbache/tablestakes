import copy
from dataclasses import dataclass

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

    data: data_module.DataParams


class FactoredLightningModule(pl.LightningModule):
    """A Lightning Module which has been factored into distinct parts."""

    def __init__(
            self,
            hp: FactoredParams,
            opt_maker: opt_mod.OptimizersMaker,
            head: head_mod.Head,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hp = copy.deepcopy(hp)
        self.opt = opt_maker
        self.opt.set_pl_module(pl_module=self)

        self.head = head

    ############
    # TRAINING #
    ############
    def forward_plus_lossmetrics(self, batch: datapoints.XYMetaDatapoint, batch_idx: int) -> datapoints.LossYDatapoint:
        # model forward
        y_hats = self(batch.x)

        # losses
        # head_to_lossmetrics = {}
        # for head_name, head in self.head.items():
        #     if head_name not in y_hats:
        #         raise ValueError(f"Tried to run a head named {head_name} but y_hats only has keys: {y_hats.keys()}")
        #     y_hat = y_hats[head_name]
        #     y = batch.y[head_name]
        #     value = head.loss_metrics(y_hat, y, batch.meta)
        #     head_to_lossmetrics[head_name] = value

        # the output will be the same type as the y datapoint
        # datapoint_cls = batch.get_y_class()
        #
        # # TODO This is saying here that the datapoint class is going to take care of combining the head losses into weighted combo
        # lossmetrics_datapoint = batch.from_dict()

        head(y_hats)

    def postproc_lossmetrics_dp(self, lossmetrics_datapoint:  datapoints.LossYDatapoint) -> datapoints.LossYDatapoint:
        return lossmetrics_datapoint

    def training_step(self, batch, batch_idx):
        lossmetrics_datapoint = self.forward_plus_lossmetrics(batch, batch_idx)
        lossmetrics_datapoint = self.postproc_lossmetrics_dp(lossmetrics_datapoint)
        self.log_lossmetrics_datapoint(utils.Phase.train, lossmetrics_datapoint)
        return lossmetrics_datapoint.loss

    def validation_step(self, batch, batch_idx):
        lossmetrics_datapoint = self.forward_plus_lossmetrics(batch, batch_idx)
        lossmetrics_datapoint = self.postproc_lossmetrics_dp(lossmetrics_datapoint)
        self.log_lossmetrics_datapoint(utils.Phase.valid, lossmetrics_datapoint)

    def test_step(self, batch, batch_idx):
        lossmetrics_datapoint = self.forward_plus_lossmetrics(batch, batch_idx)
        lossmetrics_datapoint = self.postproc_lossmetrics_dp(lossmetrics_datapoint)
        self.log_lossmetrics_datapoint(utils.Phase.test, lossmetrics_datapoint)

    #######
    # OPT #
    #######
    def configure_optimizers(self):
        return self.opt.get_optimizers()

    ###########
    # LOGGING #
    ###########
    def log_lossmetrics_datapoint(self, phase: utils.Phase, dp: datapoints.YDatapoint):
        return self.log_dict({phase.name: dp.to_dict()})

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
    def __init__(self, hp: FactoredParams, opt_maker: opt_mod.OptimizersMaker):
        super().__init__(hp, opt_maker)

    @classmethod
    def from_hp(cls, hp: FactoredParams):
        return cls(
            hp=hp,
            opt_maker=opt_mod.OptimizersMaker(hp.opt),
        )
