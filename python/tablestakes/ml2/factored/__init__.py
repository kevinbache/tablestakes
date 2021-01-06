import copy
from dataclasses import dataclass
from typing import *

import pytorch_lightning as pl

from chillpill import params

from tablestakes import constants, utils
from tablestakes.ml2.data import datapoints, data_module
from tablestakes.ml2.factored import opt_mod, head_mod, logs_mod, trunks_mod


Y_VALUE_TO_IGNORE = constants.Y_VALUE_TO_IGNORE


@dataclass
class FactoredParams(trunks_mod.BuilderParams):
    opt: opt_mod.OptParams
    logs: logs_mod.LoggingParams
    exp: logs_mod.ExperimentParams

    head: Union[head_mod.HeadParams, head_mod.WeightedHeadParams]

    data: data_module.DataParams

    def build(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't use this directly; instantiate one of its subclasses instead.")



class FactoredLightningModule(pl.LightningModule, head_mod.LossMetrics):
    """A Lightning Module which has been factored into distinct parts."""

    def __init__(
            self,
            hp: FactoredParams,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hp = copy.deepcopy(hp)
        self.opt = hp.opt.build()
        self.opt.set_pl_module(pl_module=self)

        # set me in subclasses
        self.neckhead = None

    ############
    # TRAINING #
    ############
    def forward_plus_lossmetrics(
            self,
            batch: datapoints.XYMetaDatapoint,
            batch_idx: int,
    ) -> Dict[str, Union[float, Dict[str, Any]]]:
        # import torch.autograd.profiler as profiler
        # with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
        #     y_hats_for_loss, y_hats_for_pred = self(batch.x)
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=1000))

        # from pytorch_memlab import LineProfiler
        # with LineProfiler(self.__call__) as prof:
        #     y_hats_for_loss, y_hats_for_pred = self(batch.x)
        # print(prof.display())
        y_hats_for_loss, y_hats_for_pred = self(batch.x)
        return self.neckhead.loss_metrics(y_hats_for_loss, y_hats_for_pred, batch.y, batch.meta)

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

    def log_lossmetrics_dict(self, phase: utils.Phase, d: Dict[str, Any], do_log_to_progbar=None) -> None:
        d = {phase.name: d}
        d = utils.detach_tensors(d)
        d = utils.flatten_dict(d, delimiter='/')
        if do_log_to_progbar is None:
            do_log_to_progbar = phase == utils.Phase.train

        self.log_dict(d, prog_bar=do_log_to_progbar)

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


