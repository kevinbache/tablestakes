import abc
from typing import *

from torch import optim, nn
import pytorch_lightning as pl

from chillpill import params

from tablestakes import constants
from tablestakes.ml import torch_mod, data, metrics_mod


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
        # schedulers = [reducer, coser]
        schedulers = [reducer]

        return optimizers, schedulers


class Head(nn.Module):
    """
    A head
        takes in y_hat and y and spits out loss
        and logs

    OR
        it takes the x' and transforms it into y_hat then takes in y and calculates loss

        takes in y_hat
            good definition of difference
            but then a head is just a logged loss
        a logged loss is a tracked loss
            it doesn't log it tracks metrics
            cause every time you run classification you want some basic metrics:
                confusion matrix
                every time you use region start/end head you're gonna want the same metrics
                MetricSet
                    is basically a list of metrics but can share computation?
                    probably not important.  just do a list of metrics.
                What kinds of things are you going ot be interested in
                    s2s - save the datapoint.  output sequences.
                        how well each sequence matches the original
                            is it a superset
                            is it a subset
                            venn diagram of tokens
                    token class - save a datapoint.  repaint the original docs
                        stats on number of tokens in each class
                        absolute and proportional confusion matrices
                        save the sequences that were generated
                        types of each token
                            how do i want to deal with token types?

            it takes in a y_hat and a y and it spits out an output object which includes loss and other metrics of
                interest

        a head meanwhile might produce the actual output.  like lm heads' .generate

    """

    def loss(self, y_hat, y):
        return nn.CrossEntropyLoss(y_hat, y)


class FactoredLightningModule(pl.LightningModule, torch_mod.Parameterized[torch_mod.PS], abc.ABC):
    """A Lightning Module which has been factored into distinct parts."""
    class FactoredParams(params.ParameterSet):
        data = data.TablestakesDataModule.DataParams()
        opt = OptimizersMaker.OptParams()
        head = Head.HeadParams()
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
