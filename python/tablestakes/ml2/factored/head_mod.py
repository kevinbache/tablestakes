import abc
from dataclasses import dataclass
from typing import *

import numpy as np

import torch
from torch import nn
import pytorch_lightning as pl

from tablestakes import constants, utils
from tablestakes.ml2.factored import trunks, logs_mod
from tablestakes.ml2.data import datapoints

Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class SaveYhatsMetric(pl.metrics.Metric):
    def __init__(self, p_save=0.01):
        super().__init__()
        self.p_save = p_save
        self.preds = []

    # noinspection PyMethodOverriding
    def update(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
        num_data = y.shape[0]
        do_save = torch.rand((num_data,)) < self.p_save
        self.preds.append(y_hat[do_save])

    # noinspection PyMethodOverriding
    def compute(self, y_hat: torch.Tensor, y: torch.Tensor):
        return self.preds


class SaveLossesMetric(pl.metrics.Metric):
    def __init__(self, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], p_save=1.0):
        super().__init__()
        self.p_save = p_save
        self.losses = []
        self.loss_fn = loss_fn

    # noinspection PyMethodOverriding
    def update(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
        num_data = y.shape[0]
        do_save = torch.rand((num_data,)) < self.p_save
        self.losses.append(self.loss_fn(y_hat[do_save], y[do_save]))

    # noinspection PyMethodOverriding
    def compute(self, y: torch.Tensor, y_hat: torch.Tensor):
        return torch.cat(self.losses, dim=0)


class MetaMetric(abc.ABC):
    """Like a pl.metrics.Metric but which takes batch meta in its forward. Default wraps metrics and discards meta."""

    def __init__(self, metric: pl.metrics.Metric):
        super().__init__()
        self.metric = metric

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor, metas: List[Any]):
        return self.metric.forward(y, y_hat)

    @abc.abstractmethod
    def update(self, y, y_hat) -> None:
        pass

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        pass


class LossMetrics:
    """Object which has a los and some metrics."""
    LOSS_NAME = constants.LOSS_NAME

    def __init__(self):
        self.loss_fn = None
        self.metrics_dict = {}

    def set_loss_fn_metrics_dict(
            self,
            loss: Loss,
            metrics_dict: Optional[Dict[str, Union[MetaMetric, pl.metrics.Metric]]] = None,
    ):
        self.loss_fn = loss
        self.metrics_dict = metrics_dict
        for k, v in self.metrics_dict.items():
            if isinstance(v, MetaMetric):
                pass
            elif isinstance(v, pl.metrics.Metric):
                self.metrics_dict[k] = MetaMetric(v)

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_hat, y)

    def metrics(self, y_hat: torch.Tensor, y: torch.Tensor, metas: List[Any]) -> Dict[str, torch.Tensor]:
        return {name: metric(y_hat, y, metas) for name, metric in self.metrics_dict.items()}

    def loss_metrics(self, y_hat: torch.Tensor, y: torch.Tensor, metas: List[Any]) -> Dict[str, torch.Tensor]:
        d = self.metrics(y_hat, y, metas)
        d[self.LOSS_NAME] = self.loss(y_hat, y)
        return d


class Head(LossMetrics, pl.LightningModule, abc.ABC):
    def __init__(
            self,
            num_input_features: int,
            num_classes: int,
            metrics_dict: Optional[Dict[str, pl.metrics.Metric]] = None,
    ):
        pl.LightningModule.__init__(self)
        self.num_input_features = num_input_features
        self.num_classes = num_classes
        self.metrics_dict = metrics_dict or {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.postproc_head_out(self.head(self.x_reducer_fn(x.squeeze(1))))

    def metrics(self, y_hat: torch.Tensor, y: torch.Tensor, metas: List[Any]) -> Dict[str, torch.Tensor]:
        d = {}
        for metric_name, metric in self.metrics_dict.items():
            d[metric_name] = metric(y_hat, y)
        return d

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        try:
            loss = self.loss_fn(y_hat, y)
        except BaseException as e:
            raise e
        if self.loss_reducer_fn is not None:
            loss = self.loss_reducer_fn(loss)
        return loss

    def postproc_head_out(self, head_out):
        return head_out


HeadMaker = Callable[[int], Head]
ReducerFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class HeadParams(utils.DataclassPlus):
    type: str
    num_classes: int
    x_reducer_name: str = ''
    loss_reducer_name: str = 'mean'


def get_reducer(num_input_features: int, reducer_str: str) -> Tuple[ReducerFn, int]:
    if reducer_str == 'smash':
        reducer_fn = trunks.artless_smash
        num_input_features *= 8
    elif reducer_str == 'first':
        reducer_fn = lambda x: x[:, 0, :]
    # elif reducer_str == 'mean':
    #     reducer_fn = lambda x: torch.mean(x)
    # elif reducer_str == 'sum':
    #     reducer_fn = lambda x: torch.sum(x)
    else:
        reducer_fn = nn.Identity()
    return reducer_fn, num_input_features


class _SoftmaxHead(Head):
    def __init__(
            self,
            num_input_features: int,
            num_classes: int,
            head: nn.Module,
            x_reducer: Optional[ReducerFn] = None,
            loss_reducer: Optional[ReducerFn] = torch.mean,
    ):
        super().__init__(
            num_input_features,
            num_classes,
            metrics_dict={
                'acc': logs_mod.BetterAccuracy(),
            }
        )

        self.x_reducer_fn = x_reducer
        self.loss_reducer_fn = loss_reducer

        self.head = head
        self.loss_fn = nn.NLLLoss(reduction='mean', ignore_index=constants.Y_VALUE_TO_IGNORE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.postproc_head_out(self.head(self.x_reducer_fn(x.squeeze(1))))

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        try:
            loss = self.loss_fn(y_hat, y)
        except BaseException as e:
            raise e
        if self.loss_reducer_fn is not None:
            loss = self.loss_reducer_fn(loss)
        return loss

    # noinspection PyMethodMayBeStatic
    def _get_reducers(self, num_input_features: int, hp: HeadParams) -> Tuple[ReducerFn, ReducerFn, int]:
        x_reducer, num_input_features = \
            get_reducer(num_input_features=num_input_features, reducer_str=hp.x_reducer_name)
        loss_reducer, num_input_features = \
            get_reducer(num_input_features=num_input_features, reducer_str=hp.loss_reducer_name)
        return x_reducer, loss_reducer, num_input_features


class AdaptiveSoftmaxHead(_SoftmaxHead):
    def __init__(
            self,
            num_input_features: int,
            hp: HeadParams,
    ):
        x_reducer, loss_reducer, num_input_features = self._get_reducers(num_input_features, hp)

        num_classes = hp.num_classes
        num_first_bin = round(num_classes / 20)
        head = nn.AdaptiveLogSoftmaxWithLoss(
            num_input_features,
            num_classes,
            cutoffs=[
                num_first_bin,
                5 * num_first_bin,
            ],
            div_value=4,
        )
        super().__init__(num_input_features, num_classes, head, x_reducer, loss_reducer)

    def postproc_head_out(self, head_out):
        # todo checkme
        return head_out[0]


class LinearSoftmaxHead(_SoftmaxHead):
    def __init__(
            self,
            num_input_features: int,
            hp: HeadParams,
    ):
        x_reducer, loss_reducer, num_input_features = self._get_reducers(num_input_features, hp)

        num_classes = hp.num_classes
        head = nn.Linear(num_input_features, num_classes)
        super().__init__(num_input_features, num_classes, head, x_reducer, loss_reducer)
        self.lsm = nn.LogSoftmax(dim=1)

    def postproc_head_out(self, head_output):
        return self.lsm(head_output.permute(0, 2, 1))


YDP = TypeVar('YDP', bound=datapoints.YDatapoint)


@dataclass
class WeightedHeadParams(utils.DataclassPlus):
    weights: Dict[str, float]
    head_params: Dict[str, HeadParams]
    type: str = 'weighted'


class WeightedHead(Head):
    LOSS_NAME = constants.LOSS_NAME
    """Multiheaded head with weights for loss."""

    def __init__(self, num_input_features: int, heads: Dict[str, Head], weights: Dict[str, float], y_dp_class: YDP):
        super().__init__(num_input_features=num_input_features, num_classes=-1, metrics_dict={})

        assert heads.keys() == weights.keys()

        self.heads = nn.ModuleDict(modules=heads)

        weight_vals = [weights[head_name] for head_name in self.heads.keys()]
        self.register_buffer('head_weights', torch.tensor(weight_vals, dtype=torch.float))

        self.y_dp_class = y_dp_class

    def forward(self, x: torch.Tensor) -> datapoints.YDatapoint:
        y_hats_dict = {}
        for head_name, head in self.heads.items():
            y_hats_dict[head_name] = head(x)
        return self.y_dp_class.from_dict(y_hats_dict)

    def losses_to_loss(self, losses: torch.Tensor) -> torch.Tensor:
        return losses.dot(self.head_weights)

    def loss(self, y_hat: datapoints.YDatapoint, y: datapoints.YDatapoint) -> torch.Tensor:
        losses = torch.stack(
            [head.loss(y_hat[head_name], y[head_name]) for head_name, head in self.heads.items()],
            dim=0,
        )
        return self.losses_to_loss(losses)

    def metrics(
            self,
            y_hat: datapoints.YDatapoint,
            y: datapoints.YDatapoint,
            metas: List[Any],
    ) -> datapoints.YDatapoint:
        d = {}
        for head_name, head in self.heads:
            y_tensor = y[head_name]
            y_hat_tensor = y_hat[head_name]
            d[head_name] = head.metrcis(y_hat_tensor, y_tensor, metas)
        return y.from_dict(d)

    def loss_metrics(
            self,
            y_hat: datapoints.YDatapoint,
            y: datapoints.YDatapoint,
            metas: List[Any],
    ) -> Dict[str, torch.Tensor]:

        head_name_to_loss_metrics = {}
        for head_name, head in self.heads.items():
            y_tensor = y[head_name]
            y_hat_tensor = y_hat[head_name]
            head_name_to_loss_metrics[head_name] = head.loss_metrics(y_hat_tensor, y_tensor, metas)

        losses = torch.stack(
            [head_name_to_loss_metrics[head_name][self.LOSS_NAME] for head_name in self.heads.keys()],
            dim=0,
        )
        loss = self.losses_to_loss(losses)
        head_name_to_loss_metrics[self.LOSS_NAME] = loss

        return head_name_to_loss_metrics

    @classmethod
    def maker_from_makers(
            cls,
            head_makers: Dict[str, Head],
            head_weights: Dict[str, float],
            y_dp_class: YDP,
    ) -> HeadMaker:
        def head_maker(num_input_features):
            heads = {head_name: hm(num_input_features) for head_name, hm in head_makers.items()}
            return cls(num_input_features=num_input_features, heads=heads, weights=head_weights, y_dp_class=y_dp_class)

        return head_maker


class HeadMakerFactory:
    """Heads need to be manufactured from neck_hp dict so it's got to be things that are easily serializable like strs."""

    @classmethod
    def create(
            cls,
            neck_hp: trunks.SlabNet.ModelParams,
            head_hp: Union[WeightedHeadParams, HeadParams],
            y_dp_class: YDP,
    ) -> HeadMaker:
        if head_hp.type == 'weighted':
            subhead_makers = {
                head_name: cls.create(neck_hp, subhead_hp, y_dp_class)
                for head_name, subhead_hp in head_hp.head_params.items()
            }

            def fn(num_input_features: int):
                return HeadedSlabNet(
                    num_input_features=num_input_features,
                    head=WeightedHead.maker_from_makers(
                        head_makers=subhead_makers,
                        head_weights=head_hp.weights,
                        y_dp_class=y_dp_class,
                    )(num_input_features),
                    neck_hp=neck_hp,
                )

            return fn
        elif head_hp.type == 'linear':
            def fn(num_input_features: int):
                return HeadedSlabNet(
                    num_input_features=num_input_features,
                    head=LinearSoftmaxHead(num_input_features, head_hp),
                    neck_hp=neck_hp,
                )

            return fn
        elif head_hp.type == 'softmax':
            def fn(num_input_features: int):
                return HeadedSlabNet(
                    num_input_features=num_input_features,
                    head=AdaptiveSoftmaxHead(num_input_features, head_hp),
                    neck_hp=neck_hp,
                )

            return fn
        else:
            raise ValueError(f'Got unknown type: {head_hp.type}')


class HeadedSlabNet(trunks.SlabNet, LossMetrics):
    """A slabnet with a head containing num_output_features."""

    def __init__(
            self,
            num_input_features: int,
            head: Optional[Head],
            head_maker: Optional[HeadMaker] = None,
            neck_hp: Optional[trunks.SlabNet.ModelParams] = trunks.SlabNet.ModelParams(),
    ):
        # TODO: omg, this is terrible
        trunks.SlabNet.__init__(
            self,
            num_input_features=num_input_features,
            hp=neck_hp,
        )
        LossMetrics.__init__(self)
        if head_maker is None:
            assert isinstance(head, Head)
            self.head = head
        else:
            self.head = head_maker(num_input_features)

    def forward(self, x):
        x = super().forward(x)
        x = self.head(x)
        return x

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.head.loss(y_hat, y)

    def metrics(self, y_hat: torch.Tensor, y: torch.Tensor, metas: List[Any]) -> Dict[str, torch.Tensor]:
        return self.head.metrics(y_hat, y, metas)

    def loss_metrics(self, y_hat: torch.Tensor, y: torch.Tensor, metas: List[Any]) -> Dict[str, torch.Tensor]:
        return self.head.loss_metrics(y_hat, y, metas)

# class StartEndHead(Head):
#     """Predict the start and end location within a sequence.  Prediction must be positive."""
#     def __init__(
#             self,
#             num_input_features: int,
#             num_classes: int,
#     ):
#         nn.Module.__init__(self)
#
#         self.start = nn.Linear(
#             in_features=num_input_features,
#             out_features=num_classes,
#             bias=True,
#         )
#
#         self.end = nn.Linear(
#             in_features=num_input_features,
#             out_features=num_classes,
#             bias=True,
#         )
#
#     def forward(self, x):
#         relu = nn.ReLU()
#         return {
#             'start_logits': relu(self.start(x)).permute(0, 2, 1),
#             'end_logits': relu(self.end(x)).permute(0, 2, 1),
#         }
