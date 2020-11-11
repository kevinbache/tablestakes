import abc
from typing import *

import torch
from torch import nn
import pytorch_lightning as pl

from tablestakes import constants
from tablestakes.ml2.factored import trunks


Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class SaveYhatsMetric(pl.metrics.Metric):
    def __init__(self, p_save=0.01):
        super().__init__()
        self.p_save = p_save
        self.preds = []

    # noinspection PyMethodOverriding
    def update(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
        num_data = y.shape[0]
        do_save = torch.rand((num_data, )) < self.p_save
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
        do_save = torch.rand((num_data, )) < self.p_save
        self.losses.append(self.loss_fn(y_hat[do_save], y[do_save]))

    # noinspection PyMethodOverriding
    def compute(self, y: torch.Tensor, y_hat: torch.Tensor):
        return torch.cat(self.losses, dim=0)


class MetaMetric(abc.ABC):
    """Like a pl.metrics.Metric but which takes batch meta in its forward."""
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

    def __init__(
            self,
            loss: Loss,
            metrics: Optional[Dict[str, Union[MetaMetric, pl.metrics.Metric]]] = None,
    ):
        super().__init__()
        self.loss_fn = loss
        self.metrics_dict = metrics
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


class Head(pl.LightningModule, LossMetrics, abc.ABC):
    def __init__(
            self,
            num_input_features: int,
            num_classes: int,
            metrics: Optional[Dict[str, pl.metrics.Metric]] = None,
    ):
        super().__init__()
        self.num_input_features = num_input_features
        self.num_classes = num_classes
        self.metrics_dict = metrics or {}

    @abc.abstractmethod
    def forward(self, x):
        pass


class StartEndHead(Head):
    """Predict the start and end location within a sequence.  Prediction must be positive."""
    def __init__(
            self,
            num_input_features: int,
            num_classes: int,
    ):
        nn.Module.__init__(self)

        self.start = nn.Linear(
            in_features=num_input_features,
            out_features=num_classes,
            bias=True,
        )

        self.end = nn.Linear(
            in_features=num_input_features,
            out_features=num_classes,
            bias=True,
        )

    def forward(self, x):
        relu = nn.ReLU()
        return {
            'start_logits': relu(self.start(x)).permute(0, 2, 1),
            'end_logits': relu(self.end(x)).permute(0, 2, 1),
        }


class HeadedSlabNet(trunks.SlabNet):
    """A slabnet with a head containing num_output_features."""
    def __init__(
            self,
            num_input_features: int,
            num_output_features: int,
            head_maker: Optional[Callable[[int, int], Head]] = None,
            hp: Optional[trunks.SlabNet.ModelParams] = trunks.SlabNet.ModelParams(),
    ):
        super().__init__(
            num_input_features=num_input_features,
            hp=hp,
        )
        # could be changed by GroupNorm fixup so must use this
        num_slab_outputs = self.get_num_outputs()

        if head_maker is None:
            head = nn.Linear(num_slab_outputs, num_output_features)
        else:
            head = head_maker(num_slab_outputs, num_output_features)
        self.head = head

        self._num_output_features = num_output_features

    def forward(self, x):
        x = super().forward(x)
        x = self.head(x)
        return x


# class HeadParams(utils.DataclassPlus):
#     # loss_weights: Dict[str, float]
#     num_classes: int
#     reducer: Optional[str] = None
#     adaptive_softmax_cutoff = 256


def get_reducer(num_input_features: int, reducer_str: str) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    if reducer_str == 'smash':
        reducer_fn = trunks.artless_smash
        num_input_features *= 8
    elif reducer_str == 'first':
        reducer_fn = lambda x: x[:, 0, :]
    else:
        reducer_fn = nn.Identity()
    return reducer_fn, num_input_features


class SoftmaxHead(Head):
    def __init__(
            self,
            num_input_features: int,
            num_classes: int,
            head: nn.Module,
            x_reducer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            y_reducer: Optional[Callable[[torch.Tensor], torch.Tensor]] = torch.mean,
    ):
        super().__init__(num_input_features, num_classes)

        self.x_reducer_fn = x_reducer
        self.y_reducer_fn = y_reducer

        self.head = head
        self.loss_fn = nn.NLLLoss()

    def postproc_head_out(self, head_output):
        return head_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.postproc_head_out(self.head(self.x_reducer_fn(x.squeeze(1))))

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(y_hat, y)
        if self.y_reducer_fn is not None:
            loss = self.y_reducer_fn(loss)
        return loss


class AdaptiveSoftmaxHead(SoftmaxHead):
    def __init__(
            self,
            num_input_features: int,
            num_classes: int,
            x_reducer_name: Optional[str] = None,
            y_reducer: Optional[Callable[[torch.Tensor], torch.Tensor]] = torch.mean,
    ):
        x_reducer, num_input_features = \
            get_reducer(num_input_features=num_input_features, reducer_str=x_reducer_name)

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
        super().__init__(num_input_features, num_classes, head, x_reducer, y_reducer)

    def postproc_head_out(self, head_output):
        return head_output[0]


class LinearSoftmaxHead(SoftmaxHead):
    def __init__(
            self,
            num_input_features: int,
            num_classes: int,
            x_reducer_name: Optional[str] = None,
            y_reducer: Optional[Callable[[torch.Tensor], torch.Tensor]] = torch.mean,
    ):
        x_reducer, num_input_features = \
            get_reducer(num_input_features=num_input_features, reducer_str=x_reducer_name)

        head = nn.Linear(num_input_features, num_classes)
        super().__init__(num_input_features, num_classes, head, x_reducer, y_reducer)


class WeightedLoss(pl.LightningModule):
    def __init__(self, heads: List[Head], weights: Iterable[float]):
        super().__init__()
        self.heads = heads
        self.weights = None
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float))

    def loss(self, y_hat, y):
        losses = torch.stack([h.loss(y_hat, y) for h in self.heads], dim=0)
        return losses.dot(self.weights)
