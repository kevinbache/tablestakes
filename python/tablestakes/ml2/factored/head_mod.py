import abc
import copy
from typing import *

import numpy as np
import pandas as pd

import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import confusion_matrix

from tablestakes import constants, utils
from tablestakes.ml2.factored import trunks_mod, logs_mod
from tablestakes.ml2.data import datapoints

from chillpill import params

Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# class SaveYhatsMetric(pl.metrics.Metric):
#     def __init__(self, p_save=0.01):
#         super().__init__()
#         self.p_save = p_save
#         self.preds = []
#
#     # noinspection PyMethodOverriding
#     def update(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
#         num_data = y.shape[0]
#         do_save = torch.rand((num_data,)) < self.p_save
#         self.preds.append(y_hat[do_save])
#
#     # noinspection PyMethodOverriding
#     def compute(self, y_hat: torch.Tensor, y: torch.Tensor):
#         return self.preds


# class SaveLossesMetric(pl.metrics.Metric):
#     def __init__(self, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], p_save=1.0):
#         super().__init__()
#         self.p_save = p_save
#         self.losses = []
#         self.loss_fn = loss_fn
#
#     # noinspection PyMethodOverriding
#     def update(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
#         num_data = y.shape[0]
#         do_save = torch.rand((num_data,)) < self.p_save
#         self.losses.append(self.loss_fn(y_hat[do_save], y[do_save]))
#
#     # noinspection PyMethodOverriding
#     def compute(self, y: torch.Tensor, y_hat: torch.Tensor):
#         return torch.cat(self.losses, dim=0)


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
            else:
                raise NotImplementedError(f'type(v): {type(v)}')

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_hat, y)

    def metrics(self, y_hat: torch.Tensor, y: torch.Tensor, metas: List[Any]) -> Dict[str, torch.Tensor]:
        return {name: metric(y_hat, y, metas) for name, metric in self.metrics_dict.items()}

    def loss_metrics(
            self,
            y_hats_for_loss: torch.Tensor,
            y_hats_for_pred: torch.Tensor,
            y: torch.Tensor,
            metas: List[Any],
    ) -> Dict[str, torch.Tensor]:
        d = self.metrics(y_hats_for_pred, y, metas)
        d[self.LOSS_NAME] = self.loss(y_hats_for_loss, y)
        return d


class HeadParams(trunks_mod.BuilderParams):
    type: str = 'DEFAULT_TYPE'
    num_classes: int = -1
    class_names: Tuple[str] = ()
    # See get_reducer() for x_ and loss_ reducer options
    x_reducer_name: str = ''
    loss_reducer_name: str = 'mean-0'
    do_permute_head_out: bool = True
    # apply weight between classes
    class_weights: Optional[np.array] = None
    # equalize pos/neg within a class
    pos_class_weights: Optional[np.array] = None

    neck: Optional[trunks_mod.SlabNet.ModelParams] = None

    def build(self, num_input_features: int) -> Any:
        if self.neck is not None:
            num_head_inputs = max(self.neck.num_features, self.neck.num_groups)
        else:
            num_input_features

        type_str = self.type.lower()

        if type_str == 'linear':
            head = LinearSoftmaxHead(num_head_inputs, self)
        elif type_str == 'softmax':
            head = AdaptiveSoftmaxHead(num_head_inputs, self)
        elif type_str == constants.SIGMOID_TYPE_NAME:
            head = SigmoidHead(num_head_inputs, self)
        elif head_hp.type is None or head_hp.type.lower() == 'none':
            head = EmptyHead()
        else:
            raise ValueError(f'Got unknown type: {head_hp.type}')

        if self.neck is not None:
            return NeckHead(self.neck.build(num_input_features), head)

    @classmethod
    def from_dict(cls, d: Dict):
        out = copy.deepcopy(cls(**d))
        assert hasattr(out, 'neck')
        if out.neck is not None:
            out.neck = trunks_mod.SlabNet.ModelParams.from_dict(out.neck)
        return out



class EmptyHeadParams(HeadParams):
    type: str = 'none'
    num_classes: int = -1
    x_reducer_name: str = 'none'
    do_permute_head_out: bool = False


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

        self.head = nn.Identity()

        self.loss_fn = lambda y_hat, y: -1
        self.x_reducer_fn = lambda x: x
        self.loss_reducer_fn = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_for_pred(x)

    def forward_for_loss(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.squeeze(1)
        x = self.x_reducer_fn(x)
        x = self.postproc_head_out_for_loss_fn(self.head(x))
        return x

    def forward_for_pred(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.squeeze(1)
        x = self.x_reducer_fn(x)
        x = self.postproc_head_out_for_pred(self.head(x))
        return x

    def forward_for_loss_and_pred(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: disabled for lawplus model2
        # x = x.squeeze(1)
        x = self.x_reducer_fn(x)
        x = self.head(x)
        x_for_loss = self.postproc_head_out_for_loss_fn(x)
        x_for_pred = self.postproc_head_out_for_pred(x)
        return x_for_loss, x_for_pred

    def metrics(self, y_hat: torch.Tensor, y: torch.Tensor, metas: List[Any]) -> Dict[str, torch.Tensor]:
        d = {}
        for metric_name, metric in self.metrics_dict.items():
            d[metric_name] = metric(y_hat, y)
        return d

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(y_hat, y)
        if self.loss_reducer_fn is not None:
            loss = self.loss_reducer_fn(loss)
        return loss

    def postproc_head_out_for_loss_fn(self, head_out):
        return head_out

    def postproc_head_out_for_pred(self, head_out):
        return head_out

    # noinspection PyMethodMayBeStatic
    def _get_reducers(self, num_input_features: int, hp: HeadParams) -> Tuple[utils.ReducerFn, utils.ReducerFn, int]:
        x_reducer, num_input_features = \
            utils.get_reducer(num_input_features=num_input_features, aggregator_str=hp.x_reducer_name)
        loss_reducer, num_input_features_loss_red = \
            utils.get_reducer(num_input_features=num_input_features, aggregator_str=hp.loss_reducer_name)
        assert num_input_features_loss_red == num_input_features

        return x_reducer, loss_reducer, num_input_features


class EmptyHead(Head):
    def __init__(self):
        super().__init__(-1, -1, {})
        self.loss_fn = lambda y_hat, y: -1
        self.head = nn.Identity()
        self.x_reducer_fn = lambda x: x


class SigmoidHead(Head):
    def __init__(
            self,
            num_input_features: int,
            hp: HeadParams,
            metrics_dict=None,
    ):
        x_reducer, loss_reducer, num_input_features = self._get_reducers(num_input_features, hp)

        if metrics_dict is None:
            metrics_dict = {
                'sacc': logs_mod.SigmoidBetterAccuracy(),
            }

        num_classes = hp.num_classes
        super().__init__(
            num_input_features=num_input_features,
            num_classes=num_classes,
            metrics_dict=metrics_dict,
        )
        self.head = nn.Linear(num_input_features, num_classes)

        self.num_classes = num_classes

        # these are for balancing pos/neg within classes
        class_weights = hp.class_weights or np.ones(num_classes)
        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float))
        self.class_weights /= self.class_weights.sum()

        # these are for balancing between classes
        pos_class_weights = hp.pos_class_weights or np.ones(num_classes)
        self.register_buffer('pos_class_weights', torch.tensor(pos_class_weights, dtype=torch.float))
        warnings.warn(f'Final pos_class_weights: {self.pos_class_weights}')
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_class_weights)

        self.x_reducer_fn = x_reducer
        self.loss_reducer_fn = loss_reducer

        self.do_permute_head_out = hp.do_permute_head_out

        self.s = nn.Sigmoid()

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(y_hat, y)
        if self.loss_reducer_fn is not None:
            loss = self.loss_reducer_fn(loss)
        assert loss.shape == torch.Size([self.num_classes]), \
            f'loss.size: {loss.shape}, torch.Size([self.num_classes]): {torch.Size([self.num_classes])}'
        return loss.dot(self.class_weights)

    def postproc_head_out_for_pred(self, head_out):
        return self.s(head_out)

    def postproc_head_out_for_loss_fn(self, head_out):
        return head_out


class _SoftmaxHead(Head):
    def __init__(
            self,
            num_input_features: int,
            num_classes: int,
            head: nn.Module,
            x_reducer: Optional[utils.ReducerFn] = None,
            loss_reducer: Optional[utils.ReducerFn] = torch.mean,
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

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        try:
            loss = self.loss_fn(y_hat, y)
        except BaseException as e:
            raise e
        if self.loss_reducer_fn is not None:
            loss = self.loss_reducer_fn(loss)
        return loss


class AdaptiveSoftmaxHead(_SoftmaxHead):
    def __init__(
            self,
            num_input_features: int,
            hp: HeadParams,
    ):
        x_reducer, loss_reducer, num_input_features = self._get_reducers(num_input_features=num_input_features, hp=hp)

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

    def postproc_head_out_for_loss_fn(self, head_out):
        # todo checkme
        return head_out[0]

    def postproc_head_out_for_pred(self, head_out):
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
        self.do_permute_head_out = hp.do_permute_head_out

    def postproc_head_out_for_loss_fn(self, head_output):
        if self.do_permute_head_out:
            return self.lsm(head_output.permute(0, 2, 1))
        else:
            return self.lsm(head_output)

    def postproc_head_out_for_pred(self, head_output):
        return self.postproc_head_out_for_loss_fn(head_output)


YDP = TypeVar('YDP', bound=datapoints.YDatapoint)


class WeightedHeadParams(trunks_mod.BuilderParams):
    weights: Dict[str, float] = {}
    head_params: Dict[str, HeadParams] = {}
    type: str = 'weighted'

    @classmethod
    def from_dict(cls, d: Dict):
        obj = copy.deepcopy(cls())
        assert 'weights' in d
        obj.weights = d['weights']
        assert 'head_params' in d
        assert isinstance(d['head_params'], dict)
        obj.head_params = {k: HeadParams.from_dict(v) for k, v in d['head_params'].items()}
        assert 'type' in d
        obj.type = d['type']
        return obj

    def build(self, num_input_features: int) ->  'WeightedHead':
        heads = {
            name: head_params.build(num_input_features)
            for name, head_params in self.head_params.items()
        }
        return WeightedHead(num_input_features=num_input_features, heads=heads, weights=self.weights)


class MyConfusionMatrix(pl.metrics.ConfusionMatrix):
    """Just like the normal one but with device transfer"""

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        confmat = confusion_matrix._confusion_matrix_update(preds, target, self.num_classes, self.threshold)
        confmat = confmat.to(self.confmat.device)
        self.confmat += confmat


class SigmoidConfusionMatrixCallback(pl.Callback):
    """Maintain confusion matrices for each SigmoidHead"""
    DEFAULT_HEAD_NAME = 'DEFAULT'

    def __init__(self, hp: Union[HeadParams, WeightedHeadParams]):
        super().__init__()
        self.head_name_to_col_dicts = self._get_head_name_to_col_dicts(hp)
        self.cmdict_per_epoch = []
        self.hp = copy.deepcopy(hp)

    @classmethod
    def _get_col_name_to_cm_for_sub_head(cls, sub_hp: HeadParams):
        col_name_to_cm = {}
        if sub_hp.type == constants.SIGMOID_TYPE_NAME:
            col_name_to_cm = {
                col_name: MyConfusionMatrix(
                    num_classes=2,
                    normalize=None,
                    compute_on_step=False,
                )
                for col_name in sub_hp.class_names
            }
        return col_name_to_cm

    @classmethod
    def _get_head_name_to_col_dicts(cls, hp) -> Dict[str, Dict[str, pl.metrics.ConfusionMatrix]]:
        head_name_to_col_dicts = {}
        if isinstance(hp, WeightedHeadParams):
            for head_name, sub_hp in hp.head_params.items():
                col_name_to_cm = cls._get_col_name_to_cm_for_sub_head(sub_hp)
                head_name_to_col_dicts[head_name] = col_name_to_cm
        else:
            col_name_to_cm = cls._get_col_name_to_cm_for_sub_head(hp)
            head_name_to_col_dicts[cls.DEFAULT_HEAD_NAME] = col_name_to_cm
        return head_name_to_col_dicts

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs,
            batch: datapoints.XYMetaDatapoint,
            batch_idx,
            dataloader_idx,
    ):
        batch = batch.transfer_to_device(pl_module.device)

        _, head_to_y_hats = pl_module(batch.x)
        for head_name, col_dict in self.head_name_to_col_dicts.items():
            preds: torch.Tensor = head_to_y_hats[head_name]
            target: torch.Tensor = batch.y[head_name]

            for col_idx, (col_name, cm) in enumerate(col_dict.items()):
                cm.update(preds[:, col_idx], target[:, col_idx])
                self.head_name_to_col_dicts[head_name][col_name] = cm

    def on_validation_epoch_end(self, trainer, pl_module):
        for head_name, col_dict in self.head_name_to_col_dicts.items():
            for col_name, cm in col_dict.items():
                vals = cm.compute().numpy()
                df = pd.DataFrame(vals, columns=['pred=0', 'pred=1'], index=['true=0', 'true=1'])
                self.head_name_to_col_dicts[head_name][col_name] = df

        d = {'cms': copy.deepcopy(self.head_name_to_col_dicts)}
        pl_module.log_lossmetrics_dict(phase=utils.Phase.valid, d=d)

        self.cmdict_per_epoch.append(self.head_name_to_col_dicts)
        self.head_name_to_col_dicts = self._get_head_name_to_col_dicts(self.hp)


# HeadMaker = Callable[[int], Head]


class WeightedHead(Head):
    LOSS_NAME = constants.LOSS_NAME
    """Multiheaded head with weights for loss."""

    def __init__(self, num_input_features: int, heads: Dict[str, Head], weights: Dict[str, float]):
        super().__init__(num_input_features=num_input_features, num_classes=-1, metrics_dict={})

        assert heads.keys() == weights.keys()

        self.heads = nn.ModuleDict(modules={
            head_name: head
            for head_name, head in heads.items()
        })

        weight_vals = [weights[head_name] for head_name in self.heads.keys()]
        self.register_buffer('head_weights', torch.tensor(weight_vals, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        y_hats_dict = {}
        for head_name, head in self.heads.items():
            y_hats_dict[head_name] = head(x)
        return y_hats_dict

    def forward_for_loss(self, x: torch.Tensor) -> Dict[str, Any]:
        y_hats_dict = {}
        for head_name, head in self.heads.items():
            y_hats_dict[head_name] = head.forward_for_loss(x)
        return y_hats_dict

    def forward_for_loss_and_pred(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        y_hats_for_loss_dict = {}
        y_hats_for_pred_dict = {}
        for head_name, head in self.heads.items():
            y_hats_for_loss_dict[head_name], y_hats_for_pred_dict[head_name] = head.forward_for_loss_and_pred(x)
        return y_hats_for_loss_dict, y_hats_for_pred_dict

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
    ) -> Dict[str, Dict[str, Any]]:
        d = {}
        for head_name, head in self.heads:
            y_tensor = y[head_name]
            y_hat_tensor = y_hat[head_name]
            d[head_name] = head.metrcis(y_hat_tensor, y_tensor, metas)
        return d

    def loss_metrics(
            self,
            y_hats_for_loss: datapoints.YDatapoint,
            y_hats_for_pred: datapoints.YDatapoint,
            y: datapoints.YDatapoint,
            metas: List[Any],
    ) -> Dict[str, Union[float, Dict[str, torch.Tensor]]]:

        head_name_to_loss_metrics = {}
        for head_name, head in self.heads.items():
            y_tensor = y[head_name]
            y_hat_for_loss_tensor = y_hats_for_loss[head_name]
            y_hat_for_pred_tensor = y_hats_for_pred[head_name]
            head_name_to_loss_metrics[head_name] = \
                head.loss_metrics(y_hat_for_loss_tensor, y_hat_for_pred_tensor, y_tensor, metas)

        losses = torch.stack(
            [head_name_to_loss_metrics[head_name][self.LOSS_NAME] for head_name in self.heads.keys()],
            dim=0,
        )
        loss = self.losses_to_loss(losses)
        head_name_to_loss_metrics[self.LOSS_NAME] = loss

        return head_name_to_loss_metrics

    # @classmethod
    # def maker_from_makers(
    #         cls,
    #         head_makers: Dict[str, HeadMaker],
    #         head_weights: Dict[str, float],
    # ) -> HeadMaker:
    #
    #     def head_maker(num_input_features):
    #         heads = {}
    #         head_weights_out = {}
    #         for head_name, hm in head_makers.items():
    #             if head_weights[head_name] != 0.0:
    #                 heads[head_name] = hm(num_input_features)
    #                 head_weights_out[head_name] = head_weights[head_name]
    #         return WeightedHead(
    #             num_input_features=num_input_features,
    #             heads=heads,
    #             weights=head_weights_out,
    #         )
    #
    #     return head_maker


# class HeadMakerFactory:
#     SIGMOID_TYPE_NAME = 'sigmoid'
#
#     """Heads need to be manufactured from neck_hp dict so it's got
#     to be things that are easily serializable like strs."""
#     @classmethod
#     def create(
#             cls,
#             neck_hp: trunks_mod.SlabNet.ModelParams,
#             head_hp: Union[WeightedHeadParams, HeadParams],
#             # neck_trans_hp: Optional[trunks_mod.TransBlockBuilder.ModelParams] = None,
#     ) -> HeadMaker:
#
#         num_head_inputs = max(neck_hp.num_features, neck_hp.num_groups)
#
#         if head_hp.type == 'weighted':
#             subhead_makers = {
#                 head_name: cls.create(neck_hp, subhead_hp)
#                 for head_name, subhead_hp in head_hp.head_params.items()
#             }
#             def fn(num_input_features: int):
#                 head_maker = WeightedHead.maker_from_makers(
#                     head_makers=subhead_makers,
#                     head_weights=head_hp.weights,
#                 )
#                 return head_maker(num_input_features)
#             return fn
#         elif head_hp.type == 'linear':
#             def fn(num_input_features: int):
#                 return HeadedSlabNet(
#                     num_input_features=num_input_features,
#                     head=LinearSoftmaxHead(num_head_inputs, head_hp),
#                     neck_hp=neck_hp,
#                 )
#             return fn
#         elif head_hp.type == 'softmax':
#             def fn(num_input_features: int):
#                 return HeadedSlabNet(
#                     num_input_features=num_input_features,
#                     head=AdaptiveSoftmaxHead(num_head_inputs, head_hp),
#                     neck_hp=neck_hp,
#                 )
#             return fn
#         elif head_hp.type == cls.SIGMOID_TYPE_NAME:
#             def fn(num_input_features: int):
#                 return HeadedSlabNet(
#                     num_input_features=num_input_features,
#                     head=SigmoidHead(num_head_inputs, head_hp),
#                     neck_hp=neck_hp,
#                 )
#             return fn
#         elif head_hp.type is None or head_hp.type == 'none':
#             return lambda _: EmptyHead()
#         else:
#             raise ValueError(f'Got unknown type: {head_hp.type}')


class NeckHead(pl.LightningModule, LossMetrics):
    def __init__(
            self,
            neck: Sized,
            head: Head,
    ):
        super().__init__()
        self.neck = neck
        self.head = head

    def forward(self, x):
        x = self.neck(x)
        x = self.head(x)
        return x

    def forward_for_loss(self, x: torch.Tensor) -> torch.Tensor:
        x = self.neck(x)
        x = self.head.forward_for_loss(x)
        return x

    def forward_for_loss_and_pred(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.neck(x)
        x_for_loss, x_for_pred = self.head.forward_for_loss_and_pred(x)
        return x_for_loss, x_for_pred

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.head.loss(y_hat, y)

    def metrics(self, y_hat: torch.Tensor, y: torch.Tensor, metas: List[Any]) -> Dict[str, torch.Tensor]:
        return self.head.metrics(y_hat, y, metas)

    def loss_metrics(
            self,
            y_hat_for_loss: torch.Tensor,
            y_hat_for_pred: torch.Tensor,
            y: torch.Tensor,
            metas: List[Any],
    ) -> Dict[str, torch.Tensor]:
        return self.head.loss_metrics(y_hat_for_loss, y_hat_for_pred, y, metas)
