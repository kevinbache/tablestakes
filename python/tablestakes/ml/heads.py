import abc
from typing import *

import numpy as np
import pandas as pd

import torch
import transformers
from chillpill import params
from tablestakes import utils
from tablestakes.ml.torch_mod import PS
from torch import nn

import pytorch_lightning as pl

from tablestakes.ml import torch_mod, metrics_mod
from transformers.file_utils import ModelOutput

"""
structure
    specify structure from init

    get params from list
    
specify specifics of params
specifics used to construct module.  configure module?

factored
    get default params:
        gather params from subparts
        return params

    multiheaded head
    singleheaded head
        classification head
        seq classification head
        
    head.get_params()
        params.sub_losses = [head_params_1]
        for head in subheads:
            get_params()
        params_dict
        
        crawl_params()
            for param in self.__dict__:
                if hasattr param.gather_params:
"""

# PS = TypeVar('PS', params.ParameterSet)


# class Tree(torch_mod.Parameterized):
#     @abc.abstractmethod
#     def get_params(self): pass
#
#     def get_visitables(self) -> List:
#         """ includes self so you can choose PRE-order, post-order, in-order crawl for eg. binary"""
#
#     def _visit(self):
#         pass
#
#     def visit(self):
#         return [visitable.visit() for visitable in self.get_visitables()]
#
#     @classmethod
#     def from_dict(cls, d: Dict):
#         hp = cls()
#         for k, v in d.items():
#             # TODO: don't cast array to int
#             if np.issubdtype(type(v), np.integer):
#                 v = int(v)
#             elif np.issubdtype(type(v), np.floating):
#                 v = float(v)
#             elif isinstance(v, typing.MutableMapping):
#                 v = hp.__getattribute__(k).from_dict(v)
#             hp.__setattr__(k, v)
#         return hp

#   head.log:
#       for head in self.sub_losses:
#   known sub_losses:
#     span selection
#         metrics: l2 to true value + some
#         got the correct length
#
#     whole_seq_classification
#         by reducer
#             mean
#             artless smasher
#             first ( [CLS] )
#             then linear
#     single_word_classification
#         linear


class LoggedLoss(torch_mod.ParameterizedModule, abc.ABC):
    class Output(ModelOutput):
        loss: torch.Tensor

    @classmethod
    def get_default_output(cls) -> ModelOutput:
        return cls.Output()

    class LoggedLossParams(params.ParameterSet):
        pass

    @classmethod
    def get_default_params(cls) -> LoggedLossParams:
        return cls.LoggedLossParams()

    def __init__(
            self,
            name: Optional[str],
            num_input_features: int,
            num_outputs: int,
            metrics: List[pl.metrics.Metric],
    ):
        super().__init__()
        self.name = name
        self.num_input_features = num_input_features
        self.num_outputs = num_outputs
        self.metrics: List[pl.metrics.Metric] = metrics

    def get_params(self) -> PS:
        return self.hp

    def _get_my_y(self, batch):
        x, y, meta = batch
        return y[self.name]

    @abc.abstractmethod
    def forward(self, trunk_out, batch, batch_idx):
        pass

    @abc.abstractmethod
    def my_log(self, loss_out, batch, batch_idx, phase, pl_module: pl.LightningModule) -> Dict[str, Any]:
        pass

    def training_step(self, batch, batch_idx, pl_module: pl.LightningModule):
        return self._inner_step_by_phase(batch, batch_idx, utils.Phase.train, pl_module)

    def validation_step(self, batch, batch_idx, pl_module: pl.LightningModule):
        return self._inner_step_by_phase(batch, batch_idx, utils.Phase.valid, pl_module)

    def test_step(self, batch, batch_idx, pl_module: pl.LightningModule):
        return self._inner_step_by_phase(batch, batch_idx, utils.Phase.test, pl_module)

    def _inner_step_by_phase(self, batch, batch_idx, phase, pl_module):
        # x, y, meta = batch
        loss_out = self.loss(batch)
        self.my_log(loss_out, batch, batch_idx, phase, pl_module)
        return loss_out.loss


class MultiHeadedLoggedLoss(LoggedLoss):
    def get_params(self) -> PS:
        pass

    class Output(ModelOutput):
        loss: torch.Tensor
        losses: Dict[str, ModelOutput]

    class LoggedLossParams(LoggedLoss.LoggedLossParams):
        head_weights: torch.Tensor

    def __init__(self, sub_losses: List[LoggedLoss], hp: LoggedLossParams, name: str = 'loss'):
        if len(sub_losses) == 0:
            raise NotImplementedError()

        subloss_weights = hp.head_weights
        assert len(subloss_weights) == len(sub_losses)

        for sub_loss in sub_losses:
            assert sub_loss.num_input_features == sub_losses[0].num_input_features

        super().__init__(
            name=name,
            num_input_features=sub_losses[0].num_input_features,
            num_outputs=len(sub_losses),
            metrics=[],
        )
        self.sub_losses = sub_losses
        self.subloss_weights = subloss_weights

    def forward(self, trunk_out, batch, batch_idx):
        sub_loss_outputs = {sub_loss.name: sub_loss(x) for sub_loss in self.sub_losses}
        loss = torch.dot(self.subloss_weights, torch.tensor([out.loss for out in sub_loss_outputs.values()]))
        return self.Output(loss=loss, losses=sub_loss_outputs, weights=self.subloss_weights)

    def my_log(self, loss_out, batch, batch_idx, phase, pl_module: pl.LightningModule):
        return {sub_loss.name: sub_loss.my_log(loss_out, batch, batch_idx, phase) for sub_loss in self.sub_losses}


class ClassificationLoggedLoss(LoggedLoss):
    class Output(ModelOutput):
        y_hat: torch.Tensor
        loss: torch.Tensor

    class LoggedLossParams(params.ParameterSet):
        class_weights: Optional[torch.FloatTensor] = None

    def __init__(self, name, num_input_features, num_classes):
        # noinspection PyTypeChecker
        super().__init__(
            name=name,
            num_input_features=num_input_features,
            num_outputs=num_classes,
            metrics=[metrics_mod.BetterAccuracy],
        )
        self.linear = nn.Linear(num_input_features, num_classes)

        self.is_small = self.num_outputs < 256
        if self.is_small:
            self.sm = nn.CrossEntropyLoss()
        else:
            self.sm = torch_mod.make_lm_loss(num_features=num_input_features, num_vocab=num_classes)

    def _process_my_y(self, y) -> Any:
        """Pass through by default"""
        return y

    def my_log(self, loss_out: Output, batch, batch_idx, phase: utils.Phase, pl_module: pl.LightningModule):
        y = self._get_my_y(batch)
        y = self._process_my_y(y)

        d = {}

        for metric in self.metrics:
            d[metric.name] = metric(loss_out.y_hat, y)

        d[self.LOSS_NAME] = loss_out.loss

        return d

    def forward(self, x, y):
        y_hat = self.linear(x)
        return self.Output(y_hat=output, loss=smout.loss)


class ReducedSequenceClassificationHead(ClassificationLoggedLoss):
    def __init__(self, name, num_input_features, num_classes, reducer_op):
        # noinspection PyTypeChecker
        super().__init__(
            name=name,
            num_input_features=num_input_features,
            num_classes=num_classes,
        )
        self.linear = nn.Linear(num_input_features, num_classes)
        self.reducer_op = reducer_op

    def forward(self, x):
        x = self.linear(self.reducer_op(x).squeeze(1))
        return x


class FirstSequenceClassificationHead(ReducedSequenceClassificationHead):
    def __init__(self, name, num_input_features, num_classes):
        super().__init__(
            name=name,
            num_input_features=num_input_features,
            num_classes=num_classes,
            reducer_op=lambda x: x[:, 0, :],
        )


class SmashedSequenceClassificationLoggedLoss(ReducedSequenceClassificationHead):
    def __init__(self, name, num_input_features, num_classes):
        super().__init__(
            name=name,
            num_input_features=8 * num_input_features,
            num_classes=num_classes,
            reducer_op=torch_mod.artless_smash,
        )
        self.num_input_features = num_input_features


class ClassificationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes

        self.loss = nn.CrossEntropyLoss()

        self.metrics = [
        ]


# class SpanSelectionHead(ReducedSequenceClassificationHead):
#     def __init__(self, num_input_features, num_classes):
#         super().__init__(num_input_features, num_classes)
#         self.start = nn.Linear(num_input_features, num_classes)
#         self.end = nn.Linear(num_input_features, num_classes)


loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)

if __name__ == '__main__':
    n_batch = 5
    n_seq = 7
    n_feat = 3

    n_class = 2

    x = torch.rand([n_batch, n_seq, n_feat])

    head_name = 'tok_class'
    tok_class_head_hp = ClassificationLoggedLoss.LoggedLossParams()
    tok_class_loss = ClassificationLoggedLoss(head_name, n_feat, n_class)

    head_name = 'doc_class'
    doc_class_head_hp = SmashedSequenceClassificationLoggedLoss.LoggedLossParams()
    doc_class_loss = SmashedSequenceClassificationLoggedLoss(head_name, n_feat, n_class)

    trunk = nn.Linear(n_feat, n_feat)
    trunk_out = trunk(x)

    hp = MultiHeadedLoggedLoss.LoggedLossParams()
    hp.head_weights = torch.tensor([2., 3.])

    loss = MultiHeadedLoggedLoss(
        name='weighted',
        sub_losses=[tok_class_loss, doc_class_loss],
        hp=hp,
    )
    loss_out = loss(trunk_out)


