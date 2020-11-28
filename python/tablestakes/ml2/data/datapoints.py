import abc
from dataclasses import dataclass
from typing import *

import numpy as np
import pandas as pd

import torch

from tablestakes import utils, constants
from transformers import BertTokenizer

VALUE_TO_IGNORE = constants.Y_VALUE_TO_IGNORE


def _convert_arrays_to_tensors(arrays: List[np.array], dtype):
    # noinspection PyTypeChecker
    return [torch.tensor(a, dtype=dtype) for a in arrays]


def _rnn_pad_tensors(seqs: List[torch.Tensor], max_seq_len: int, pad_value=0):
    seq_dim = 1
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_value)
    seq_len = padded.shape[seq_dim]
    if seq_len > max_seq_len:
        padded = padded.narrow(dim=seq_dim, start=0, length=max_seq_len)
    return padded


def _pad_arrays(arrays: List[np.array], dtype, max_seq_len, pad_val=0):
    return _rnn_pad_tensors(
        _convert_arrays_to_tensors(arrays, dtype=dtype),
        max_seq_len=max_seq_len,
        pad_value=pad_val,
    )


def _get_df_or_tensor_num_features(e: Union[pd.DataFrame, torch.Tensor]):
    if hasattr(e, 'get_num_features'):
        return e.get_num_features()
    elif isinstance(e, (pd.DataFrame, torch.Tensor)):
        return e.shape[-1]
    else:
        raise NotImplementedError(f'type: {type(e)}, e: {e}')


class Datapoint(utils.DataclassPlus, abc.ABC):
    @classmethod
    @abc.abstractmethod
    def collate(cls, dps: List['Datapoint'], max_seq_len: int) -> 'Datapoint':
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    def transfer_to_device(self, device: torch.device):
        """Recurses on members."""
        def _inner(obj: Any, device: torch.device):
            if hasattr(obj, 'to'):
                return obj.to(device)

            if isinstance(obj, utils.DataclassPlus):
                for k, v in obj:
                    obj[k] = _inner(v, device)
                # d = {k: _inner(v, device) for k, v in obj}
                return obj

            if isinstance(obj, Dict):
                return {k: _inner(v, device) for k, v in obj.items()}

            if isinstance(obj, (list, tuple)):
                return [_inner(o, device) for o in obj]

            return obj

        return _inner(self, device)

    def get_num_features(self):
        # noinspection PyArgumentList
        return self.__class__(**{
            k: _get_df_or_tensor_num_features(v) for k, v in self
        })

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)


@dataclass
class BaseVocabDatapoint(Datapoint):
    base: Union[pd.DataFrame, torch.Tensor]
    vocab: Union[pd.DataFrame, torch.Tensor]

    def __len__(self):
        return len(self.base)

    # noinspection PyMethodOverriding
    @classmethod
    def collate(cls, dps: List['BaseVocabDatapoint'], max_seq_len: int) -> 'BaseVocabDatapoint':
        # noinspection PyTypeChecker
        base = _pad_arrays(
            arrays=[dp.base.values for dp in dps],
            dtype=torch.float,
            max_seq_len=max_seq_len,
            pad_val=0,
        )
        vocab = _pad_arrays(
            arrays=[dp.vocab.values for dp in dps],
            dtype=torch.long,
            max_seq_len=max_seq_len,
            pad_val=utils.VOCAB_PAD_VALUE,
        ).squeeze(2)

        return cls(
            base=base,
            vocab=vocab,
        )

    def get_batch_lens(self):
        return np.where(self.vocab == utils.VOCAB_PAD_VALUE)[0]


@dataclass
class XDatapoint(Datapoint, abc.ABC):
    pass


@dataclass
class YDatapoint(Datapoint, abc.ABC):
    def get_num_features(self):
        d = {}
        for k, v in self:
            if isinstance(v, pd.DataFrame):
                d[k] = v.values.max() + 1
            elif isinstance(v, torch.Tensor):
                d[k] = v.max().item() + 1

        # noinspection PyArgumentList
        return self.__class__(**d)


@dataclass
class LossMetricsDatapoint(Datapoint, abc.ABC):
    loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]

    @classmethod
    def collate(cls, dps: List['LossMetricsDatapoint'], max_seq_len: int) -> 'LossMetricsDatapoint':
        raise NotImplementedError()


# @dataclass
# class WeightedLossYDatapoint(Datapoint, abc.ABC):
#     loss: torch.Tensor
#     weights: torch.Tensor
#     metrics: YDatapoint
#
#     def __len__(self):
#         return len(self.metrics)
#
#     @classmethod
#     def collate(cls, dps: List['WeightedLossYDatapoint'], max_seq_len: int) -> 'WeightedLossYDatapoint':
#         raise NotImplementedError()


@dataclass
class KorvWhichDatapoint(YDatapoint):
    korv: Any
    which_kv: Any

    def __len__(self):
        return len(self.korv)

    # noinspection PyMethodOverriding
    @classmethod
    def collate(cls, dps: List['KorvWhichDatapoint'], max_seq_len: int) -> 'KorvWhichDatapoint':
        korv = _pad_arrays(
            arrays=[dp.korv.values.squeeze() for dp in dps],
            dtype=torch.long,
            max_seq_len=max_seq_len,
            pad_val=0,
        )
        which_kv = _pad_arrays(
            arrays=[dp.which_kv.values.squeeze() for dp in dps],
            dtype=torch.long,
            max_seq_len=max_seq_len,
            pad_val=constants.Y_VALUE_TO_IGNORE,
        )

        return cls(
            korv=korv,
            which_kv=which_kv,
        )


@dataclass
class MetaDatapoint(Datapoint):
    datapoint_dir: Union[str, List[str]]

    def __len__(self):
        return len(self.datapoint_dir)

    # noinspection PyMethodOverriding
    @classmethod
    def collate(cls, dps: List['MetaDatapoint'], max_seq_len: int) -> 'MetaDatapoint':
        return cls(
            datapoint_dir=[dp.datapoint_dir for dp in dps],
        )

    def get_num_features(self):
        return None

    def transfer_to_device(self, device: torch.device):
        return self


@dataclass
class XYMetaDatapoint(Datapoint):
    x: Any
    y: Any
    meta: Any

    def __len__(self):
        return len(self.x)

    @classmethod
    def from_makers(
            cls,
            data_dir,
            x_maker: [Callable[[utils.DirtyPath], Any]],
            y_maker: [Callable[[utils.DirtyPath], Any]],
            meta_maker: [Callable[[utils.DirtyPath], Any]]
    ):
        return cls(
            x=x_maker(data_dir),
            y=y_maker(data_dir),
            meta=meta_maker(data_dir),
        )

    # def get_y_class(self):
    #     return self.y.__class__

    @classmethod
    def collate(cls, datapoints: List['XYMetaDatapoint'], max_seq_len: int) -> 'XYMetaDatapoint':
        assert len(datapoints) > 0

        dp = datapoints[0]

        return cls(
            x=dp.x.collate([dp.x for dp in datapoints], max_seq_len=max_seq_len),
            y=dp.y.collate([dp.y for dp in datapoints], max_seq_len=max_seq_len),
            meta=dp.meta.collate([dp.meta for dp in datapoints], max_seq_len=None),
        )
