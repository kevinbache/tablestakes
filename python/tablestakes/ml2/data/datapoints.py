import abc
from dataclasses import dataclass, asdict, fields
from typing import *

import numpy as np
import pandas as pd

import torch

from tablestakes import utils, constants

VALUE_TO_IGNORE = constants.Y_VALUE_TO_IGNORE


def _convert_arrays_to_tensors(arrays: List[np.array], dtype):
    # noinspection PyTypeChecker
    return [torch.tensor(a, dtype=dtype) for a in arrays]


def _rnn_pad_tensors(seqs: List[torch.Tensor], max_seq_len: int, pad_value=0):
    seq_dim = 1
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_value)
    seq_len = padded.shape[seq_dim]
    if seq_len >= max_seq_len:
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
        raise NotImplementedError


class Datapoint(utils.DataclassPlus, abc.ABC):
    @classmethod
    @abc.abstractmethod
    def collate(cls, dps: List['Datapoint'], max_seq_len: int) -> 'Datapoint':
        pass

    def transfer_to_device(self, device: torch.device):
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


@dataclass
class BaseVocabDatapoint(Datapoint):
    base: Union[pd.DataFrame, torch.Tensor]
    vocab: Union[pd.DataFrame, torch.Tensor]

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
            pad_val=constants.Y_VALUE_TO_IGNORE,
        )

        return cls(
            base=base,
            vocab=vocab,
        )


class XDatapoint(Datapoint, abc.ABC):
    pass


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


class LossYDatapoint(YDatapoint, abc.ABC):
    loss: Optional[torch.Tensor] = None


@dataclass
class KorvWhichDatapoint(YDatapoint):
    korv: Any
    which_kv: Any

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
class SeqTokClassDatapoint(Datapoint):
    seq_class: Any
    token_classes: Any
    # sequence: Any

    @classmethod
    def collate(cls, dps: List['SeqTokClassDatapoint'], max_seq_len: int) -> 'SeqTokClassDatapoint':
        raise NotImplementedError()


@dataclass
class XYMetaDatapoint(Datapoint):
    x: Any
    y: Any
    meta: Any

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

    def get_y_class(self):
        return self.y.__class__

    # def __add__(self, other):
    #     return self.__class__(*(getattr(self, dim.name)+getattr(other, dim.name) for dim in fields(self)))

    @classmethod
    def collate(cls, datapoints: List['XYMetaDatapoint'], max_seq_len: int) -> 'XYMetaDatapoint':
        assert len(datapoints) > 0

        dp = datapoints[0]

        return cls(
            x=dp.x.collate([dp.x for dp in datapoints], max_seq_len=max_seq_len),
            y=dp.y.collate([dp.y for dp in datapoints], max_seq_len=max_seq_len),
            meta=dp.meta.collate([dp.meta for dp in datapoints], max_seq_len=None),
        )
