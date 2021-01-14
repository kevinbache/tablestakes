import abc
import copy
from typing import *

import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

import transformers

import pytorch_lightning as pl

from tablestakes import utils, constants
from tablestakes.ml import ablation

from chillpill import params


class BuilderParams(params.ParameterSet, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def build(self, *args, **kwargs) -> Any:
        pass


class Sized(abc.ABC):
    @abc.abstractmethod
    def get_num_output_features(self):
        pass

class SizedBuilderParams(Sized, BuilderParams, abc.ABC):
    pass



class SizedPlModule(pl.LightningModule, Sized):
    pass

class SizedSequential(Sized, nn.Sequential):
    def __init__(self, args: Union[OrderedDict, List, Tuple], num_output_features):
        if isinstance(args, OrderedDict):
            super().__init__(args)
        elif isinstance(args, (list, tuple)):
            super().__init__(*args)
        else:
            raise ValueError(f"Cant handle input of type {type(args)}")

        self._num_output_features = num_output_features

    def get_num_output_features(self):
        return self._num_output_features


def resnet_conv1_block(
        num_input_features: int,
        num_output_features: int,
        num_gn_groups: int = 32,
        activation=nn.LeakyReLU,
        do_include_activation=True,
        do_include_group_norm=True,
        kernel_size=1,
        stride=1,
):
    """activation, fc, groupnorm"""
    layers = OrderedDict()
    if do_include_group_norm:
        layers['norm'] = nn.GroupNorm(num_groups=num_gn_groups, num_channels=num_input_features)
    if do_include_activation:
        layers['act'] = activation()

    layers['conv'] = nn.Conv1d(
        in_channels=num_input_features,
        out_channels=num_output_features,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

    return SizedSequential(layers, num_output_features=num_output_features)


class BertEmbedder(pl.LightningModule):
    class ModelParams(BuilderParams):
        dim: int = 64
        requires_grad: bool = True
        position_embedding_requires_grad: bool = False

        def build(self, max_seq_len: int) -> 'BertEmbedder':
            return BertEmbedder(self, max_seq_len=max_seq_len)

    def __init__(self, hp: Optional[ModelParams] = ModelParams(), max_seq_len=1024):
        super().__init__()
        self.hp = hp

        config = transformers.BertConfig(
            hidden_size=self.hp.dim,
            num_hidden_layers=0,
            num_attention_heads=1,
            intermediate_size=0,
            max_position_embeddings=max_seq_len,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        self.e = transformers.BertModel(config, add_pooling_layer=False)
        for name, param in self.e.named_parameters():
            # param names
            #   embeddings.word_embeddings.weight
            #   embeddings.position_embeddings.weight
            #   embeddings.token_type_embeddings.weight
            #   embeddings.LayerNorm.weight
            #   embeddings.LayerNorm.bias
            if 'position_embeddings' in name:
                requires_grad = self.hp.position_embedding_requires_grad
            else:
                requires_grad = self.hp.requires_grad
            param.requires_grad = requires_grad

    def forward(self, x):
        try:
            return self.e.forward(x)
        except BaseException as e:
            raise e


class FullyConv1Resnet(pl.LightningModule):
    """
    Would be a normal resnet but I don't want to force a known input size input size constant.

    https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035 for why the layers are ordered
    groupnorm, activation, fc
    """

    SKIP_SUFFIX = '_skip'
    DROP_SUFFIX = '_drop'
    BLOCK_SUFFIX = ''

    class ModelParams(BuilderParams):
        num_groups: int = 32
        num_blocks_per_residual: int = 2
        num_blocks_per_dropout: int = 2
        dropout_p: float = 0.5
        activation: nn.Module = nn.LeakyReLU
        do_include_first_norm: bool = True

        def build(self, num_input_features: int, neuron_counts: List[int], do_error_if_group_div_off=False):
            return FullyConv1Resnet(
                num_input_features=num_input_features,
                neuron_counts=neuron_counts,
                do_error_if_group_div_off=do_error_if_group_div_off,
                hp=self,
            )

    def __init__(
            self,
            num_input_features: int,
            neuron_counts: List[int],
            do_error_if_group_div_off=False,
            hp: Optional[ModelParams] = ModelParams(),
    ):
        super().__init__()
        self.hp = hp
        self.neuron_counts = neuron_counts
        all_counts = [num_input_features] + neuron_counts

        # round up counts so they're group divisible
        remainders = [count % self.hp.num_groups for count in all_counts]
        if any(remainders) and do_error_if_group_div_off:
            raise ValueError(
                f"Number of neurons ({all_counts}) must be divisible by number of groups ({self.hp.num_groups})")

        # neurons which are added to each layer to ensure that layer sizes are divisible by num_groups
        self._extra_counts = [self.hp.num_groups - r if r else 0 for r in remainders]

        all_counts = [c + e for c, e in zip(all_counts, self._extra_counts)]

        self._num_output_features = all_counts[-1]

        blocks = OrderedDict()
        prev_size = all_counts[0]
        for block_ind, (num_in, num_hidden) in enumerate(zip(all_counts, all_counts[1:])):
            do_include_norm = block_ind > 0 or self.hp.do_include_first_norm

            block = resnet_conv1_block(
                num_input_features=num_in,
                num_output_features=num_hidden,
                num_gn_groups=self.hp.num_groups,
                activation=self.hp.activation,
                do_include_group_norm=do_include_norm,
            )
            blocks[f'{block_ind}{self.BLOCK_SUFFIX}'] = block

            if (block_ind + 1) % self.hp.num_blocks_per_residual == 0:
                new_size = block.get_num_output_features()
                if new_size == prev_size:
                    layer = nn.Identity()
                else:
                    layer = nn.Conv1d(prev_size, new_size, 1)
                blocks[f'{block_ind}{self.SKIP_SUFFIX}'] = layer
                prev_size = new_size

            if (block_ind + 1) % self.hp.num_blocks_per_dropout == 0:
                blocks[f'{block_ind}{self.DROP_SUFFIX}'] = nn.Dropout(self.hp.dropout_p)

        self.blocks = nn.ModuleDict(blocks)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        if self._extra_counts[0] > 0:
            # the remainders fixup to all_counts in __init__  doesn't account for the fact that the input x may
            # still be the wrong size.  pad it up if needed
            x = F.pad(x, pad=(0, 0, 0, self._extra_counts[0]), mode='constant', value=0)

        old_x = x

        for name, block in self.blocks.items():
            if name.endswith(self.SKIP_SUFFIX):
                # this block is a resizer for the skip connection
                x += block(old_x)
                old_x = x
            else:
                try:
                    x = block(x)
                except BaseException as e:
                    raise e

        x = x.permute(0, 2, 1)
        return x

    def get_num_output_features(self):
        return self._num_output_features


class ConvBlock(pl.LightningModule):
    SKIP_SUFFIX = '_skip'

    class ModelParams(BuilderParams):
        num_layers: int = 8
        num_features: int = 32
        kernel_size: int = 3
        stride: int = 1
        pool_size: int = 2
        num_groups: int = 32
        do_sqrt_groups: bool = True
        num_blocks_per_pool: int = 2
        num_blocks_per_skip: int = 2
        activation: nn.Module = nn.LeakyReLU
        do_include_first_norm: bool = True
        requires_grad: bool = True

        def __post_init__(self):
            if self.do_sqrt_groups or self.num_groups > self.num_features:
                self.num_groups == 2 ** (np.log2(self.num_features) // 2)

        def build(self, num_input_features: int, do_error_if_group_div_off=False):
            return ConvBlock(
                num_input_features=num_input_features,
                hp=self,
                do_error_if_group_div_off=do_error_if_group_div_off,
            )

    def __init__(
            self,
            num_input_features: int,
            hp: 'ConvBlock.ModelParams',
            do_error_if_group_div_off: bool = False,
    ):
        super().__init__()
        self.hp = hp

        all_counts = [num_input_features] + ([self.hp.num_features] * int(self.hp.num_layers))

        # round up counts so they're group divisible
        remainders = [count % self.hp.num_groups for count in all_counts]
        if any(remainders) and do_error_if_group_div_off:
            raise ValueError(
                f"Number of neurons ({all_counts}) must be divisible by number of groups ({self.hp.num_groups})")

        # neurons which are added to each layer to ensure that layer sizes are divisible by num_groups
        self._extra_counts = [self.hp.num_groups - r if r else 0 for r in remainders]
        all_counts = [c + e for c, e in zip(all_counts, self._extra_counts)]

        blocks = OrderedDict()
        num_old_x_features = num_prev_features = all_counts[0]
        self._num_output_features = all_counts[-1]
        for block_ind, num_features in enumerate(all_counts[1:]):
            blocks[f'{block_ind}'] = resnet_conv1_block(
                num_input_features=num_prev_features,
                num_output_features=num_features,
                num_gn_groups=self.hp.num_groups,
                activation=self.hp.activation,
                kernel_size=self.hp.kernel_size,
                stride=self.hp.stride,
            )
            if (block_ind + 1) % self.hp.num_blocks_per_pool == 0:
                blocks[f'{block_ind}_pool'] = nn.MaxPool1d(kernel_size=self.hp.pool_size)

            if (block_ind + 1) % self.hp.num_blocks_per_skip == 0:
                if num_old_x_features == num_features:
                    layer = nn.Identity()
                else:
                    layer = nn.Conv1d(num_old_x_features, num_features, kernel_size=1)

                blocks[f'{block_ind}{self.SKIP_SUFFIX}'] = layer
                num_old_x_features = num_features
            num_prev_features = num_features

        self.blocks = nn.ModuleDict(blocks)
        for block in self.blocks.values():
            for name, param in block.named_parameters():
                param.requires_grad = self.hp.requires_grad

    def forward(self, x):
        x = x.permute(0, 2, 1)

        if self._extra_counts[0] > 0:
            x = F.pad(x, pad=(0, 0, 0, self._extra_counts[0]), mode='constant', value=0)

        old_x = x
        for name, block in self.blocks.items():
            if name.endswith(self.SKIP_SUFFIX):
                # this block is a resizer (or identity) for the skip connection
                x += block(old_x)
                old_x = x
            else:
                x = block(x)

        x = x.permute(0, 2, 1)

        return x

    def get_num_output_features(self):
        return self._num_output_features


class SlabNet(FullyConv1Resnet, Sized):
    """A fully convolutional resnet with constant number of features across layers."""

    class ModelParams(BuilderParams):
        num_features: int = 32
        num_layers: int = 4
        num_groups: int = 32
        do_sqrt_groups: bool = True
        num_blocks_per_residual: int = 2
        num_blocks_per_dropout: int = 2
        dropout_p: float = 0.5
        activation: nn.Module = nn.LeakyReLU
        do_include_first_norm: bool = True
        requires_grad: bool = True

        def __post_init__(self):
            # if self.num_features_per_group is not None:
            #     assert self.num_features % self.num_features_per_group == 0
            #     self.num_groups = self.num_features // self.num_features_per_group
            if self.do_sqrt_groups or self.num_groups > self.num_features:
                self.num_groups == 2 ** (np.log2(self.num_features) // 2)

        def build(self, num_input_features: int):
            return SlabNet(
                num_input_features=num_input_features,
                hp=self,
            )

    def __init__(
            self,
            num_input_features: int,
            hp: ModelParams = ModelParams(),
    ):
        super().__init__(
            num_input_features=num_input_features,
            neuron_counts=[hp.num_features] * hp.num_layers,
            hp=hp,
        )


def artless_smash(input: torch.Tensor) -> torch.Tensor:
    """
    Smash all vectors in time dimension into a series of summary layers.  cat them and work with that.
    This is about the simplest way you can move from O(n) time to O(1) time.
    If you want, make it better by starting with some conv layers.
    """
    first = input[:, 0, :].squeeze(1)
    means = torch.mean(input, dim=1, keepdim=True)
    vars = torch.var(input, dim=1, keepdim=True)
    l1s = torch.norm(input, p=1, dim=1, keepdim=True)
    lse = torch.logsumexp(input, dim=1, keepdim=True)

    maxs = torch.max(input, dim=1, keepdim=True)[0]
    mins = torch.min(input, dim=1, keepdim=True)[0]
    medians = torch.median(input, dim=1, keepdim=True)[0]

    return torch.cat((first, means, vars, l1s, lse, maxs, mins, medians), dim=1).view(input.shape[0], 1, -1)


class ArtlessSmasher(nn.Module):
    def __init__(self, num_input_channels: int):
        super().__init__()
        self.num_input_channels = num_input_channels

    @staticmethod
    def forward(x):
        return artless_smash(x)

    def get_num_output_features(self):
        return 8 * self.num_input_channels


def build_fast_trans_block(
        hp: 'TransBlockBuilder.ModelParams',
        num_input_features: int,
        get_decoder=False,
) -> nn.Module:
    from fast_transformers import builders, feature_maps
    s = TransBlockBuilder.split_impl_name(hp.impl)
    feature_map_name = None if len(s) == 1 else s[1]

    num_query_features = hp.num_query_features or num_input_features

    if feature_map_name == 'favor':
        feature_map = feature_maps.Favor.factory(n_dims=num_query_features)
    elif feature_map_name == 'grf':
        feature_map = feature_maps.GeneralizedRandomFeatures.factory(num_query_features)
    else:
        feature_map = None

    kwargs = {
        'n_layers': hp.num_layers,
        'n_heads': hp.num_heads,
        'query_dimensions': num_query_features // hp.num_heads,
        'value_dimensions': num_input_features // hp.num_heads,
        'feed_forward_dimensions': num_input_features * hp.fc_dim_mult,
        'activation': 'gelu',
        'feature_map': feature_map,
        'dropout': hp.p_dropout,
        'attention_dropout': hp.p_attention_dropout,
    }

    # Create the builder for our transformers
    if get_decoder:
        kwargs['attention_type'] = 'causal-linear'
        kwargs['cross_attention_type'] = 'linear'
        builder = builders.TransformerDecoderBuilder.from_kwargs(**kwargs)
        builder.final_normalization = False
    else:
        kwargs['attention_type'] = 'linear'
        builder = builders.TransformerEncoderBuilder.from_kwargs(**kwargs)
        builder.final_normalization = False

    return builder.get()


class Appender(nn.Module):
    """Append an array of zeros """
    def __init__(self, num_extra_dims: int, append_dim=-1, dtype=torch.float):
        super().__init__()
        self._num_extra_dims = num_extra_dims
        self._append_dim = append_dim
        self._dtype = dtype

    def forward(self, x):
        shape = [e for e in x.shape]
        shape[self._append_dim] = self._num_extra_dims
        extra = torch.zeros(*shape, requires_grad=False, dtype=self._dtype, device=x.device)
        return torch.cat(tensors=[x, extra], dim=self._append_dim)


def build_performer_trans_block(
        hp: 'TransBlockBuilder.ModelParams',
        num_input_features: int,
        get_decoder=False,
) -> nn.Module:
    if get_decoder:
        raise NotImplementedError('Impelment a decoder')

    """https://github.com/lucidrains/performer-pytorch"""
    from performer_pytorch import Performer
    return Performer(
        dim=num_input_features,
        depth=hp.num_layers,
        heads=hp.num_heads,
        ff_mult=hp.fc_dim_mult,
        nb_features=hp.num_query_features,
    )


def build_pytorch_trans_block(
        hp: 'TransBlockBuilder.ModelParams',
        num_input_features: int,
        get_decoder=False,
) -> nn.Module:
    if get_decoder:
        layer = nn.TransformerDecoderLayer(
            d_model=num_input_features,
            nhead=hp.num_heads,
            dim_feedforward=hp.fc_dim_mult * num_input_features,
            activation='gelu',
            dropout=hp.p_dropout,
        )

        return nn.TransformerDecoder(
            decoder_layer=layer,
            num_layers=hp.num_layers,
        )
    else:
        layer = nn.TransformerEncoderLayer(
            d_model=num_input_features,
            nhead=hp.num_heads,
            dim_feedforward=hp.fc_dim_mult * num_input_features,
            activation='gelu',
            dropout=hp.p_dropout,
        )

        return nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=hp.num_layers,
        )


def build_ablatable_trans_block(
        hp: 'TransBlockBuilder.ModelParams',
        num_input_features: int,
        get_decoder=False,
) -> nn.Module:
    if get_decoder:
        raise NotImplementedError('Impelment a decoder')

    do_drop_k = True

    return ablation.Encoder(
        d_model=num_input_features,
        num_layers=hp.num_layers,
        num_heads=hp.num_heads,
        p_dropout=hp.p_dropout,
        d_ff_mult=hp.fc_dim_mult,
        do_drop_k=do_drop_k,
    )


class TransBlockBuilder:
    class ModelParams(BuilderParams):
        impl: str = 'fast-favor'
        num_layers: int = 2
        num_heads: int = 8
        num_query_features: int = 32
        fc_dim_mult: int = 2
        p_dropout: int = 0.1
        p_attention_dropout: int = 0.1

        def build(self, num_input_features: int, get_decoder: bool=False) -> Any:
            return TransBlockBuilder.build(hp=self, num_input_features=num_input_features, get_decoder=get_decoder)

    IMPL_NAME_SPLIT_STR = '-'
    name = None

    KNOWN_ENCODER_BUILDERS = {
        'fast': build_fast_trans_block,
        'performer': build_performer_trans_block,
        'pytorch': build_pytorch_trans_block,
        'ablatable': build_ablatable_trans_block,
        'none': lambda: nn.Identity(),
    }

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @classmethod
    def split_impl_name(cls, enc_name):
        return enc_name.split(cls.IMPL_NAME_SPLIT_STR)

    @classmethod
    def build(cls, hp: ModelParams, num_input_features: int, get_decoder=False) -> SizedSequential:
        base_name = cls.split_impl_name(hp.impl)[0]

        if base_name not in cls.KNOWN_ENCODER_BUILDERS:
            raise ValueError(f"Couldn't find an encoder called {base_name}, "
                             f"only have: {list(cls.KNOWN_ENCODER_BUILDERS.keys())}.")

        builder_fn = cls.KNOWN_ENCODER_BUILDERS[base_name]

        remainder = num_input_features % hp.num_heads
        num_extra_dims = hp.num_heads - remainder if remainder else 0
        num_total_features = num_input_features + num_extra_dims

        d = OrderedDict()
        d['appender'] = Appender(num_extra_dims=num_extra_dims) if remainder else nn.Identity()
        d['trans'] = builder_fn(hp=hp, num_input_features=num_total_features, get_decoder=get_decoder)

        return SizedSequential(d, num_output_features=num_total_features)


class SplitAggCat(pl.LightningModule, Sized, abc.ABC):
    """Aggregate groups then cat results."""
    def __init__(self, num_input_features: int, aggregator_name: str, cat_dim: int=0):
        super().__init__()
        self.cat_dim = cat_dim

        self.splitter = Splitter(num_input_features=num_input_features, split_dim=self.cat_dim)
        self.agg, self.num_output_features = utils.get_reducer(
            num_input_features=num_input_features,
            aggregator_str=aggregator_name,
        )

    def forward(self, x: torch.Tensor, group_indices: torch.Tensor):
        xs = self.splitter(x, group_indices)
        return torch.cat([self.agg(x) for x in xs], dim=self.cat_dim)

    def get_num_output_features(self):
        return self.num_output_features


class ModelSplitAggCombineData(pl.LightningModule, Sized):
    """Group input array by grouping vector, run each group through a model, and aggregate each group at the end
        and concatenate the resulting post-aggregated vectors.

    It's basically a map reduce along the data dimension (or any other really) with a neural net in the middle.

    Inputs are sized:
        (num_docs, seq_len, num_input_features)
    Outputs are sized:
        (num_datapoints_per_batch, seq_len, num_output_features)

    Where num_docs is the sum of the number of docs for each datapoint in the batch.
    It's basically batch size but the collate fn expands multiple docs within each datapoint and cats all of the
        resulting docs across datapoints within the batch into the batch size dimension.
    """

    class ModelParams(BuilderParams):
        split_cat_dim: int = 0
        # TODO: re-generalize
        model: 'TransConvCatFc.ModelParams'
        agg_fn_name: str = 'mean-0-keep'

        def build(self, num_input_features: int, num_features_to_cat: int) -> 'ModelSplitAggCombineData':
            return ModelSplitAggCombineData(
                self,
                num_input_features=num_input_features,
                num_features_to_cat=num_features_to_cat,
            )

        @classmethod
        def from_dict(cls, d: Dict) -> 'ModelSplitAggCombineData.ModelParams':
            out = copy.deepcopy(cls(**d))
            out.model = TransConvCatFc.ModelParams.from_dict(out.model)
            return out


    def __init__(self, hp: ModelParams, num_input_features: int, num_features_to_cat: int, *args, **kwargs):
        super().__init__()
        self.hp = hp
        self.num_input_features = num_input_features
        self.num_base_features = num_features_to_cat

        # self.splitter = Splitter(num_input_features=num_input_features, split_dim=self.hp.split_cat_dim)
        self.model = self.hp.model.build(
            num_input_features=self.num_input_features,
            num_base_features=self.num_base_features,
            *args, **kwargs
        )
        self.sac = SplitAggCat(
            num_input_features=self.model.get_num_output_features(),
            aggregator_name=self.hp.agg_fn_name,
            cat_dim=self.hp.split_cat_dim,
        )
        self._num_output_features = self.sac.get_num_output_features()

    def forward(self, x: torch.Tensor, to_cat: torch.Tensor, group_indices: torch.Tensor):
        # xs = self.splitter(x=x, group_indices=group_indices)
        # to_cats = self.splitter(x=to_cat, group_indices=group_indices)
        # TODO: passing base into a general model like this is odd.  Maybe just pass in whole datapoint.
        # xs = [self.model(x, base=to_cat) for x, to_cat in zip(xs, to_cats)]
        x = self.model(x, base=to_cat)
        x = self.sac(x, group_indices)
        return x

    @staticmethod
    def groupby(data_tensor: torch.Tensor, doc_inds: torch.Tensor, split_dim=0) -> List[torch.Tensor]:
        # https://twitter.com/jeremyphoward/status/1185062637341593600
        idxs, vals = torch.unique(doc_inds, return_counts=True)
        split_arrays = torch.split_with_sizes(data_tensor, tuple(vals), dim=split_dim)

        doc_arrays = [None] * max(idxs)
        for idx, split_array in zip(idxs, split_arrays):
            doc_arrays[idx.item()] = split_array

        doc_arrays = [e for e in doc_arrays if e is not None]
        return doc_arrays

    def get_num_output_features(self):
        return self._num_output_features


class Splitter(SizedPlModule):
    def __init__(self, num_input_features: int, split_dim=0):
        super().__init__()
        self.num_input_features = num_input_features
        self.split_dim = split_dim

    def forward(self, x: torch.Tensor, group_indices: torch.Tensor):
        """group indices is a vector of group indices in [0, n)"""
        return self.splitby(data_tensor=x, group_indices=group_indices, split_dim=self.split_dim)

    @staticmethod
    def splitby(data_tensor: torch.Tensor, group_indices: torch.Tensor, split_dim=0) -> List[torch.Tensor]:
        # https://twitter.com/jeremyphoward/status/1185062637341593600
        idxs, vals = torch.unique(group_indices, return_counts=True)
        split_arrays = torch.split_with_sizes(data_tensor, tuple(vals), dim=split_dim)

        doc_tensors = []
        for idx, split_array in sorted(zip(idxs, split_arrays), key=lambda t: t[0]):
            doc_tensors.append(split_array)

        return doc_tensors

    def get_num_output_features(self):
        return self.num_input_features


class DupeModelCombineFeats(SizedPlModule):
    """Duplicate an input into each of several models then recombine them by catting along some dimension."""
    def __init__(
            self,
            num_input_features: int,
            model_name_to_params: Dict[str, SizedBuilderParams],
            cat_dim: int=constants.FEATURE_DIM,
    ):
        super().__init__()
        self.num_input_features = num_input_features
        self.cat_dim = cat_dim

        self.name_to_model = {}
        self.name_to_num_output_features = {}
        for model_name, model_params in model_name_to_params.items():
            model = model_params.build(num_input_features=num_input_features)
            self.name_to_model[model_name] = model
            self.name_to_num_output_features[model_name] = model.get_num_output_features()

        self.models = nn.ModuleDict(self.name_to_model)

        self._num_output_features = sum(self.name_to_num_output_features.values())

    def forward(self, x: torch.Tensor):
        xs = [model(x) for _, model in self.name_to_model.items()]
        return torch.cat(xs, dim=self.cat_dim)

    def _get_model_and_num_output_features(self, model_name: str, model_params: BuilderParams):
        if model_params.num_layers == 0:
            model = None
            num_features = 0
        else:
            model = self.hp.conv.build(num_input_features=num_embedcat_features)
            num_features = self.conv.get_num_output_features()

        return model, num_features

    def get_num_output_features(self):
        return self._num_output_features


class TransConvCatFc(SizedPlModule):
    class ModelParams(BuilderParams):
        conv: ConvBlock.ModelParams = ConvBlock.ModelParams()
        trans: TransBlockBuilder.ModelParams = TransBlockBuilder.ModelParams()
        fc: SlabNet.ModelParams = SlabNet.ModelParams()

        cat_dim: int = -1

        def build(
                self,
                num_input_features: int,
                num_base_features: int,
        ) -> 'TransConvCatFc':
            return TransConvCatFc(num_input_features, num_base_features, self)

        @classmethod
        def from_dict(cls, d: Dict) -> 'TransConvCatFc.ModelParams':
            out = copy.deepcopy(cls(**d))
            out.conv = ConvBlock.ModelParams.from_dict(out.conv)
            out.trans = TransBlockBuilder.ModelParams.from_dict(out.trans)
            out.fc = SlabNet.ModelParams.from_dict(out.fc)
            return out


    def __init__(self, num_input_features: int, num_base_features, hp: ModelParams):
        super().__init__()
        self.hp = hp

        self.tc = DupeModelCombineFeats(
            num_input_features=num_input_features,
            model_name_to_params={'conv': self.hp.conv, 'trans': self.hp.trans},
            cat_dim=self.hp.cat_dim,
        )
        self.fc = SlabNet(self.tc.get_num_output_features() + num_base_features, self.hp.fc)

    def forward(self, x: torch.Tensor, base: torch.Tensor):
        x = self.tc(x)
        x = torch.cat([x, base], dim=self.hp.cat_dim)
        x = self.fc(x)
        return x

    def get_num_output_features(self):
        return self.fc.get_num_output_features()


class EmbedCat(SizedPlModule):
    """Embed a vocab vector, then concatenate a feature vector."""
    class ModelParams(BuilderParams):
        embed: BertEmbedder.ModelParams = BertEmbedder.ModelParams()
        cat_dim: int = -1

        def build(self, num_features_to_cat: int, max_seq_len: int) -> 'EmbedCat':
            return EmbedCat(self, num_features_to_cat, max_seq_len=max_seq_len)

        @classmethod
        def from_dict(cls, d: Dict) -> 'EmbedCat.ModelParams':
            out = copy.deepcopy(cls(**d))
            out.embed = BertEmbedder.ModelParams.from_dict(out.embed)
            return out

    def __init__(
            self,
            hp: ModelParams,
            num_features_to_cat: int,
            max_seq_len: int,
    ):
        super().__init__()
        self.hp = hp
        self.num_features_to_cat = num_features_to_cat
        self.cat_dim = self.hp.cat_dim

        self.embed: BertEmbedder = self.hp.embed.build(max_seq_len=max_seq_len)

    def forward(self, vocab: torch.Tensor, to_cat: torch.Tensor):
        assert to_cat.shape[self.cat_dim] == self.num_features_to_cat
        x = self.embed(vocab).last_hidden_state
        assert x.shape[self.cat_dim] == self.hp.embed.dim, f'x.shape: {x.shape}, hp.dim: {self.hp.dim}'
        return torch.cat([x, to_cat], dim=self.cat_dim)

    def get_num_output_features(self):
        return self.hp.embed.dim + self.num_features_to_cat
