import abc
import glob
from argparse import Namespace
from typing import *


import torch
from torch import nn as nn
from torch.nn import functional as F

import transformers

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from chillpill import params

from tablestakes import utils
from tablestakes.ml import ablation

PS = TypeVar('PS', bound=params.ParameterSet)


class Parameterized(Generic[PS]):
    # def __init__(self, hp: PS):
    #     super().__init__()
    #     self.hp = hp

    # def get_params(self) -> PS:
    #     raise NotImplementedError('Implement it in your subclass')
    pass


class ParameterizedModule(Parameterized, nn.Module, Generic[PS]):
    # def __init__(self, hp: PS):
    #     nn.Module.__init__(self)
    pass


class SizedSequential(nn.Sequential):
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


class BertEmbedder(ParameterizedModule['BertEmbedder.ModelParams']):
    class ModelParams(params.ParameterSet):
        dim = 64
        max_seq_len = 1024
        requires_grad = True

    def __init__(self, hp: Optional[ModelParams] = ModelParams()):
        super().__init__()
        self.hp = hp

        config = transformers.BertConfig(
            hidden_size=self.hp.dim,
            num_hidden_layers=0,
            num_attention_heads=1,
            intermediate_size=0,
            max_position_embeddings=self.hp.max_seq_len,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        self.e = transformers.BertModel(config, add_pooling_layer=False)
        for name, param in self.e.named_parameters():
            if name == 'position_embeddings':
                requires_grad = False
            else:
                requires_grad = self.hp.requires_grad
            param.requires_grad = requires_grad

    def forward(self, x):
        return self.e.forward(x)


class FullyConv1Resnet(ParameterizedModule['FullyConv1Resnet.ModelParams']):
    """
    Would be a normal resnet but I don't want to force a known input size input size constant.

    https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035 for why the layers are ordered
    groupnorm, activation, fc
    """

    SKIP_SUFFIX = '_skip'
    DROP_SUFFIX = '_drop'
    BLOCK_SUFFIX = ''

    class ModelParams(params.ParameterSet):
        num_groups = 32
        num_blocks_per_residual = 2
        num_blocks_per_dropout = 2
        dropout_p = 0.5
        activation = nn.LeakyReLU
        do_include_first_norm = True

        @classmethod
        def search_default(cls) -> "FullyConv1Resnet.ModelParams":
            return cls(
                num_groups=params.Discrete([8, 16, 32, 64]),
                num_blocks_per_residual=params.Integer(1, 5),
                activation=params.Categorical([nn.LeakyReLU, nn.GELU]),
                do_include_first_norm=params.Boolean(p_true=0.8),
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
                x = block(x)

        x = x.permute(0, 2, 1)
        return x

    def get_num_outputs(self):
        return self._num_output_features


class ConvBlock(ParameterizedModule['ConvBlock.ModelParams']):
    SKIP_SUFFIX = '_skip'

    class ModelParams(params.ParameterSet):
        num_layers = 8
        num_features = 32
        kernel_size = 3
        stride = 1
        pool_size = 2
        num_groups = 32
        num_blocks_per_pool = 2
        num_blocks_per_skip = 2
        activation = nn.LeakyReLU
        do_include_first_norm = True
        requires_grad = True

    def __init__(
            self,
            num_input_features: int,
            hp: Optional[ModelParams] = None,
            do_error_if_group_div_off: bool = False,
    ):
        if hp is None:
            hp = self.ModelParams()

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


class SlabNet(FullyConv1Resnet):
    """A fully convolutional resnet with constant number of features across layers."""

    class ModelParams(params.ParameterSet):
        num_features = 32
        num_layers = 4
        num_groups = 32
        num_blocks_per_residual = 2
        num_blocks_per_dropout = 2
        dropout_p = 0.5
        activation = nn.LeakyReLU
        do_include_first_norm = True
        requires_grad = True
        special_heads = {}

        @classmethod
        def search_default(cls) -> "SlabNet.ModelParams":
            return cls(
                num_features=params.Categorical([16, 32, 64, 128, 256]),
                num_layers=params.Integer(1, 5),
                **FullyConv1Resnet.ModelParams.search_default().__dict__,
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


class Head(nn.Module):
    def __init__(
            self,
            num_input_features: int,
            num_output_features: int,
    ):
        nn.Module.__init__(self)
        self.head = nn.Conv1d(num_input_features, num_output_features, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.head(x).permute(0, 2, 1)


def make_lm_loss(num_features, num_vocab):
    """Linear head + loss"""
    num_first_bin = round(num_vocab / 20)
    return nn.AdaptiveLogSoftmaxWithLoss(
        num_features,
        num_vocab,
        cutoffs=[
            num_first_bin,
            5 * num_first_bin,
        ],
        div_value=4,
    )


class AdaptiveSoftmaxHead(Head):
    def __init__(
            self,
            num_input_features: int,
            num_output_features: int,
    ):
        nn.Module.__init__(self)
        self.head = make_lm_loss(num_input_features, num_output_features)

    def forward(self, x):
        return self.head(x)


class StartEndHead(Head):
    """Predict the start and end location within a sequence.  Prediction must be positive."""
    def __init__(
            self,
            num_input_features: int,
            num_classes: int,
    ):
        nn.Module.__init__(self)
        self.start = nn.Conv1d(
            in_channels=num_input_features,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
        )

        self.end = nn.Conv1d(
            in_channels=num_input_features,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x):
        relu = nn.ReLU()
        return {
            'start_logits': relu(self.start(x)).permute(0, 2, 1),
            'end_logits': relu(self.end(x)).permute(0, 2, 1),
        }


class HeadedSlabNet(SlabNet):
    """A slabnet with a head containing num_output_features."""
    def __init__(
            self,
            num_input_features: int,
            head_maker: Optional[Callable[[int], Head]] = None,
            hp: Optional[SlabNet.ModelParams] = SlabNet.ModelParams(),
    ):
        super().__init__(
            num_input_features=num_input_features,
            hp=hp,
        )
        # could be changed by GroupNorm fixup
        self.head = head_maker(num_input_features=self.get_num_outputs(), )
        self._num_output_features = num_output_features

    def forward(self, x):
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        x = self.head(x)
        if isinstance(x, MutableMapping):
            x = {k: v.permute(0, 2, 1) for k, v in x.items()}
        else:
            x = x.permute(0, 2, 1)
        return x


class ExperimentParams(params.ParameterSet):
    project_name = 'my_project'
    experiment_name = 'my_experiment'
    experiment_tags = ['testing']
    sources_glob_str = '*.py'

    def get_project_exp_name(self):
        return f'{self.project_name}-{self.experiment_name}'


# noinspection PyProtectedMember
class MyLightningNeptuneLogger(pl_loggers.NeptuneLogger):
    def __init__(self, hp: ExperimentParams, version: str = '', offline_mode=False):
        source_files = glob.glob(str(hp.sources_glob_str), recursive=True)

        super().__init__(
            api_key=utils.get_logger_api_key(),
            project_name=utils.get_neptune_fully_qualified_project_name(hp.project_name),
            close_after_fit=True,
            offline_mode=offline_mode,
            experiment_name=f'pl_log-{version}',
            params=hp.to_dict(),
            tags=hp.experiment_tags,
            upload_source_files=source_files,
        )
        self.append_tags(hp.experiment_tags)

    @pl.utilities.rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        params = self._sanitize_params(params)
        return self.set_properties(params)

    def set_properties(self, new_properties: Dict):
        properties = self.experiment._backend.get_experiment(self.experiment.internal_id).properties
        properties = {p.key: p.value for p in properties}
        properties.update({k: str(v) for k, v in new_properties.items()})
        return self.experiment._backend.update_experiment(
            experiment=self.experiment,
            properties=properties,
        )


def get_pl_logger(hp: ExperimentParams, tune=None, offline_mode=False):
    version = 'local' if tune is None else tune.get_trial_id()
    logger = MyLightningNeptuneLogger(hp=hp, version=version, offline_mode=offline_mode),

    return logger


def artless_smash(input: torch.Tensor) -> torch.Tensor:
    """
    Smash all vectors in time dimension into a series of summary layers.  cat them and work with that.
    This is about the simplest way you can move from O(n) time to O(1) time.
    If you want, make it better by starting with some conv layers.
    """
    means = torch.mean(input, dim=1, keepdim=True)
    vars = torch.var(input, dim=1, keepdim=True)
    l1s = torch.norm(input, p=1, dim=1, keepdim=True)
    l2s = torch.norm(input, p=2, dim=1, keepdim=True)
    lse = torch.logsumexp(input, dim=1, keepdim=True)

    maxs = torch.max(input, dim=1, keepdim=True)[0]
    mins = torch.min(input, dim=1, keepdim=True)[0]
    medians = torch.median(input, dim=1, keepdim=True)[0]

    return torch.cat((means, vars, l1s, l2s, lse, maxs, mins, medians), dim=1).view(input.shape[0], 1, -1)


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
    s = TransBlockBuilder.split(hp.impl)
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
    else:
        kwargs['attention_type'] = 'linear'
        builder = builders.TransformerEncoderBuilder.from_kwargs(**kwargs)

    return builder.get()


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


class TransBlockBuilder(abc.ABC):
    class ModelParams(params.ParameterSet):
        impl = 'fast-favor'
        num_layers = 2
        num_heads = 8
        num_query_features = 32
        fc_dim_mult = 2
        p_dropout = 0.1
        p_attention_dropout = 0.1

    IMPL_NAME_SPLIT_STR = '-'
    name = None

    KNOWN_ENCODER_BUILDERS = {
        'fast': build_fast_trans_block,
        'performer': build_performer_trans_block,
        'pytorch': build_pytorch_trans_block,
        'ablatable': build_ablatable_trans_block,
    }

    def __init__(self, name, hp: ModelParams):
        super().__init__()
        self.hp = hp
        self.name = name

    @classmethod
    def split(cls, enc_name):
        return enc_name.split(cls.IMPL_NAME_SPLIT_STR)

    @classmethod
    def build(cls, hp: ModelParams, num_input_features: int, get_decoder=False) -> nn.Module:
        base_name = TransBlockBuilder.split(hp.impl)[0]

        if base_name not in cls.KNOWN_ENCODER_BUILDERS:
            raise ValueError(f"Couldn't find an encoder called {base_name}, "
                             f"only have: {list(cls.KNOWN_ENCODER_BUILDERS.keys())}.")

        builder_fn = cls.KNOWN_ENCODER_BUILDERS[base_name]
        return builder_fn(hp=hp, num_input_features=num_input_features, get_decoder=get_decoder)

