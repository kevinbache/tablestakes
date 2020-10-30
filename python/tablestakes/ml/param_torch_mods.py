import glob
from argparse import Namespace
from typing import *

from torch import nn as nn
from torch.nn import functional as F

import transformers

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from chillpill import params

from tablestakes import utils


PS = TypeVar('PS', bound=params.ParameterSet)


class Parametrized(Generic[PS]):
    def __init__(self, hp: PS):
        super().__init__()
        self.hp = hp


class ParametrizedModule(Parametrized, nn.Module, Generic[PS]):
    def __init__(self, hp: PS):
        Parametrized.__init__(self, hp)
        nn.Module.__init__(self)


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


class BertEmbedder(ParametrizedModule['BertEmbedder.ModelParams']):
    class ModelParams(params.ParameterSet):
        dim = 64
        max_seq_len = 1024
        requires_grad = True

    def __init__(self, hp: Optional[ModelParams] = ModelParams()):
        super().__init__(hp)

        config = transformers.BertConfig(
            hidden_size=self.search_params.dim,
            num_hidden_layers=0,
            num_attention_heads=1,
            intermediate_size=0,
            max_position_embeddings=self.search_params.max_seq_len,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        self.e = transformers.BertModel(config, add_pooling_layer=False)
        for name, param in self.e.named_parameters():
            param.requires_grad = self.search_params.requires_grad

    def forward(self, x):
        return self.e.forward(x)


class FullyConv1Resnet(ParametrizedModule['FullyConv1Resnet.ModelParams']):
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

        super().__init__(hp)
        self.neuron_counts = neuron_counts

        # self.num_blocks_per_residual = num_blocks_per_residual
        # self.activation = activation

        all_counts = [num_input_features] + neuron_counts

        # round up counts so they're group divisible
        remainders = [count % self.search_params.num_groups for count in all_counts]
        if any(remainders) and do_error_if_group_div_off:
            raise ValueError(
                f"Number of neurons ({all_counts}) must be divisible by number of groups ({self.search_params.num_groups})")

        # neurons which are added to each layer to ensure that layer sizes are divisible by num_groups
        self._extra_counts = [self.search_params.num_groups - r if r else 0 for r in remainders]

        all_counts = [c + e for c, e in zip(all_counts, self._extra_counts)]

        self._num_output_features = all_counts[-1]

        blocks = OrderedDict()
        prev_size = all_counts[0]
        for block_ind, (num_in, num_hidden) in enumerate(zip(all_counts, all_counts[1:])):
            do_include_norm = block_ind > 0 or self.search_params.do_include_first_norm

            block = resnet_conv1_block(
                num_input_features=num_in,
                num_output_features=num_hidden,
                num_gn_groups=self.search_params.num_groups,
                activation=self.search_params.activation,
                do_include_group_norm=do_include_norm,
            )
            blocks[f'{block_ind}{self.BLOCK_SUFFIX}'] = block

            if (block_ind + 1) % self.search_params.num_blocks_per_residual == 0:
                new_size = block.get_num_output_features()
                if new_size == prev_size:
                    layer = nn.Identity()
                else:
                    layer = nn.Conv1d(prev_size, new_size, 1)
                blocks[f'{block_ind}{self.SKIP_SUFFIX}'] = layer
                prev_size = new_size

            if (block_ind + 1) % self.search_params.num_blocks_per_dropout == 0:
                blocks[f'{block_ind}{self.DROP_SUFFIX }'] = nn.Dropout(self.search_params.dropout_p)

        self.blocks = nn.ModuleDict(blocks)

    def forward(self, x):
        if self._extra_counts[0] > 0:
            old_x = F.pad(x, pad=(0, 0, 0, self._extra_counts[0]), mode='constant', value=0)
        else:
            old_x = x

        for (name, block), extra in zip(self.blocks.items(), self._extra_counts):
            # the remainders fixup to all_counts in __init__  doesn't account for the fact that the input x may
            # still be the wrong size.  pad it up if needed
            x = F.pad(x, pad=(0, 0, 0, extra), mode='constant', value=0)

            if name.endswith(self.SKIP_SUFFIX):
                # this block is a resizer for the skip connection
                x += block(old_x)
                old_x = x
            else:
                x = block(x)

        return x

    def get_num_outputs(self):
        return self._num_output_features


class ConvBlock(ParametrizedModule['ConvBlock.ModelParams']):
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

        super().__init__(hp)
        self.hp = hp

        all_counts = [num_input_features] + [self.hp.num_features] * self.hp.num_layers

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
        if self._extra_counts[0] > 0:
            old_x = F.pad(x, pad=(0, 0, 0, self._extra_counts[0]), mode='constant', value=0)
        else:
            old_x = x

        for extra_count, (name, block) in zip(self._extra_counts, self.blocks.items()):
            if extra_count > 0:
                x = F.pad(x, pad=(0, 0, 0, extra_count), mode='constant', value=0)

            if name.endswith(self.SKIP_SUFFIX):
                # this block is a resizer (or identity) for the skip connection
                x += block(old_x)
                old_x = x
            else:
                x = block(x)

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


class HeadedSlabNet(SlabNet):
    """A slabnet with a head containing num_output_features."""

    def __init__(
            self,
            num_input_features: int,
            num_output_features: int,
            hp: Optional[SlabNet.ModelParams] = SlabNet.ModelParams(),
    ):
        super().__init__(
            num_input_features=num_input_features,
            hp=hp,
        )
        # could be changed by GroupNorm fixup
        num_slab_outputs = self.get_num_outputs()

        self.head = nn.Conv1d(num_slab_outputs, num_output_features, 1)
        self._num_output_features = num_output_features

    def forward(self, x):
        x = super().forward(x)
        x = self.head(x)
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
    def __init__(self, hp: ExperimentParams, version: str = ''):
        source_files = glob.glob(str(hp.sources_glob_str), recursive=True)

        super().__init__(
            api_key=utils.get_logger_api_key(),
            project_name=utils.get_neptune_fully_qualified_project_name(hp.project_name),
            close_after_fit=True,
            offline_mode=False,
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


def get_pl_logger(hp: ExperimentParams, tune=None):
    version = 'local' if tune is None else tune.get_trial_id()
    logger = MyLightningNeptuneLogger(hp, version),

    return logger
