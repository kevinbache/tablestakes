from typing import Any, List

import torch
from pytorch_lightning.metrics import TensorMetric
from tablestakes.ml import hyperparams, ablation
from torch import nn as nn


class WordAccuracy(TensorMetric):
    def __init__(
            self,
            reduce_group: Any = None,
    ):
        super().__init__('acc', reduce_group)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        is_correct = target.view(-1) == torch.argmax(pred, dim=-1).view(-1)
        return is_correct.float().mean()


class SizedSequential(nn.Sequential):
    def __init__(self, *args, num_output_features):
        super().__init__(*args)

        self._num_output_features = num_output_features

    def get_num_output_features(self):
        return self._num_output_features


def resnet_conv1_block(
        num_input_features: int,
        num_hidden_features: int,
        activation=nn.LeakyReLU,
):
    return SizedSequential(
        nn.BatchNorm1d(num_input_features),
        activation(),
        nn.Conv1d(num_input_features, num_hidden_features, 1),
        num_output_features=num_hidden_features,
    )


class FullyConv1Resnet(nn.Module):
    """
    Would be a normal resnet but I don't want to fix the size.

    https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
    """
    def __init__(
            self,
            num_input_features: int,
            neuron_counts: List[int],
            num_blocks_per_residual=2,
            activation=nn.LeakyReLU,
    ):
        super().__init__()
        self.neuron_counts = neuron_counts
        self.num_blocks_per_residual = num_blocks_per_residual
        self.activation = activation

        all_counts = [num_input_features] + neuron_counts
        self._num_output_features = all_counts[-1]

        self.blocks = nn.ModuleList([
            resnet_conv1_block(num_in, num_hidden, self.activation)
            for num_in, num_hidden in zip(all_counts, all_counts[1:])
        ])

        resizers = []
        prev_size = all_counts[0]
        for block_ind, block in enumerate(self.blocks):
            if block_ind and block_ind % self.num_blocks_per_residual == 0:
                new_size = block.get_num_output_features()
                if new_size == prev_size:
                    continue
                resizers.append(nn.Conv1d(prev_size, new_size, 1))
                prev_size = new_size
            else:
                resizers.append(None)

        self.resizers = nn.ModuleList(resizers)

    def forward(self, x):
        old_x = x
        for block_index, (block, resizer) in enumerate(zip(self.blocks, self.resizers)):
            x = block(x)
            if block_index and block_index % self.num_blocks_per_residual == 0:
                x += resizer(old_x)
                old_x = x
        return x

    def get_num_outputs(self):
        return self._num_output_features


class ReachAroundTransformer(nn.Module):
    """x_base and x_vocab are inputs to the transformer and a raw x_base is also appended to the output"""
    def __init__(self, blah):
        super().__init__()


def get_fast_linear_attention_encoder(hp: hyperparams.LearningParams, num_trans_input_dims: int):
    """https://github.com/idiap/fast-transformers"""
    from fast_transformers.builders import TransformerEncoderBuilder

    # Create the builder for our transformers
    builder = TransformerEncoderBuilder.from_kwargs(
        n_layers=hp.num_trans_enc_layers,
        n_heads=hp.num_trans_heads,
        query_dimensions=num_trans_input_dims // hp.num_trans_heads,
        value_dimensions=num_trans_input_dims // hp.num_trans_heads,
        feed_forward_dimensions=num_trans_input_dims * hp.num_trans_fc_dim_mult,
        attention_type='linear',
        activation='gelu',
    )

    # Build a transformer with linear attention
    return builder.get()


def get_performer_encoder(hp: hyperparams.LearningParams, num_trans_input_dims: int):
    """https://github.com/lucidrains/performer-pytorch"""
    from performer_pytorch import Performer
    return Performer(
        dim=num_trans_input_dims,
        depth=hp.num_trans_enc_layers,
        heads=hp.num_trans_heads,
        ff_mult=hp.num_trans_fc_dim_mult,
    )


def get_pytorch_transformer_encoder(hp: hyperparams.LearningParams, num_trans_input_dims: int):
    enc_layer = nn.TransformerEncoderLayer(
        d_model=num_trans_input_dims,
        nhead=hp.num_trans_heads,
        dim_feedforward=hp.num_trans_fc_dim_mult * num_trans_input_dims,
        activation='gelu'
    )

    return nn.TransformerEncoder(
        encoder_layer=enc_layer,
        num_layers=hp.num_trans_enc_layers,
    )


def get_simple_ablatable_transformer_encoder(
        hp: hyperparams.LearningParams,
        num_trans_input_dims: int,
        do_drop_k=True,
):
    return ablation.Encoder(
        d_model=num_trans_input_dims,
        num_layers=hp.num_trans_enc_layers,
        num_heads=hp.num_trans_heads,
        d_ff_mult=hp.num_trans_fc_dim_mult,
        do_drop_k=do_drop_k,
    )