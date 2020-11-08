"""Simple transformer based on https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def attention(q, k, v, d_k, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_mult=4, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_model * d_mult)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_model * d_mult, d_model)

    def forward(self, x):
        x = self.dropout(F.gelu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1, do_drop_k=True):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.do_drop_k = do_drop_k

        # self.q_linear = nn.Linear(d_model, d_model)
        # self.v_linear = nn.Linear(d_model, d_model)

        self.qv_linear = nn.Linear(d_model, 2 * d_model)

        if self.do_drop_k:
            self.k_linear = nn.Identity()
        else:
            self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def forward(self, q, k, v):
        num_batch = q.size(0)

        # perform linear operation and split into h sub_losses

        k = self.k_linear(k).view(num_batch, -1, self.h, self.d_k)
        q, v = self.qv_linear(q).view(num_batch, -1, self.h * 2, self.d_k).chunk(2, dim=-2)
        # v = self.v_linear(v).view(num_batch, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, self.dropout)

        # concatenate sub_losses and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(num_batch, -1, self.d_model)

        output = self.out(concat)

        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, d_ff_mult=4, do_drop_k=True):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, do_drop_k=do_drop_k)
        self.dropout_1 = nn.Dropout(dropout)
        self.norm_2 = Norm(d_model)
        self.ff = FeedForward(d_model, d_ff_mult)
        self.dropout_2 = nn.Dropout(dropout)

        self._reset_parameters()

    def forward(self, x):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


def get_clones(module, num_layers):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, p_dropout, d_ff_mult, do_drop_k=True):
        super().__init__()
        self.num_layers = num_layers
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(
            EncoderLayer(d_model, num_heads, dropout=p_dropout, d_ff_mult=d_ff_mult, do_drop_k=do_drop_k),
            num_layers,
        )
        self.norm = Norm(d_model)

    def forward(self, x):
        # x = self.embed(src)
        # x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
