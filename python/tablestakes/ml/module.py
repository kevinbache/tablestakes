import math
from pathlib import Path
from typing import Any, Optional, List, Dict, Union

import utils

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.metrics import TensorMetric, Metric

from tablestakes import constants
from tablestakes.ml import data
from tablestakes.ml.hyperparams import MyHyperparams


class WordAccuracy(TensorMetric):
    def __init__(
            self,
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        super().__init__('word_acc', reduce_group, reduce_op)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        is_correct = target.view(-1) == torch.argmax(pred, dim=-1).view(-1)
        return is_correct.float().mean()


class TrapezoidConv1Module(pl.LightningModule):
    # LOSS_NAMES = ['korv', 'which_kv']
    LOSS_NAMES = ['korv']
    LOSS_VAL_NAME = 'loss'
    WORD_ACC_VAL_NAME = 'word_acc'

    # todo: really i need to split the datapoints first and then calculate the vocab

    @staticmethod
    def _recalc_vocab_size(ds: data.XYCsvDataset):
        return torch.stack([xs[1].max() for xs in ds._xs]).max().item()

    def __init__(
            self,
            hp: MyHyperparams,
            data_dir: Union[Path, str],
    ):
        super().__init__()

        self.data_dir = data_dir

        self.metrics = [WordAccuracy()]

        self.num_y_classes = utils.load_json(data_dir / constants.NUM_Y_CLASSES_FILENAME)
        self.word_to_id = utils.load_json(data_dir / constants.WORD_ID_FILENAME)
        self.word_to_count = utils.load_json(data_dir / constants.WORD_COUNT_FILENAME)

        self.hp = hp
        self.ds = data.XYCsvDataset(self.data_dir, limit_num_data=self.hp.limit_num_data)

        if hp.limit_num_data:
            self.num_vocab = self._recalc_vocab_size(self.ds)
        else:
            self.num_vocab = len(self.word_to_id)

        self.hp.num_vocab = self.num_vocab

        self.example_input_array = [
            torch.tensor(np.random.rand(1, 200, self.ds.num_x_dims[0]), dtype=torch.float),
            torch.tensor(np.random.rand(1, 200, self.ds.num_x_dims[1]), dtype=torch.long),
        ]

        self.hp.num_x_dims = self.ds.num_x_dims
        self.hp.num_y_dims = self.ds.num_y_dims

        # or implement only in gradient averaging
        assert self.hp.batch_size_log2 == 0
        self.hp.batch_size = int(math.pow(2, self.hp.batch_size_log2))

        # save all variables in __init__ signature to self.hparams
        self.hparams = self.hp.to_dict()
        self.save_hyperparameters()

        self.train_dataset, self.valid_dataset, self.test_dataset = None, None, None

        # self.net = nn.Linear(self.hp.num_x_dims[0], self.num_y_classes['korv'])

        #############
        # emb
        if hp.do_include_embeddings:
            self.embedder = nn.Embedding(self.num_vocab, self.hp.num_embedding_dim)

        #############
        # conv
        #  alright, so the logspace makes it really more of a funnel than a trapezoid
        num_conv_filters = np.logspace(
            start=hp.log2num_filters_start,
            stop=hp.log2num_filters_end,
            num=hp.num_conv_layers,
            base=2,
        ).astype(np.integer)

        conv_layers = []
        prev_num_filters = hp.num_x_dims[0]
        if hp.do_include_embeddings:
            prev_num_filters += hp.num_embedding_dim

        for conv_ind, num_filters in enumerate(num_conv_filters):
            conv_layers.append(
                nn.Conv1d(prev_num_filters, num_filters, hp.kernel_size, padding=int(hp.kernel_size/2)))
            conv_layers.append(nn.ReLU())
            prev_num_filters = num_filters

            # if not conv_ind % hp.num_conv_layers_per_pool:
            #     conv_layers.append(nn.MaxPool1d(hp.pool_size))

        # so torch can find your parameters
        self.conv_layers = nn.ModuleList(conv_layers)

        #############
        # fc
        fc_layers = []

        prev_num_neurons = prev_num_filters
        num_fc_neurons = np.logspace(
            start=hp.log2num_neurons_start,
            stop=hp.log2num_neurons_end,
            num=hp.num_fc_hidden_layers,
            base=2,
        ).astype(np.integer)
        for fc_ind, num_neurons in enumerate(num_fc_neurons):
            fc_layers.append(nn.Conv1d(prev_num_neurons, num_neurons, 1))
            fc_layers.append(nn.ReLU())
            prev_num_neurons = num_neurons

            # TODO: reenable dropout
            if not fc_ind % hp.num_fc_layers_per_dropout:
                fc_layers.append(nn.Dropout(p=hp.dropout_p))

        self.fc_layers = nn.ModuleList(fc_layers)

        # output
        self.heads = nn.ModuleList([
            nn.Conv1d(prev_num_neurons, self.num_y_classes[name], 1)
            for name in self.LOSS_NAMES
        ])

    def forward(self, x_base, vocab_ids):
        # x_base, vocab_ids = x
        x_base = x_base.float()

        if hp.do_include_embeddings:
            emb = self.embedder(vocab_ids)
            # squeeze out the "time" dimension we expect for language models
            emb = emb.squeeze(2)

            x = torch.cat([x_base, emb], dim=-1)
        else:
            x = x_base.float()

        # batch x word x feature --> batch x feature x word
        x = x.permute([0, 2, 1])

        for layer in self.conv_layers:
            x = layer(x)

        for layer in self.fc_layers:
            x = layer(x)

        y_hats = [layer(x).permute([0, 2, 1]) for layer in self.heads]

        # y_hats = self.net(x_base)

        return y_hats

    def _inner_forward_step(self, batch):
        xs, ys = batch
        y_hats = self(*xs)

        # losses = []
        # for y, y_hat in zip(ys, y_hats):
        #     this_loss = F.cross_entropy(y_hat.squeeze(0), y.view(-1))
        #     print(f'this_loss: {this_loss}')
        #     losses.append(this_loss)
        # losses = torch.tensor(losses, dtype=torch.float, requires_grad=True)
        #
        # # losses = torch.tensor([
        # #     F.cross_entropy(y_hat.squeeze(0), y.view(-1))
        # #     for y, y_hat in zip(ys, y_hats)
        # # ], requires_grad=True)
        # weights = torch.tensor(hp.loss_weights, dtype=torch.float, requires_grad=True)
        # print()
        # print('losses1:')
        # print(losses)
        # losses = torch.mul(losses, weights)
        # print()
        # print('losses2:')
        # print(losses)
        #
        # loss = losses.sum()
        # print('loss:')
        # print(loss)

        y = ys[0]
        y_hat = y_hats[0]
        # print('y.shape: ', y.shape)
        # print('y_hat.shape: ', y_hat.shape)
        # ysq = y.squeeze(2).squeeze(0)
        # print('ysq.shape: ', ysq.shape)

        weights = torch.tensor(self.hp.loss_weights, dtype=torch.float, requires_grad=False)
        loss = torch.tensor([F.cross_entropy(y_hat.squeeze(0), y.view(-1)), torch.tensor([0])], requires_grad=True)
        print()
        print('loss1:', loss)
        loss = weights * loss
        print()
        print('loss:2', loss)
        print()
        losses = torch.tensor([0, 0])
        # print('losses:', losses)

        return xs, ys, y_hats, losses, loss

    TRAIN_PHASE_NAME = 'train'
    VALID_PHASE_NAME = 'valid'
    TEST_PHASE_NAME = 'test'

    @staticmethod
    def _make_phase_name(phase_name: str, metric_name: str, loss_name: Optional[str] = None):
        out = f'{phase_name}_{metric_name}'
        if loss_name is not None:
            out += f'_{loss_name}'
        return out

    @classmethod
    def _make_metrics_dict(
            cls,
            ys: List[torch.Tensor],
            y_hats: List[torch.Tensor],
            loss_names: List[str],
            metrics: List[Metric],
            phase_name: str,
    ):
        out = {}
        for y, y_hat, loss_name in zip(ys, y_hats, loss_names):
            out.update(cls._make_metrics_dict_one_loss(y, y_hat, loss_name, metrics, phase_name))
        return out

    @classmethod
    def _make_metrics_dict_one_loss(
            cls,
            y: torch.Tensor,
            y_hat: torch.Tensor,
            loss_name: str,
            metrics: List[Metric],
            phase_name: str,
    ):
        return {
            cls._make_phase_name(phase_name, metric.name, loss_name): metric(y_hat, y)
            for metric in metrics
        }

    @classmethod
    def _make_losses_dict(cls, losses, loss_names, phase_name):
        return {
            cls._make_phase_name(phase_name, cls.LOSS_VAL_NAME, loss_name): loss.item()
            for loss, loss_name in zip(losses, loss_names)
        }

    @classmethod
    def _make_log_dict(
            cls,
            ys: List[torch.Tensor],
            y_hats: List[torch.Tensor],
            losses: torch.Tensor,
            loss: torch.Tensor,
            loss_names: List[str],
            metrics: List[Metric],
            phase_name: str,
    ):
        out = cls._make_metrics_dict(ys, y_hats, loss_names, metrics, phase_name)
        out.update(cls._make_losses_dict(losses, loss_names, phase_name))
        out[cls._make_phase_name(phase_name, cls.LOSS_VAL_NAME)] = loss
        return out

    @staticmethod
    def _itemize_dict(d: Dict):
        for k, v in d:
            if isinstance(v, torch.Tensor):
                d[k] = v.item()
        return d

    def training_step(self, batch, batch_idx):
        xs, ys, y_hats, losses, loss = self._inner_forward_step(batch)
        logs = self._make_log_dict(ys, y_hats, losses, loss, self.LOSS_NAMES, self.metrics, self.TRAIN_PHASE_NAME)
        # why is loss different than train_loss on the prog bar?  running avg?
        # print(logs)
        # print(loss)
        # print('--------------')
        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        xs, ys, y_hats, losses, loss = self._inner_forward_step(batch)
        return  self._make_log_dict(ys, y_hats, losses, loss, self.LOSS_NAMES, self.metrics, self.VALID_PHASE_NAME)

    # def validation_epoch_end(self, outputs):
    #     # loss = torch.stack([d[self.VALID_LOSS_NAME] for d in outputs]).mean()
    #     # word_acc = np.mean([d[self.VALID_WORD_ACC_NAME] for d in outputs])
    #     # logs = {self.VALID_LOSS_NAME: loss, self.VALID_WORD_ACC_NAME: word_acc}
    #     # return {'loss': loss, 'log': logs, 'progress_bar': logs}
    #     return {}

    def test_step(self, batch, batch_idx):
        xs, ys, y_hats, losses, loss = self._inner_forward_step(batch)
        return self._make_log_dict(ys, y_hats, losses, loss, self.LOSS_NAMES, self.metrics, self.TEST_PHASE_NAME)

    # def test_epoch_end(self, outputs):
    #     # loss = torch.stack([d[self.TEST_LOSS_NAME] for d in outputs]).mean()
    #     # word_acc = np.mean([d[self.TEST_WORD_ACC_NAME] for d in outputs])
    #     # logs = {self.TEST_LOSS_NAME: loss, self.TEST_WORD_ACC_NAME: word_acc}
    #     # return {'loss': loss, 'log': logs, 'progress_bar': logs}
    #     return {}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hp.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def prepare_data(self):
        # called on one gpu
        self.hp.num_data_total = len(self.ds)
        self.hp.num_data_test = int(self.hp.num_data_total * self.hp.p_test)
        self.hp.num_data_valid = int(self.hp.num_data_total * self.hp.p_valid)
        self.hp.num_data_train = self.hp.num_data_total - self.hp.num_data_test - self.hp.num_data_valid

        num_data_per_phase = [self.hp.num_data_train, self.hp.num_data_valid, self.hp.num_data_test]

        self.train_dataset, self.valid_dataset, self.test_dataset = \
            torch.utils.data.random_split(self.ds, num_data_per_phase)

        print(f'module prepare_date ds lens: '
              f'{len(self.train_dataset)}, {len(self.valid_dataset)}, {len(self.test_dataset)}')

    def setup(self, stage):
        # called on every gpu
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hp.batch_size, num_workers=self.hp.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.hp.batch_size, num_workers=self.hp.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hp.batch_size, num_workers=self.hp.num_workers)


if __name__ == '__main__':
    # from torchtext.data import Field, BucketIterator
    # from torchnlp.encoders.text import WhitespaceEncoder
    # from torchnlp.word_to_vector import GloVe

    hp = MyHyperparams()
    trainer = pl.Trainer(
        max_epochs=hp.num_epochs,
        weights_summary='full',
        fast_dev_run=False,
    )
    net = TrapezoidConv1Module(hp, constants.DOCS_DIR)

    print("HP:")
    utils.print_dict(hp.to_dict())
    fit_out = trainer.fit(net)
    print('fit_out:', fit_out)
