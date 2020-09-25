import math
from typing import List, Any

import utils

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from tablestakes.ml import data
from tablestakes.ml.hyperparams import MyHyperparams
from torch import optim
from torch.utils.data import DataLoader

from pytorch_lightning.metrics import TensorMetric


class WordAccuracy(TensorMetric):
    def __init__(
            self,
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        super().__init__('word_acc', reduce_group, reduce_op)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        is_correct = target == torch.argmax(pred, dim=-1)
        return is_correct.float().mean()


class TrapezoidConv1Module(pl.LightningModule):
    Y_NAMES = ['korv', 'which_kv']
    LOSS_VAL_NAME = 'loss'
    WORD_ACC_VAL_NAME = 'word_acc'

    def __init__(
            self,
            hp: MyHyperparams,
    ):
        super().__init__()

        self.metrics = [WordAccuracy()]

        self.hp = hp
        # self.ds = data.XYCsvDataset(self.hp.data_dir, ys_postproc=lambda ys: [y[0] for y in ys])
        self.ds = data.XYCsvDataset(self.hp.data_dir)

        self.hp.num_x_dims = self.ds.num_x_dims
        self.hp.num_y_dims = self.ds.num_y_dims
        self.hp.num_vocab = self.ds.num_vocab

        # or implement only in gradient averaging
        assert self.hp.batch_size_log2 == 0
        self.hp.batch_size = int(math.pow(2, self.hp.batch_size_log2))

        # save all variables in __init__ signature to self.hparams
        self.hparams = self.hp.to_dict()
        self.save_hyperparameters()

        self.train_dataset, self.valid_dataset, self.test_dataset = None, None, None

        #############
        # emb
        self.embedder = nn.Embedding(self.hp.num_vocab, self.hp.num_embedding_dim)

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
        prev_num_filters = hp.num_x_dims[0] + hp.num_embedding_dim
        for conv_ind, num_filters in enumerate(num_conv_filters):
            conv_layers.append(nn.Conv1d(prev_num_filters, num_filters, hp.kernel_size, padding=int(hp.kernel_size/2)))
            conv_layers.append(nn.ReLU(inplace=True))
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
            fc_layers.append(nn.ReLU(inplace=True))
            prev_num_neurons = num_neurons

            if not fc_ind % hp.num_fc_layers_per_dropout:
                fc_layers.append(nn.Dropout(p=hp.dropout_p))

        self.fc_layers = nn.ModuleList(fc_layers)

        # output
        self.heads = nn.ModuleList([
            nn.Conv1d(prev_num_neurons, self.hp.num_y_dims[0], 1),
            nn.Conv1d(prev_num_neurons, self.hp.num_y_dims[1], 1),
        ])

    def forward(self, x):
        x_base, vocab_ids = x
        emb = self.embedder(vocab_ids)
        # squeeze out the "time" dimension we expect for language models
        emb = emb.squeeze(2)

        # batch x word x feature --> batch x feature x word
        x = torch.cat([x_base.float(), emb], dim=-1).permute([0, 2, 1])
        for layer in self.conv_layers:
            x = layer(x)

        for layer in self.fc_layers:
            x = layer(x)

        y_hats = [layer(x).permute([0, 2, 1]) for layer in self.heads]

        return y_hats

    def _inner_forward_step(self, batch):
        xs, ys = batch
        for y in ys:
            print(y.shape)
        ys = [y.argmax(dim=-1) for y in ys]
        for y in ys:
            print(y.shape)
        print('y')
        print(ys)
        y_hats = self(xs)

        losses = torch.tensor([
            F.cross_entropy(y_hat.squeeze(0), y.squeeze(0))
            for y, y_hat in zip(ys, y_hats)
        ])

        loss = losses.dot(torch.tensor(hp.loss_weights, dtype=torch.float))

        return xs, ys, y_hats, losses, loss

    TRAIN_PHASE_NAME = 'train'
    VALID_PHASE_NAME = 'valid'
    TEST_PHASE_NAME = 'test'

    @staticmethod
    def _make_log_name(phase_name: str, metric_name: str):
        return f'{phase_name}_{metric_name}'

    @staticmethod
    def _make_metrics_dict(y, y_hat, do_include_logs=True):
        out = {}
        if do_include_logs:
            logs = {}

    # def _make_losses_dict(self):

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        xs, ys, y_hats, losses, loss = self._inner_forward_step(batch)

        word_acc = self._calculate_word_acc(ys, y_hats)

        logs = {self.TRAIN_LOSS_NAME: losses, self.TRAIN_WORD_ACC_NAME: word_acc}
        return {'loss': losses, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        phase = 'valid'

        xs, ys, y_hats, loss = self._inner_forward_step(batch)
        word_acc = self._calculate_word_acc(ys, y_hats)
        return {self.VALID_LOSS_NAME: loss, self.VALID_WORD_ACC_NAME: word_acc}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        loss = torch.stack([d[self.VALID_LOSS_NAME] for d in outputs]).mean()
        word_acc = np.mean([d[self.VALID_WORD_ACC_NAME] for d in outputs])
        logs = {self.VALID_LOSS_NAME: loss, self.VALID_WORD_ACC_NAME: word_acc}
        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_idx):
        x, y, y_hat, loss = self._inner_forward_step(batch)
        word_acc = self._calculate_word_acc(y, y_hat)
        return {self.TEST_LOSS_NAME: loss, self.TEST_WORD_ACC_NAME: word_acc}

    def test_epoch_end(self, outputs):
        loss = torch.stack([d[self.TEST_LOSS_NAME] for d in outputs]).mean()
        word_acc = np.mean([d[self.TEST_WORD_ACC_NAME] for d in outputs])
        logs = {self.TEST_LOSS_NAME: loss, self.TEST_WORD_ACC_NAME: word_acc}
        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
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

    from pytorch_lightning.core.memory import ModelSummary

    hp = MyHyperparams()
    trainer = pl.Trainer(
        max_epochs=hp.num_epochs,
        weights_summary=ModelSummary.MODE_FULL,
        fast_dev_run=False,
    )
    net = TrapezoidConv1Module(hp)

    print("HP:")
    utils.print_dict(hp.to_dict())
    fit_out = trainer.fit(net)
    print('fit_out:', fit_out)
