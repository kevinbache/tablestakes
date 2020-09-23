import os
from argparse import ArgumentParser

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tablestakes.ml import data
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import pytorch_lightning as pl

from chillpill import params


class MyHyperparams(params.ParameterSet):
    ##############
    # model
    #  conv
    num_conv_layers = 4
    log2num_filters_start = 5
    log2num_filters_end = 4

    kernel_size = 3

    num_conv_layers_per_pool = 2
    pool_size = 2

    #  fc
    num_fc_hidden_layers = 2
    log2num_neurons_start = 6
    log2num_neurons_end = 5

    num_fc_layers_per_dropout = 2
    dropout_p = 0.5

    num_vocab = 10
    num_embedding_dim = 32

    ##############
    # optimization
    lr = 0.001
    momentum = 0.9
    limit_n_data = None

    num_epochs = 10

    ##############
    # data
    batch_size_log2 = 2
    p_valid = 0.1
    p_test = 0.1
    data_dir = '../scripts/generate_ocrd_doc_2/docs'


class TrapezoidConv1Model(pl.LightningModule):
    def __init__(
            self,
            hp: MyHyperparams,
    ):
        super().__init__()

        self.hp = hp

        ds = data.XYCsvDataset(self.hp.data_dir)

        # save all variables in __init__ signature to self.hparams
        self.save_hyperparameters()
        self.hparams = self.hp.to_dict()

        self.train_dataset, self.valid_dataset, self.test_dataset = None, None, None

        #############
        # conv
        self.embedder = nn.Embedding(self.hp.num_vocab, self.hp.num_embedding_dim)

        prev_num_filters = hp.num_input_features + hp.num_embedding_dim

        #############
        # conv
        conv_layers = []

        #  alright, so it's really more of a funnel than a trapezoid
        num_conv_filters = np.logspace(
            start=hp.log2num_filters_start,
            stop=hp.log2num_filters_end,
            num=hp.num_conv_layers,
            base=2,
        ).astype(np.integer)

        for conv_ind, num_filters in enumerate(num_conv_filters):
            conv_layers.append(nn.Conv1d(prev_num_filters, num_filters, hp.kernel_size))
            conv_layers.append(nn.ReLU(inplace=True))
            prev_num_filters = num_filters

            if not conv_ind % hp.num_conv_layers_per_pool:
                conv_layers.append(nn.MaxPool2d(hp.pool_size))

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

        fc_layers.append(nn.Linear(prev_num_neurons, self.INPUT_NUM_CLASSES))
        self.fc_layers = nn.ModuleList(fc_layers)

    def forward(self, x):
        x_base, vocab_ids = x

        emb = self.embedder(vocab_ids)
        print('forward.x_base.size:', x_base.size())
        print('forward.emb.size:', emb.size())

        x = torch.cat([x_base, emb], dim=1)
        for layer in self.conv_layers + self.fc_layers:
            x = layer(x)

        print('forward.x.size at end:', x.size())
        return x

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'val_loss': val_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'test_loss': test_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': test_acc}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

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
        self.hp.num_data_total = len(ds)
        self.hp.num_data_test = int(self.hp.num_data_total * self.hp.p_test)
        self.hp.num_data_valid = int(self.hp.num_data_total * self.hp.p_valid)
        self.hp.num_data_train = self.hp.num_data_total - self.hp.num_data_test - self.hp.num_data_valid

        dataset_counts = [self.hp.num_data_train, self.hp.num_data_valid, self.hp.num_data_test]
        self.train_dataset, self.valid_dataset, self.test_dataset = torch.utils.data.random_split(ds, dataset_counts)

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
    )
    net = TrapezoidConv1Model(hp)
    fit_out = trainer.fit(net)
    print(fit_out)
