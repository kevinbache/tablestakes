import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from tablestakes.ml import data
from tablestakes.ml.hyperparams import MyHyperparams
from torch import optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl


class TrapezoidConv1Module(pl.LightningModule):
    def __init__(
            self,
            hp: MyHyperparams,
    ):
        super().__init__()

        self.hp = hp
        self.ds = data.XYCsvDataset(self.hp.data_dir, ys_postproc=lambda ys: [y[0] for y in ys])

        self.hp.num_x_dims = self.ds.num_x_dims
        self.hp.num_y_dims = self.ds.num_y_dims
        self.hp.num_vocab = self.ds.num_vocab

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

        ## upsample tensor 'src' to have the same spatial size with tensor 'tar'
        def _upsample_like(src, tar):

            src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

            return src

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

        # output
        fc_layers.append(nn.Conv1d(prev_num_neurons, self.hp.num_y_dims[0], 1))

        self.fc_layers = nn.ModuleList(fc_layers)

    def forward(self, x):
        x_base, vocab_ids = x

        # emb = torch.squeeze(self.embedder(vocab_ids), dim=2)
        emb = self.embedder(vocab_ids)
        # print('forward.x_base type:', type(x_base))
        # print('forward.emb    type:', type(emb))
        # print('forward.x_base.size:', x_base.size())
        # print('forward.emb.   size:', emb.size())
        emb = emb.squeeze(2)
        # print('forward.emb.   size:', emb.size())

        x = torch.cat([x_base.float(), emb], dim=-1).permute([0, 2, 1])
        # print('now conv:')
        for layer in self.conv_layers:
            # print(x.shape, layer)
            x = layer(x)

        # print('now fc:')
        for layer in self.fc_layers:
            # print(x.shape, layer)
            x = layer(x)

        # print('final')
        # print(x.shape)
        x = x.permute([0, 2, 1])
        return x

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch
        y = y.argmax(dim=1)
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
        y = y[0]
        # print('validation_step y.shape', y.shape)
        y = y.argmax(dim=1)
        # print('validation_step y.shape', y.shape)
        y_hat = self(x)
        # print('valid_step shapes:')
        # print('y    ', y.shape)
        # print('y_hat', y_hat.shape)
        val_loss = F.cross_entropy(y_hat.squeeze(0), y)
        labels_hat = torch.argmax(y_hat, dim=-1)
        # print('y     ', y.shape)
        # print('yhat  ', y_hat.shape)
        # print('labhat', labels_hat.shape)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'val_loss': val_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y[0].argmax(dim=1)
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=-1)
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
    )
    net = TrapezoidConv1Module(hp)

    print("HP:")
    utils.print_dict(hp.to_dict())
    fit_out = trainer.fit(net)
    print('fit_out:', fit_out)
