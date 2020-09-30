import math
from pathlib import Path
from typing import Any, Optional, List, Dict, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.metrics import TensorMetric, Metric
from pytorch_lightning import loggers as pl_loggers

from tablestakes import constants, utils
from tablestakes.ml import data
from tablestakes.ml.hyperparams import MyHyperparams


class WordAccuracy(TensorMetric):
    def __init__(
            self,
            reduce_group: Any = None,
            # reduce_op: Any = None,
    ):
        super().__init__('acc', reduce_group)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        is_correct = target.view(-1) == torch.argmax(pred, dim=-1).view(-1)
        return is_correct.float().mean()


class RectTransformerModule(pl.LightningModule):
    LOSS_VAL_NAME = 'loss'
    WORD_ACC_VAL_NAME = 'acc'

    # todo: really i need to split the datapoints first and then calculate the vocab
    # todo: remove rare words
    # todo: use some kind of pretrained embeddings
    """ 
    If using pretrained embeddings then don't need to think about vocab and data splitting
    Just need something in the tokenizer to handle out of vocab words
    """
    # @staticmethod
    # def _recalc_vocab_size(ds: data.XYCsvDataset):
    #     return torch.stack([xs[1].max() for xs in ds._xs]).max().item()

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
        self.ds = data.XYCsvDataset(self.data_dir)

        self.num_vocab = len(self.word_to_id)

        self.hp.num_vocab = self.num_vocab

        num_example_words = 200
        num_x_basic_dims = self.ds.num_x_dims[constants.X_BASIC_NAME]
        num_x_vocab_dims = self.ds.num_x_dims[constants.X_VOCAB_NAME]
        self.example_input_array = {
            constants.X_BASIC_NAME: torch.tensor(np.random.rand(1, num_example_words, num_x_basic_dims)).float(),
            constants.X_VOCAB_NAME: torch.tensor(np.random.rand(1, num_example_words, num_x_vocab_dims)).long(),
        }

        self.hp.num_x_dims = self.ds.num_x_dims
        self.hp.num_y_dims = self.ds.num_y_dims

        # save all variables in __init__ signature to self.hparams
        self.hparams = self.hp.to_dict()
        self.save_hyperparameters()

        self.train_dataset, self.valid_dataset, self.test_dataset = None, None, None

        #############
        # expand embedding_dim so that embedding_dim + meta_dim is divisible by hp.num_trans_heads

        num_x_basic_dims = self.hp.num_x_dims[constants.X_BASIC_NAME]

        num_basic_plus_embed_dims = num_x_basic_dims
        if hp.do_include_embeddings:
            num_basic_plus_embed_dims += hp.num_embedding_dim

        remainder = num_basic_plus_embed_dims % hp.num_trans_heads
        if remainder:
            self.hp.num_extra_embedding_dim = hp.num_trans_heads - remainder
            self.hp.num_total_embedding_dim = self.hp.num_embedding_dim + self.hp.num_extra_embedding_dim
            num_basic_plus_embed_dims += self.hp.num_extra_embedding_dim

        self.hp.num_basic_plus_embed_dims = num_basic_plus_embed_dims

        #############
        # emb
        if hp.do_include_embeddings:
            self.embedder = nn.Embedding(self.num_vocab, self.hp.num_total_embedding_dim)

        #############
        # trans
        if self.hp.pre_trans_linear_dim is not None:
            self.pre_enc_linear = nn.Conv1d(self.hp.num_basic_plus_embed_dims, self.hp.pre_trans_linear_dim, 1)
            num_trans_input_dims = self.hp.pre_trans_linear_dim
        else:
            num_trans_input_dims = self.hp.num_basic_plus_embed_dims

        enc_layer = nn.TransformerEncoderLayer(
            d_model=num_trans_input_dims,
            nhead=self.hp.num_trans_heads,
            dim_feedforward=self.hp.num_trans_fc_units,
            activation='gelu'
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=hp.num_trans_enc_layers,
        )

        #############
        # fc
        fc_layers = []

        prev_num_neurons = num_trans_input_dims
        if self.hp.do_cat_x_base_before_fc:
            prev_num_neurons += num_x_basic_dims

        num_fc_neurons = np.logspace(
            start=hp.log2num_neurons_start,
            stop=hp.log2num_neurons_end,
            num=hp.num_fc_hidden_layers,
            base=2,
        ).astype(np.int32)
        for fc_ind, num_neurons in enumerate(num_fc_neurons):
            fc_layers.append(nn.Conv1d(prev_num_neurons, num_neurons, 1))
            fc_layers.append(nn.LeakyReLU())
            prev_num_neurons = num_neurons

            if not fc_ind % hp.num_fc_layers_per_dropout:
                fc_layers.append(nn.Dropout(p=hp.dropout_p))

        self.fc_layers = nn.ModuleList(fc_layers)

        # TODO: requires grad = false? buffer?
        self.loss_weights = torch.tensor(self.hp.loss_weights, dtype=torch.float)

        # output
        self.heads = nn.ModuleList([
            nn.Conv1d(prev_num_neurons, num_classes, 1)
            for name, num_classes in self.num_y_classes.items()
        ])

    def forward(self, x_basic: torch.Tensor, x_vocab: torch.Tensor):
        x_base = x_basic.float()
        x_vocab = x_vocab.long()

        if hp.do_include_embeddings:
            emb = self.embedder(x_vocab)
            # squeeze out the "time" dimension we expect for language models
            emb = emb.squeeze(2)

            x = torch.cat([x_base, emb], dim=-1)
        else:
            x = x_base

        if self.hp.pre_trans_linear_dim is not None:
            x = x.permute(0, 2, 1)
            x = self.pre_enc_linear(x)
            x = x.permute(0, 2, 1)

        x = self.encoder(x)

        if self.hp.do_cat_x_base_before_fc:
            x = torch.cat([x, x_base], dim=-1)

        x = x.permute(0, 2, 1)
        for layer in self.fc_layers:
            x = layer(x)

        y_hats = {name: layer(x).permute(0, 2, 1) for name, layer in zip(self.num_y_classes.keys(), self.heads)}
        return y_hats

    def _inner_forward_step(self, batch):
        xs_dict, ys_dict = batch
        y_hats_dict = self(**xs_dict)

        losses = torch.stack([
            F.cross_entropy(y_hat.squeeze(0), y.view(-1))
            for y, y_hat in zip(ys_dict.values(), y_hats_dict.values())
        ])
        losses = torch.mul(losses, self.loss_weights)
        loss = losses.sum()

        return xs_dict, ys_dict, y_hats_dict, losses, loss

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
            ys: Dict[str, torch.Tensor],
            y_hats: Dict[str, torch.Tensor],
            metrics: List[Metric],
            phase_name: str,
    ):
        out = {}
        loss_names = list(ys.keys())
        for y, y_hat, loss_name in zip(ys.values(), y_hats.values(), loss_names):
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
            cls._make_phase_name(phase_name, cls.LOSS_VAL_NAME, loss_name): loss
            for loss, loss_name in zip(losses, loss_names)
        }

    @classmethod
    def _make_log_dict(
            cls,
            ys_dict: Dict[str, torch.Tensor],
            y_hats_dict: Dict[str, torch.Tensor],
            losses: torch.Tensor,
            loss: torch.Tensor,
            metrics: List[Metric],
            phase_name: str,
    ):
        out = cls._make_metrics_dict(ys_dict, y_hats_dict, metrics, phase_name)
        out.update(cls._make_losses_dict(losses, list(y_hats_dict.keys()), phase_name))
        out[cls._make_phase_name(phase_name, cls.LOSS_VAL_NAME)] = loss
        return out

    @staticmethod
    def _average_output_logs(outputs, do_include_progress_bar=True):
        if not outputs:
            return {}

        logs = outputs
        keys = logs[0].keys()
        output_means = {f'avg_{k}': torch.stack([log[k] for log in logs]).mean() for k in keys}
        d = {'log': output_means}
        if do_include_progress_bar:
            d['progress_bar'] = output_means
        return d

    def training_step(self, batch, batch_idx):
        xs_dict, ys_dict, y_hats_dict, losses, loss = self._inner_forward_step(batch)
        logs = self._make_log_dict(ys_dict, y_hats_dict, losses, loss, self.metrics, self.TRAIN_PHASE_NAME)
        return {'loss': loss, **logs}

    def training_epoch_end(self, outputs):
        return self._average_output_logs(outputs)

    def validation_step(self, batch, batch_idx):
        xs_dict, ys_dict, y_hats_dict, losses, loss = self._inner_forward_step(batch)
        logs = self._make_log_dict(ys_dict, y_hats_dict, losses, loss, self.metrics, self.VALID_PHASE_NAME)
        return logs

    def validation_epoch_end(self, outputs):
        return self._average_output_logs(outputs)

    def test_step(self, batch, batch_idx):
        xs_dict, ys_dict, y_hats_dict, losses, loss = self._inner_forward_step(batch)
        logs = self._make_log_dict(ys_dict, y_hats_dict, losses, loss, self.metrics, self.VALID_PHASE_NAME)
        return logs

    def test_step_end(self, outputs):
        return self._average_output_logs(outputs, do_include_progress_bar=False)

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hp.lr)
        return [optimizer]
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        # return [optimizer], [scheduler]

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
        return DataLoader(self.train_dataset, batch_size=1, num_workers=self.hp.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=1, num_workers=self.hp.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.hp.num_workers)


if __name__ == '__main__':
    # from torchtext.data import Field, BucketIterator
    # from torchnlp.encoders.text import WhitespaceEncoder
    # from torchnlp.word_to_vector import GloVe

    dataset_name = 'num=100_extra=0'

    hp = MyHyperparams()
    trainer = pl.Trainer(
        logger=pl_loggers.TensorBoardLogger('tensorboard_logs/', name="trans1d_trial"),
        max_epochs=hp.num_epochs,
        weights_summary='full',
        fast_dev_run=False,
        accumulate_grad_batches=int(math.pow(2, hp.batch_size_log2)),
    )
    net = RectTransformerModule(hp, constants.DOCS_DIR / dataset_name)

    print("HP:")
    utils.print_dict(hp.to_dict())
    fit_out = trainer.fit(net)

    print('fit_out:', fit_out)
