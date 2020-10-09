from typing import Optional, List, Dict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from tablestakes import constants, utils
from tablestakes.ml import data, torch_helpers, hyperparams


class RectTransformerModule(pl.LightningModule):
    LOSS_VAL_NAME = 'loss'
    TOTAL_NAME = 'total'

    METRICS = {
        'acc': pl.metrics.Accuracy(),
    }

    def __init__(self, hp: hyperparams.LearningParams):
        super().__init__()

        self.hp = hp

        self.num_y_classes = utils.load_json(self.hp.data_dir / constants.NUM_Y_CLASSES_FILENAME)
        self.word_to_id = utils.load_json(self.hp.data_dir / constants.WORD_ID_FILENAME)
        self.word_to_count = utils.load_json(self.hp.data_dir / constants.WORD_COUNT_FILENAME)

        self.ds = data.XYCsvDataset(self.hp.data_dir)

        self.num_vocab = len(self.word_to_id)

        self.hp.num_vocab = self.num_vocab

        num_example_words = 200
        num_x_basic_dims = self.ds.num_x_dims[constants.X_BASIC_BASE_NAME]
        num_x_vocab_dims = self.ds.num_x_dims[constants.X_VOCAB_BASE_NAME]

        self.example_input_array = {
            constants.X_BASIC_BASE_NAME: torch.tensor(np.random.rand(1, num_example_words, num_x_basic_dims)).float(),
            constants.X_VOCAB_BASE_NAME: torch.tensor(np.random.rand(1, num_example_words, num_x_vocab_dims)).long(),
        }

        self.hp.num_x_dims = self.ds.num_x_dims
        self.hp.num_y_dims = self.ds.num_y_dims

        self.train_dataset, self.valid_dataset, self.test_dataset = None, None, None

        #############
        # expand embedding_dim so that embedding_dim + meta_dim is divisible by hp.num_trans_heads
        # (attention requirement)
        num_x_basic_dims = self.hp.num_x_dims[constants.X_BASIC_BASE_NAME]

        num_basic_plus_embed_dims = num_x_basic_dims
        if hp.do_include_embeddings:
            num_basic_plus_embed_dims += hp.num_embedding_dim

        remainder = num_basic_plus_embed_dims % hp.num_trans_heads
        if remainder and self.hp.pre_trans_linear_dim:
            self.hp.num_extra_embedding_dim = hp.num_trans_heads - remainder
        else:
            self.hp.num_extra_embedding_dim = 0
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

        self.encoder = torch_helpers.get_pytorch_transformer_encoder(self.hp, num_trans_input_dims)
        # self.encoder = get_simple_ablatable_transformer_encoder(
        #     self.hp,
        #     num_trans_input_dims,
        #     do_drop_k=False,
        # )

        ######################################################################
        ### CAT
        ######################################################################
        if self.hp.do_cat_x_base_before_fc:
            self.hp.num_fc_inputs = num_trans_input_dims + num_x_basic_dims
        else:
            self.hp.num_fc_inputs = num_trans_input_dims

        num_fc_neurons = list(np.logspace(
            start=hp.log2num_neurons_start,
            stop=hp.log2num_neurons_end,
            num=hp.num_fc_blocks,
            base=2,
        ).astype(np.int32))
        self.fc_module = torch_helpers.FullyConv1Resnet(
            num_input_features=self.hp.num_fc_inputs,
            neuron_counts=num_fc_neurons,
            num_blocks_per_residual=self.hp.num_fc_blocks_per_resid,
        )

        # TODO: requires grad = false? buffer?
        self.loss_weights = torch.tensor([self.hp.korv_loss_weight, 1.0], dtype=torch.float, device=self.device)

        # output
        self.heads = nn.ModuleList([
            nn.Conv1d(self.fc_module.get_num_outputs(), num_classes, 1)
            for name, num_classes in self.num_y_classes.items()
        ])

        # save all variables in __init__ signature to self.hparams
        self.save_hyperparameters(self.hp.to_dict())

    def forward(self, basic: torch.Tensor, vocab: torch.Tensor):
        basic = basic.float()
        vocab = vocab.long()

        if self.hp.do_include_embeddings:
            x = self.embedder(vocab)
            # squeeze out the "time" dimension we expect for language models
            x = x.squeeze(2)
            x = torch.cat([basic, x], dim=-1)
        else:
            x = basic

        if self.hp.pre_trans_linear_dim is not None:
            x = x.permute(0, 2, 1)
            x = self.pre_enc_linear(x)
            x = x.permute(0, 2, 1)

        ##########
        x = self.encoder(x)

        if self.hp.do_cat_x_base_before_fc:
            x = torch.cat([x, basic], dim=-1)

        x = x.permute(0, 2, 1)
        x = self.fc_module(x)

        y_hats = {
            name: layer(x).permute(0, 2, 1)
            for name, layer in zip(self.num_y_classes.keys(), self.heads)
        }
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

    PHASE_NAMES = [
        TRAIN_PHASE_NAME,
        VALID_PHASE_NAME,
        TEST_PHASE_NAME,
    ]

    @staticmethod
    def _get_phase_name(phase_name: str, metric_name: str, output_name: Optional[str] = None):
        out = f'{phase_name}'
        out += f'_{metric_name}'
        if output_name is not None:
            out += f'_{output_name}'
        return out

    @classmethod
    def get_valid_metric_name(cls, metric_name: str, output_name: Optional[str] = None):
        return cls._get_phase_name(cls.VALID_PHASE_NAME, metric_name, output_name)

    @classmethod
    def get_train_metric_name(cls, metric_name: str, output_name: Optional[str] = None):
        return cls._get_phase_name(cls.TRAIN_PHASE_NAME, metric_name, output_name)

    @classmethod
    def get_all_metric_names_for_phase(cls, phase_name: str):
        metrics_names = [m for m in cls.METRICS.keys()] + [cls.LOSS_VAL_NAME]
        output_names = constants.Y_BASE_NAMES

        out = []
        for metric_name in metrics_names:
            for output_name in output_names:
                out.append(cls._get_phase_name(phase_name, metric_name, output_name))
        out.append(cls._get_phase_name(phase_name, cls.LOSS_VAL_NAME, cls.TOTAL_NAME))

        return out

    def _log_losses_and_metrics(self, phase_name, loss, losses, y_hats_dict, ys_dict, prog_bar=False):
        output_names = ys_dict.keys()

        on_epoch = None

        self.log(self._get_phase_name(phase_name, 'loss', 'total'), loss, prog_bar=prog_bar, on_epoch=on_epoch)
        for output_name, current_loss in zip(output_names, losses):
            full_metric_name = self._get_phase_name(phase_name, 'loss', output_name)
            self.log(full_metric_name, current_loss, prog_bar=False, on_epoch=on_epoch)

        for metric_name, metric in self.METRICS.items():
            for output_name in output_names:
                y_hat = y_hats_dict[output_name].squeeze()
                y = ys_dict[output_name].squeeze()
                full_metric_name = self._get_phase_name(phase_name, metric_name, output_name)
                metric_value = metric(y_hat, y)
                self.log(full_metric_name, metric_value, prog_bar=False, on_epoch=on_epoch)

    def training_step(self, batch, batch_idx):
        xs_dict, ys_dict, y_hats_dict, losses, loss = self._inner_forward_step(batch)
        self._log_losses_and_metrics(self.TRAIN_PHASE_NAME, loss, losses, y_hats_dict, ys_dict, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xs_dict, ys_dict, y_hats_dict, losses, loss = self._inner_forward_step(batch)
        self._log_losses_and_metrics(self.VALID_PHASE_NAME, loss, losses, y_hats_dict, ys_dict)

    def test_step(self, batch, batch_idx):
        xs_dict, ys_dict, y_hats_dict, losses, loss = self._inner_forward_step(batch)
        self._log_losses_and_metrics(self.TEST_PHASE_NAME, loss, losses, y_hats_dict, ys_dict)

    def on_after_backward(self):
        if self.hp.num_steps_per_histogram_log and not self.global_step % self.hp.num_steps_per_histogram_log:
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(f'weights/{name}', param, self.current_epoch)
                self.logger.experiment.add_histogram(f'grads/{name}', param.grad, self.current_epoch)

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hp.lr)
        # return [optimizer]
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
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
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=self.hp.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=1, shuffle=False, num_workers=self.hp.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.hp.num_workers)


if __name__ == '__main__':
    # dataset_name = 'num=100_60d9'
    dataset_name = 'num=1000_4475'

    hp = hyperparams.LearningParams()
    hp.data_dir = constants.DOCS_DIR / dataset_name

    trainer = pl.Trainer(
        logger=pl_loggers.TensorBoardLogger(constants.LOGS_DIR, name="trans1d_trial2"),
        max_epochs=hp.num_epochs,
        weights_summary='full',
        fast_dev_run=False,
        accumulate_grad_batches=utils.pow2int(hp.log2_batch_size),
        profiler=True,
    )
    net = RectTransformerModule(hp)

    print("HP:")
    utils.print_dict(hp.to_dict())
    fit_out = trainer.fit(net)

    print('fit_out:', fit_out)
