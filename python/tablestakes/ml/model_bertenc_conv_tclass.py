import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl

from chillpill import params

from tablestakes import constants
from tablestakes.ml import torch_helpers, param_torch_mods, factored, data


class ModelBertEncConvTClass(factored.FactoredLightningModule):
    class Params(params.ParameterSet):
        data = data.TablestakesDataModule.DataParams()
        opt = factored.OptimizersMaker.OptParams()
        metrics = factored.MetricTracker.Params()
        exp = param_torch_mods.ExperimentParams()

        embed = param_torch_mods.BertEmbedder.Params()
        conv = param_torch_mods.ConvBlock.Params()
        fc = param_torch_mods.SlabNet.Params()
        heads = param_torch_mods.HeadedSlabNet.Params()

        def __init__(self, max_seq_len=1024, batch_size=32):
            super().__init__()
            self.embed.max_seq_len = self.data.max_seq_length = max_seq_len
            self.data.batch_size = self.opt.batch_size = batch_size

    def __init__(
            self,
            hp: Params,
            data_module: data.XYDocumentDataModule,
            metrics_tracker: factored.MetricTracker,
            opt: factored.OptimizersMaker,
    ):
        super().__init__(hp, metrics_tracker, opt)
        self.dm = data_module

        self.num_y_classes = self.dm.num_y_classes
        self.example_input_array = self.dm.example_input_array

        ###############################################################
        # MODEL
        self.embed = param_torch_mods.BertEmbedder(self.hp.embed)

        self.conv = param_torch_mods.ConvBlock(
            num_input_features=self.hp.embed.dim + self.dm.num_x_base_dims,
            hp=self.hp.conv,
        )

        self.fc = param_torch_mods.SlabNet(
            num_input_features=self.conv.get_num_output_features() + self.dm.num_x_base_dims,
            hp=self.hp.fc,
        )

        self.heads = nn.ModuleDict({
            name: param_torch_mods.HeadedSlabNet(
                num_input_features=self.fc.get_num_outputs(),
                num_output_features=num_classes,
                hp=self.hp.heads,
            )
            for name, num_classes in self.num_y_classes.items()
        })
        # END MODEL
        ###############################################################

        self.save_hyperparameters(self.hp.to_dict())

    def forward(self, base: torch.Tensor, vocab: torch.Tensor):
        x = self.embed(vocab)
        x = torch.cat([base, x.last_hidden_state], dim=-1)

        x = x.permute(0, 2, 1)
        x = self.conv(x)

        # sharp
        x = x.permute(0, 2, 1)
        x = torch.cat([base, x], dim=-1)
        x = x.permute(0, 2, 1)

        x = self.fc(x)

        y_hats = {
            head_name: head(x).permute(0, 2, 1)
            for head_name, head in self.heads.items()
        }
        return y_hats


def run(hp: ModelBertEncConvTClass.Params, fast_dev_run=False):
    dm = data.TablestakesDataModule(hp.data)
    net = ModelBertEncConvTClass(
        hp=hp,
        data_module=dm,
        metrics_tracker=factored.MetricTracker(hp.metrics),
        opt=factored.OptimizersMaker(hp.opt),
    )

    trainer = pl.Trainer(
        logger=True if fast_dev_run else param_torch_mods.get_pl_logger(hp.exp),
        default_root_dir=hp.metrics.output_dir,
        callbacks=[torch_helpers.CounterTimerCallback()],
        max_epochs=hp.opt.num_epochs,
        weights_summary='full',
        fast_dev_run=fast_dev_run,
        accumulate_grad_batches=1,
        profiler=True,
        deterministic=True,
        auto_lr_find=False,
        log_every_n_steps=hp.metrics.num_steps_per_metric_log,
    )

    print("Starting trainer.fit:")
    print(f'dataset file: {hp.data.get_dataset_file()}')
    trainer.fit(net, datamodule=dm)


if __name__ == '__main__':
    fast_dev_run = False

    hp = ModelBertEncConvTClass.Params(
        max_seq_len=int(np.power(2, 11)),
        batch_size=32,
    )

    hp.data.dataset_name = 'num=1000_02b7'
    hp.data.do_ignore_cached_dataset = False
    hp.data.seed = 42
    hp.data.num_workers = 4
    hp.data.num_gpus = 1
    hp.data.num_cpus = 4

    hp.opt.search_metric = 'valid_loss_total'
    hp.opt.search_mode = 'min'
    hp.opt.num_epochs = 1000
    hp.opt.lr = 0.001
    hp.opt.patience = 10

    hp.metrics.num_steps_per_histogram_log = 5
    hp.metrics.num_steps_per_metric_log = 5
    hp.metrics.output_dir = constants.OUTPUT_DIR

    hp.exp.project_name = 'tablestakes'
    hp.exp.experiment_name = 'korv which'
    hp.exp.experiment_tags = ['korv_which', 'conv', 'sharp', 'testing']
    hp.exp.sources_glob_str = constants.THIS_DIR.parent.parent / '**/*.py'

    hp.embed.dim = 16
    hp.embed.requires_grad = True

    hp.conv.num_features = 128
    hp.conv.num_layers = 8
    hp.conv.kernel_size = 3
    hp.conv.num_groups = 16
    hp.conv.num_blocks_per_pool = 20
    hp.conv.requires_grad = True

    hp.fc.num_neurons = 128
    hp.fc.num_layers = 2
    hp.fc.num_groups = 16
    hp.fc.num_blocks_per_residual = 2
    hp.fc.num_blocks_per_dropout = 2
    hp.fc.requires_grad = True

    hp.heads.num_neurons = 128
    hp.heads.num_layers = 4
    hp.heads.num_groups = 16
    hp.heads.num_blocks_per_residual = 2
    hp.heads.num_blocks_per_dropout = 2
    hp.heads.requires_grad = True

    run(hp, fast_dev_run)