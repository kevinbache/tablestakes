import numpy as np
import tablestakes.ml.metrics_mod
import tablestakes.ml.torch_mod

import torch
import torch.nn as nn
import pytorch_lightning as pl

from tablestakes import constants, utils
from tablestakes.ml import metrics_mod, torch_mod, factored, data


class ModelBertEncConvTClass(factored.FactoredLightningModule):
    class Params(factored.FactoredLightningModule.FactoredParams):
        data = data.TablestakesDataModule.DataParams()
        opt = factored.OptimizersMaker.OptParams()
        metrics = tablestakes.ml.metrics_mod.MetricsTracker.MetricParams()
        exp = torch_mod.ExperimentParams()

        embed = torch_mod.BertEmbedder.ModelParams()
        conv = torch_mod.ConvBlock.ModelParams()
        trans = torch_mod.TransBlockBuilder.ModelParams()
        fc = torch_mod.SlabNet.ModelParams()
        heads = torch_mod.HeadedSlabNet.ModelParams()

        def __init__(self, max_seq_len=1024, batch_size=32):
            super().__init__()
            self.embed.max_seq_len = self.data.max_seq_length = max_seq_len
            self.data.batch_size = self.opt.batch_size = batch_size

    def __init__(
            self,
            hp: Params,
            data_module: data.XYDocumentDataModule,
            metrics_tracker: tablestakes.ml.metrics_mod.MetricsTracker,
            opt: factored.OptimizersMaker,
    ):
        super().__init__(hp, metrics_tracker, opt)
        self.dm = data_module
        num_x_base_features = self.dm.num_x_base_dims

        self.num_y_classes = self.dm.num_y_classes
        self.example_input_array = self.dm.example_input_array

        ###############################################################
        # MODEL
        self.embed = torch_mod.BertEmbedder(self.hp.embed)

        # cat here
        num_embedcat_features = self.hp.embed.dim + num_x_base_features

        if self.hp.conv.num_layers == 0:
            self.conv = None
            num_conv_features = 0
        else:
            self.conv = torch_mod.ConvBlock(
                num_input_features=num_embedcat_features,
                hp=self.hp.conv,
            )
            num_conv_features = self.conv.get_num_output_features()

        if self.hp.trans.num_layers == 0:
            self.trans = None
            num_trans_features = 0
        else:
            self.trans = torch_mod.TransBlockBuilder.build(hp=hp.trans, num_input_features=num_embedcat_features)
            num_trans_features = num_embedcat_features

        # cat here
        print(f'num_embedcat_features:  {num_embedcat_features}')
        print(f'num_trans_features:     {num_trans_features}')
        print(f'num_conv_features:      {num_conv_features}')
        num_fc_features = num_x_base_features + num_trans_features + num_conv_features
        self.fc = torch_mod.SlabNet(
            num_input_features=num_fc_features,
            hp=self.hp.fc,
        )

        self.heads = nn.ModuleDict({
            name: torch_mod.HeadedSlabNet(
                num_input_features=self.fc.get_num_outputs(),
                num_output_features=num_classes,
                hp=self.hp.heads,
            )
            for name, num_classes in self.num_y_classes.items()
        })
        # END MODEL
        ###############################################################

        self.save_hyperparameters(self.hp.to_dict())
        self.hparams.lr = self.hp.opt.lr

    def forward(self, base: torch.Tensor, vocab: torch.Tensor):
        x = self.embed(vocab)
        x = torch.cat([base, x.last_hidden_state], dim=-1)

        num_batch, num_seq, _ = base.shape

        x_trans = self.trans(x) if self.trans else torch.zeros(num_batch, num_seq, 0)
        x_conv = self.conv(x) if self.conv else torch.zeros(num_batch, num_seq, 0)

        # concatenate for sharpness
        x = torch.cat([base, x_trans, x_conv], dim=-1)

        x = self.fc(x)

        y_hats = {
            head_name: head(x)
            for head_name, head in self.heads.items()
        }
        return y_hats


def run(
        net: pl.LightningModule,
        dm: pl.LightningDataModule,
        hp: ModelBertEncConvTClass.Params,
        fast_dev_run=False,
        do_find_lr=False,
):
    trainer = pl.Trainer(
        logger=True if fast_dev_run else torch_mod.get_pl_logger(hp.exp),
        default_root_dir=hp.metrics.output_dir,
        callbacks=[metrics_mod.CounterTimerCallback()],
        max_epochs=hp.opt.num_epochs,
        gpus=hp.data.num_gpus,
        weights_summary='full',
        fast_dev_run=fast_dev_run,
        accumulate_grad_batches=1,
        profiler=True,
        deterministic=True,
        auto_lr_find=do_find_lr,
        log_every_n_steps=hp.metrics.num_steps_per_metric_log,
    )

    if do_find_lr:
        utils.hprint("Starting trainer.tune:")
        lr_tune_out = trainer.tune(net, datamodule=dm)
        print(f'  Tune out: {lr_tune_out}')
    else:
        utils.hprint("Starting trainer.fit:")
        print(f'  Dataset file: {hp.data.get_dataset_file()}')
        trainer.fit(net, datamodule=dm)

    utils.hprint('Done with model run fn')


if __name__ == '__main__':
    fast_dev_run = False

    hp = ModelBertEncConvTClass.Params(
        max_seq_len=int(np.power(2, 11)),
        batch_size=32,
    )

    hp.data.dataset_name = 'num=1000_02b7'
    # hp.data.dataset_name = 'num=4000_9b9f'
    hp.data.do_ignore_cached_dataset = False
    hp.data.seed = 42
    hp.data.num_workers = 4
    hp.data.num_gpus = 0
    hp.data.num_cpus = 4

    hp.opt.search_metric = 'valid_loss_total'
    hp.opt.search_mode = 'min'
    hp.opt.num_epochs = 10
    hp.opt.lr = 0.003
    hp.opt.patience = 10

    hp.metrics.num_steps_per_histogram_log = 5
    hp.metrics.num_steps_per_metric_log = 5
    hp.metrics.output_dir = constants.OUTPUT_DIR

    hp.exp.project_name = 'tablestakes'
    hp.exp.experiment_name = 'korv which'
    hp.exp.experiment_tags = ['korv_which', 'conv', 'sharp', 'testing']
    hp.exp.sources_glob_str = constants.THIS_DIR.parent.parent / '**/*.py'

    hp.embed.dim = 15
    hp.embed.requires_grad = True

    hp.conv.num_features = 128
    hp.conv.num_layers = 0
    hp.conv.kernel_size = 3
    hp.conv.num_groups = 16
    hp.conv.num_blocks_per_pool = 20
    hp.conv.requires_grad = True

    hp.trans.impl = 'fast-favor'
    hp.trans.num_heads = 8
    hp.trans.num_layers = 2
    hp.trans.num_query_features = None
    hp.trans.fc_dim_mult = 2

    hp.fc.num_features = 128
    hp.fc.num_layers = 2
    hp.fc.num_groups = 16
    hp.fc.num_blocks_per_residual = 2
    hp.fc.num_blocks_per_dropout = 2
    hp.fc.requires_grad = True

    hp.heads.num_features = 128
    hp.heads.num_layers = 4
    hp.heads.num_groups = 16
    hp.heads.num_blocks_per_residual = 2
    hp.heads.num_blocks_per_dropout = 2
    hp.heads.requires_grad = True

    dm = data.TablestakesDataModule(hp.data)
    net = ModelBertEncConvTClass(
        hp=hp,
        data_module=dm,
        metrics_tracker=tablestakes.ml.metrics_mod.MetricsTracker(hp.metrics),
        opt=factored.OptimizersMaker(hp.opt),
    )

    utils.hprint('About to start model run:')
    utils.print_dict(hp.to_dict())

    run(net, dm, hp, fast_dev_run, do_find_lr=False)
