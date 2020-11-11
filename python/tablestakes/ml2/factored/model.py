from dataclasses import dataclass

import torch
from torch import nn
import pytorch_lightning as pl

from tablestakes import utils, constants
from tablestakes.ml2.data import datapoints, tablestakes_data
from tablestakes.ml2.factored import data_module, logs_mod, opt_mod, trunks, head_mod
from tablestakes.ml2 import factored


@dataclass
class TotalParams(utils.DataclassPlus):
    data: data_module.DataParams
    opt: opt_mod.OptParams = opt_mod.OptParams()
    exp: logs_mod.ExperimentParams = logs_mod.ExperimentParams()
    # data: data_module.DataParams = data_module.DataParams(dataset_name='num=100_8163')
    logs: logs_mod.LoggingParams = logs_mod.LoggingParams()

    embed: trunks.BertEmbedder.ModelParams = trunks.BertEmbedder.ModelParams()
    conv: trunks.ConvBlock.ModelParams = trunks.ConvBlock.ModelParams()
    trans: trunks.TransBlockBuilder.ModelParams = trunks.TransBlockBuilder.ModelParams()
    fc: trunks.SlabNet.ModelParams = trunks.SlabNet.ModelParams()
    heads: head_mod.HeadedSlabNet.ModelParams = head_mod.HeadedSlabNet.ModelParams()

    verbose: bool = False

    def __init__(self, data: data_module.DataParams, max_seq_len=1024, batch_size=32):
        super().__init__()
        self.data = data

        self.embed.max_seq_len = self.data.max_seq_length = max_seq_len
        self.data.batch_size = self.opt.batch_size = batch_size


class ModelBertConvTransTClass2(factored.FactoredLightningModule):
    def __init__(
            self,
            hp: TotalParams,
            dm: data_module.XYMetaHandlerDatasetModule,
            opt_maker: opt_mod.OptimizersMaker,
    ):
        super().__init__(hp, opt_maker)
        # for autocomplete
        assert isinstance(self.hp, TotalParams)

        self.verbose = self.hp.verbose

        self.dm = dm
        num_x_base_features = self.dm.num_x_base_dims

        self.num_y_classes: datapoints.YDatapoint = self.dm.num_y_classes
        self.example_input_array = self.dm.example_input_array

        ###############################################################
        # MODEL
        self.embed = trunks.BertEmbedder(self.hp.embed)

        # cat here
        num_embedcat_features = self.hp.embed.dim + num_x_base_features

        if self.hp.conv.num_layers == 0:
            self.conv = None
            num_conv_features = 0
        else:
            self.conv = trunks.ConvBlock(
                num_input_features=num_embedcat_features,
                hp=self.hp.conv,
            )
            num_conv_features = self.conv.get_num_output_features()

        if self.hp.trans.num_layers == 0:
            self.trans = None
            num_trans_features = 0
        else:
            self.trans = trunks.TransBlockBuilder.build(hp=hp.trans, num_input_features=num_embedcat_features)
            num_trans_features = num_embedcat_features

        num_fc_features = num_x_base_features + num_trans_features + num_conv_features
        self.fc = trunks.SlabNet(
            num_input_features=num_fc_features,
            hp=self.hp.fc,
        )

        heads = {}
        for y_name, num_classes in self.num_y_classes:
            heads[y_name] = head_mod.HeadedSlabNet(
                num_input_features=self.fc.get_num_outputs(),
                num_output_features=num_classes,
                head_maker=hp.heads.makers[y_name],
                hp=self.hp.heads,
            )

        self.heads = nn.ModuleDict(heads)
        # END MODEL
        ###############################################################

        hpd = self.hp.to_dict()
        self.hparams = hpd
        self.save_hyperparameters(hpd)
        self.hparams.lr = self.hp.opt.lr

    def forward(self, x: datapoints.BaseVocabDatapoint):
        base = x.base
        vocab = x.vocab

        if self.verbose:
            print(f'base.shape: {base.shape}')
            print(f'vocab.shape: {vocab.shape}')

        x = self.embed(vocab)

        if self.verbose:
            print(f'x.shape after embed: {x.last_hidden_state.shape}')

        x = torch.cat([base, x.last_hidden_state], dim=-1)

        num_batch, num_seq, _ = base.shape

        if self.verbose:
            print(f'x.shape after base_cat: {x.shape}')

        x_trans = self.trans(x) if self.trans else torch.zeros(num_batch, num_seq, 0, requires_grad=False)
        x_conv = self.conv(x) if self.conv else torch.zeros(num_batch, num_seq, 0, requires_grad=False)

        if self.verbose:
            print(f'x_trans.shape: {x_trans.shape}')
            print(f'x_conv.shape: {x_conv.shape}')

        # concatenate for sharpness
        x = torch.cat([base, x_trans, x_conv], dim=-1)

        if self.verbose:
            print('x shape after cat before fc: ', x.shape)

        x = self.fc(x)

        if self.verbose:
            print('x shape after fc before heads: ', x.shape)

        y_hats = {
            head_name: head(x)
            for head_name, head in self.heads.items()
        }

        return y_hats


def run(
        net: pl.LightningModule,
        dm: pl.LightningDataModule,
        hp: TotalParams,
        fast_dev_run=False,
        do_find_lr=False,
):
    print("model run about to create trainer")
    trainer = pl.Trainer(
        logger=True if fast_dev_run else logs_mod.get_pl_logger(hp.exp),
        default_root_dir=hp.logs.output_dir,
        callbacks=[logs_mod.CounterTimerCallback()],
        max_epochs=hp.opt.num_epochs,
        gpus=hp.data.num_gpus,
        weights_summary='full',
        fast_dev_run=fast_dev_run,
        accumulate_grad_batches=1,
        profiler=True,
        deterministic=True,
        auto_lr_find=do_find_lr,
        log_every_n_steps=hp.logs.num_steps_per_metric_log,
    )
    print("model run done creating trainer")

    if do_find_lr:
        utils.hprint("Starting trainer.tune:")
        lr_tune_out = trainer.tune(net, datamodule=dm)
        print(f'  Tune out: {lr_tune_out}')
    else:
        utils.hprint("Starting trainer.fit:")
        print(f'  Dataset file: {hp.data.dataset_file}')
        trainer.fit(net, datamodule=dm)

    utils.hprint('Done with model run fn')


if __name__ == '__main__':
    fast_dev_run = False

    dp = data_module.DataParams(
        dataset_name='num=100_8163',
    )
    hp = TotalParams(
        max_seq_len=8192,
        batch_size=32,
        data=dp,
    )
    # hp = TotalParams()
    # print(f'hp: {hp}')

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

    hp.logs.num_steps_per_histogram_log = 5
    hp.logs.num_steps_per_metric_log = 5
    hp.logs.output_dir = constants.OUTPUT_DIR

    hp.exp.project_name = 'tablestakes'
    hp.exp.experiment_name = 'korv which'
    hp.exp.experiment_tags = ['korv_which', 'conv', 'sharp', 'testing']
    hp.exp.sources_glob_str = constants.THIS_DIR.parent.parent / '**/*.py'
    hp.exp.offline_mode = True

    hp.embed.dim = 15
    hp.embed.requires_grad = True

    hp.conv.num_features = 128
    hp.conv.num_layers = 4
    hp.conv.kernel_size = 3
    hp.conv.num_groups = 16
    hp.conv.num_blocks_per_pool = 20
    hp.conv.requires_grad = True

    hp.trans.impl = 'fast-favor'
    # hp.trans.impl = 'fast'
    hp.trans.num_heads = 8
    hp.trans.num_layers = 6
    hp.trans.num_query_features = None
    hp.trans.fc_dim_mult = 2
    hp.trans.p_dropout = 0.1
    hp.trans.p_attention_dropout = 0.1

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
    hp.heads.makers = {
        constants.Y_WHICH_KV_BASE_NAME: None,
        constants.Y_KORV_BASE_NAME: None,
    }

    hp.verbose = True

    dm = tablestakes_data.TablestakesHandlerDataModule(hp.data)
    net = ModelBertConvTransTClass2(
        hp=hp,
        dm=dm,
        opt_maker=opt_mod.OptimizersMaker(hp.opt),
    )

    utils.hprint('About to start model run:')
    utils.print_dict(hp.to_dict())

    run(net, dm, hp, fast_dev_run, do_find_lr=False)
