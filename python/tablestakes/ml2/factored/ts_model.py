from typing import *

import torch
import pytorch_lightning as pl

from tablestakes import utils, constants
from tablestakes.ml2 import factored
from tablestakes.ml2.data import datapoints, tablestakes_data
from tablestakes.ml2.factored import data_module, logs_mod, opt_mod, trunks_mod, head_mod

from chillpill import params


class TotalParams(params.ParameterSet):
    data: data_module.DataParams
    opt: opt_mod.OptParams = opt_mod.OptParams()
    exp: logs_mod.ExperimentParams = logs_mod.ExperimentParams()
    logs: logs_mod.LoggingParams = logs_mod.LoggingParams()

    embed: trunks_mod.BertEmbedder.ModelParams = trunks_mod.BertEmbedder.ModelParams()
    conv: trunks_mod.ConvBlock.ModelParams = trunks_mod.ConvBlock.ModelParams()
    trans: trunks_mod.TransBlockBuilder.ModelParams = trunks_mod.TransBlockBuilder.ModelParams()
    fc: trunks_mod.SlabNet.ModelParams = trunks_mod.SlabNet.ModelParams()
    neck: head_mod.HeadedSlabNet.ModelParams = head_mod.HeadedSlabNet.ModelParams()
    head: head_mod.WeightedHeadParams = head_mod.WeightedHeadParams(weights={}, head_mod={})

    verbose: bool = False

    def __init__(self, data: data_module.DataParams = None, max_seq_len=1024, batch_size=32):
        super().__init__()
        # TODO: hack.
        if data is None:
            data = data_module.DataParams()
            data.data_name = 'DUMMY_DATASET'
        self.data = data

        self.embed.max_seq_len = max_seq_len
        self.data.max_seq_length = max_seq_len
        self.data.batch_size = batch_size
        self.opt.batch_size = batch_size


class ModelBertConvTransTClass2(factored.FactoredLightningModule):
    def __init__(
            self,
            hp: TotalParams,
            dm: data_module.XYMetaHandlerDatasetModule,
            opt: opt_mod.OptimizersMaker,
            head_maker: Callable[[int], head_mod.Head],
    ):
        super().__init__(hp, opt)
        # for autocomplete
        assert isinstance(self.hp, TotalParams)

        self.verbose = self.hp.verbose

        self.dm = dm
        num_x_base_features = self.dm.num_x_base_dims

        self.num_y_classes: datapoints.YDatapoint = self.dm.num_y_classes
        self.example_input_array = self.dm.example_input_array

        ###############################################################
        # MODEL
        self.embed = trunks_mod.BertEmbedder(self.hp.embed)

        # cat here
        num_embedcat_features = self.hp.embed.dim + num_x_base_features

        if self.hp.conv.num_layers == 0:
            self.conv = None
            num_conv_features = 0
        else:
            self.conv = trunks_mod.ConvBlock(
                num_input_features=num_embedcat_features,
                hp=self.hp.conv,
            )
            num_conv_features = self.conv.get_num_output_features()

        if self.hp.trans.num_layers == 0:
            self.trans = None
            num_trans_features = 0
        else:
            self.trans = trunks_mod.TransBlockBuilder.build(hp=hp.trans, num_input_features=num_embedcat_features)
            num_trans_features = self.trans.get_num_output_features()

        num_fc_features = num_x_base_features + num_trans_features + num_conv_features
        self.fc = trunks_mod.SlabNet(
            num_input_features=num_fc_features,
            hp=self.hp.fc,
        )
        self.head = head_maker(self.fc.get_num_outputs())
        # END MODEL
        ###############################################################

        hpd = self.hp.to_dict()
        self.hparams = hpd
        self.save_hyperparameters(hpd)
        self.hparams.lr = self.hp.opt.lr

    def forward(self, x: datapoints.BaseVocabDatapoint):
        base = x.base
        vocab = x.vocab
        assert isinstance(base, torch.Tensor)
        assert isinstance(vocab, torch.Tensor)

        # with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True, profile_memory=True) as prof:
        if self.verbose:
            print(f'base.shape: {base.shape}')
            print(f'vocab.shape: {vocab.shape}')

        with utils.Timer('ts_model forward embed', do_print_outputs=self.hp.verbose):
            x = self.embed(vocab)

        if self.verbose:
            print(f'x.shape after embed: {x.last_hidden_state.shape}')

        x = torch.cat([base, x.last_hidden_state], dim=-1)

        num_batch, num_seq, _ = base.shape

        if self.verbose:
            print(f'x.shape after base_cat: {x.shape}')

        with utils.Timer('ts_model forward trans', do_print_outputs=self.hp.verbose):
            x_trans = self.trans(x) if self.trans \
                else torch.zeros(num_batch, num_seq, 0, requires_grad=False, device=self.device)

        with utils.Timer('ts_model forward conv', do_print_outputs=self.hp.verbose):
            x_conv = self.conv(x) if self.conv \
                else torch.zeros(num_batch, num_seq, 0, requires_grad=False, device=self.device)

        if self.verbose:
            print(f'x_trans.shape: {x_trans.shape}')
            print(f'x_conv.shape: {x_conv.shape}')

        # concatenate for sharpness
        x = torch.cat([base, x_trans, x_conv], dim=-1)

        if self.verbose:
            print('x shape after cat before fc: ', x.shape)

        with utils.Timer('ts_model forward fc', do_print_outputs=self.hp.verbose):
            x = self.fc(x)

        if self.verbose:
            print('x shape after fc before head: ', x.shape)

        with utils.Timer('ts_model forward head', do_print_outputs=self.hp.verbose):
            x_for_loss, x_for_pred = self.head.forward_for_loss_and_pred(x)

        # utils.hprint('MODEL FORWARD PROFILER REPORT:')
        # print(prof.key_averages().table(sort_by="cuda_memory_usage"))

        return x_for_loss, x_for_pred


def run(
        net: pl.LightningModule,
        hp: TotalParams,
        fast_dev_run=False,
        do_find_lr=False,
        callbacks=None,
):
    utils.set_seeds(hp.data.seed)

    if callbacks is None:
        callbacks = [
            logs_mod.CounterTimerLrCallback(),
            logs_mod.VocabLengthCallback(),
        ]

    print("model run about to create trainer")
    trainer = pl.Trainer(
        logger=True if fast_dev_run else logs_mod.get_pl_logger(hp.exp),
        default_root_dir=hp.logs.output_dir,
        callbacks=callbacks,
        max_epochs=hp.opt.num_epochs,
        gpus=hp.data.num_gpus,
        weights_summary='full',
        fast_dev_run=fast_dev_run,
        accumulate_grad_batches=1,
        profiler='simple',
        deterministic=True,
        auto_lr_find=do_find_lr,
        log_every_n_steps=hp.logs.num_steps_per_metric_log,
    )
    print("model run done creating trainer")

    if do_find_lr:
        utils.hprint("Starting trainer.tune:")
        lr_tune_out = trainer.tune(net, datamodule=net.dm)
        print(f'  Tune out: {lr_tune_out}')
    else:
        utils.hprint("Starting trainer.fit:")
        print(f'  Dataset file: {hp.data.dataset_file}')
        trainer.fit(net, datamodule=net.dm)

    utils.hprint('Done with model run fn')


class TablestakesBertConvTransTClassModel(ModelBertConvTransTClass2):
    @classmethod
    def from_hp(cls, hp: TotalParams):
        dm = tablestakes_data.TablestakesHandlerDataModule(hp.data)
        net = cls(
            hp=hp,
            dm=dm,
            opt=opt_mod.OptimizersMaker(hp.opt),
            head_maker=head_mod.HeadMakerFactory.create(
                neck_hp=hp.neck,
                head_hp=hp.head,
            ),
        )
        return net


if __name__ == '__main__':
    fast_dev_run = False

    dp = data_module.DataParams(
        # dataset_name='num=100_057b',
        dataset_name='num=1000_2cfc',
    )
    hp = TotalParams(
        data=dp,
        max_seq_len=8192,
        batch_size=32,
    )

    hp.data.do_ignore_cached_dataset = False
    hp.data.seed = 42
    hp.data.num_workers = 4
    hp.data.num_gpus = 0
    hp.data.num_cpus = 4

    hp.opt.search_metric = 'valid/loss'
    hp.opt.search_mode = 'min'
    hp.opt.num_epochs = 4
    hp.opt.lr = 0.001
    hp.opt.patience = 10

    hp.logs.num_steps_per_histogram_log = 20
    hp.logs.num_steps_per_metric_log = 5
    hp.logs.output_dir = constants.OUTPUT_DIR

    hp.exp.project_name = 'tablestakes'
    hp.exp.experiment_name = 'korv which'
    hp.exp.experiment_tags = ['korv_which', 'testing']
    hp.exp.sources_glob_str = constants.THIS_DIR.parent.parent / '**/*.py'
    hp.exp.offline_mode = False

    hp.embed.dim = 15
    hp.embed.requires_grad = True
    hp.embed.position_embedding_requires_grad = False

    hp.conv.num_features = 128
    hp.conv.num_layers = 4
    hp.conv.kernel_size = 3
    hp.conv.num_groups = 16
    hp.conv.num_blocks_per_pool = 20
    hp.conv.requires_grad = True

    hp.trans.impl = 'fast-favor'
    # neck_hp.trans.impl = 'fast'
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

    hp.neck.num_features = 128
    hp.neck.num_layers = 4
    hp.neck.num_groups = 16
    hp.neck.num_blocks_per_residual = 2
    hp.neck.num_blocks_per_dropout = 2
    hp.neck.requires_grad = True

    hp.head = head_mod.WeightedHeadParams(
        weights={
            constants.Y_KORV_BASE_NAME: 0.3,
            constants.Y_WHICH_KV_BASE_NAME: 1.0,
        },
        head_params={
            constants.Y_KORV_BASE_NAME: head_mod.HeadParams(
                type='linear',
                num_classes=2,
                class_names=('key', 'value'),
            ),
            constants.Y_WHICH_KV_BASE_NAME: head_mod.HeadParams(
                type='linear',
                num_classes=11,
                class_names=(
                    'to_address',
                    'sale_address',
                    'from_address',
                    'date_sent',
                    'date_received',
                    'invoice_number',
                    'total',
                    'subtotal',
                    'phone',
                    'fax',
                    'field_0',
                ),
            ),
        },
    )

    hp.verbose = False

    # utils.hprint('tablestakes.model.hp before to_dict:')
    # print(hp)
    # hpd = hp.to_dict()
    # utils.hprint('tablestakes.model.hp after to_dict:')
    # print(hp)
    #
    # utils.hprint('hpd:')
    # utils.print_dict(hpd)
    #
    # utils.hprint('tablestakes.model.hp before from_dict:')
    # print(hp)
    #
    # hprt = TotalParams.from_dict(hpd)
    # utils.hprint('tablestakes.model.hp after from_dict:')
    # print(hp)
    #
    # utils.hprint('tablestakes.model.hprt:')
    # print(hprt)
    # utils.hprint('tablestakes.model.hp after all:')
    # print(hp)

    net = TablestakesBertConvTransTClassModel.from_hp(hp)

    utils.hprint('About to start model run:')
    utils.print_dict(hp.to_dict())

    run(net, hp, fast_dev_run, do_find_lr=False)
