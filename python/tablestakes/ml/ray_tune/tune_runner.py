import argparse
from pathlib import Path
from typing import *

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import ray
from ray import tune
from ray.tune import logger as tune_logger
from ray.tune.integration import wandb as tune_wandb, pytorch_lightning as tune_pl

from chillpill import params

from tablestakes import constants, utils
from tablestakes.ml import hyperparams, model_transformer


class ParamCounterCallback(pl.Callback):
    PARAM_COUNT_NAME = 'param_count'

    def on_train_start(self, trainer, pl_module):
        d = {
            self.PARAM_COUNT_NAME: sum(p.numel() for p in pl_module.parameters() if p.requires_grad),
        }
        tune.report(**d)


def train_fn(config: Dict):
    hp = hyperparams.LearningParams.from_dict(config)
    assert isinstance(hp, hyperparams.LearningParams)

    utils.set_seeds(hp.seed)

    phase_names = model_transformer.RectTransformerModule.PHASE_NAMES

    phase_metric_names = {
        name: model_transformer.RectTransformerModule.get_all_metric_names_for_phase(name) for name in phase_names
    }

    callbacks = [
        ParamCounterCallback(),
        tune_pl.TuneReportCallback(
            metrics=phase_metric_names[model_transformer.RectTransformerModule.TRAIN_PHASE_NAME],
            on='train_end',
        ),
        tune_pl.TuneReportCheckpointCallback(
            metrics=phase_metric_names[model_transformer.RectTransformerModule.VALID_PHASE_NAME],
            filename=constants.CHECKPOINT_FILE_BASENAME,
            on='validation_end'
        ),
        tune_pl.TuneReportCallback(
            metrics=phase_metric_names[model_transformer.RectTransformerModule.TEST_PHASE_NAME],
            on='test_end',
        ),
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=pl_loggers.TensorBoardLogger(constants.LOGS_DIR, name=hp.project_name),
        max_epochs=hp.num_epochs,
        weights_summary='full',
        accumulate_grad_batches=utils.pow2int(hp.log2_batch_size),
        profiler=True,
    )
    net = model_transformer.RectTransformerModule(hp)

    fit_out = trainer.fit(net)
    return fit_out


if __name__ == '__main__':
    search_params = hyperparams.LearningParams(
        ##############
        # model
        #  embedder
        num_embedding_dim=params.Discrete([20, 20+16, 20+32]),
        do_include_embeddings=params.Boolean(p_true=0.9),

        #  transformer
        pre_trans_linear_dim=None,

        num_trans_enc_layers=params.Discrete([4, 6, 8, 12, 16]),
        num_trans_heads=params.Discrete([2, 4, 8, 16]),
        num_trans_fc_dim_mult=params.Discrete([2, 3, 4, 6]),

        do_cat_x_base_before_fc=params.Boolean(p_true=0.9),

        #  fully connected
        num_fc_blocks=params.Discrete([4, 6, 8, 12, 16]),
        log2num_neurons_start=params.Integer(3, 10),
        log2num_neurons_end=params.Integer(3, 10),
        num_fc_blocks_per_resid=params.Integer(1, 4),

        num_fc_layers_per_dropout=params.Discrete([1, 2, 4, 8]),
        # prob of dropping each unit
        dropout_p=params.Float(0.1, 0.7),

        ##############
        # optimization
        lr=params.Discrete([1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),

        # korv, which_kv
        korv_loss_weight=params.Discrete([0.1, 0.5, 1.0]),

        num_epochs=1000,

        ##############
        # hp search
        num_hp_samples=100,
        search_metric=model_transformer.RectTransformerModule.get_valid_metric_name('acc', 'which_kv'),
        search_mode='max',
        asha_grace_period=4,
        asha_reduction_factor=4,

        ##############
        # data
        # batch size must be 1
        batch_size_log2=params.Integer(0, 12),
        p_valid=0.1,
        p_test=0.1,
        data_dir=constants.DOCS_DIR / 'num=1000_4475',

        # for data loading
        num_workers=4,

        ##############
        # extra
        num_steps_per_histogram_log=50,

        upload_dir='s3://kb-tester-2020-10-08',
        project_name='tablestakes_trans1d_tests',

        seed=42,
    )

    import socket
    do_fast_test = socket.gethostname().endswith('.local')

    if do_fast_test:
        search_params.num_epochs = 10
        search_params.num_hp_samples = 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default='localhost:6379')
    args = parser.parse_args()

    ray.init(
        address=None if do_fast_test else str(args.address),
        ignore_reinit_error=True,
        include_dashboard=not do_fast_test,
        local_mode=do_fast_test,
    )

    tune_scheduler = tune.schedulers.ASHAScheduler(
        metric=search_params.search_metric,
        mode=search_params.search_mode,
        grace_period=search_params.asha_grace_period,
        reduction_factor=search_params.asha_reduction_factor,
    )

    train_loss_name = \
        model_transformer.RectTransformerModule.get_train_metric_name(metric_name='loss', output_name='total')
    valid_loss_name = \
        model_transformer.RectTransformerModule.get_valid_metric_name(metric_name='loss', output_name='total')

    reporter_metric_cols = [
        'iter',
        'total time (s)',
        'ts',
        model_transformer.RectTransformerModule.get_train_metric_name(metric_name='loss', output_name='korv'),
        model_transformer.RectTransformerModule.get_train_metric_name(metric_name='loss', output_name='which_kv'),
        train_loss_name,
        model_transformer.RectTransformerModule.get_valid_metric_name(metric_name='loss', output_name='korv'),
        model_transformer.RectTransformerModule.get_valid_metric_name(metric_name='loss', output_name='which_kv'),
        valid_loss_name,
        search_params.search_metric,
    ]

    reporter = tune.CLIReporter(
        max_progress_rows=search_params.num_hp_samples,
        parameter_columns=[ParamCounterCallback.PARAM_COUNT_NAME] + search_params.get_samplable_param_names(),
        metric_columns=reporter_metric_cols,
    )

    search_dict = search_params.to_ray_tune_search_dict()
    loggers = list(tune_logger.DEFAULT_LOGGERS)
    if not do_fast_test:
        search_dict['wandb'] = {
            "project": search_params.project_name,
            "api_key_file": Path('~/.wandb_api_key').expanduser().resolve(),
            "log_config": True,
        }
        loggers += [tune_wandb.WandbLogger]

    analysis = tune.run(
        run_or_experiment=train_fn,
        # name=None,
        stop={"training_iteration": search_params.num_epochs},
        config=search_dict,
        resources_per_trial={"cpu": 1, "gpu": 0},
        num_samples=search_params.num_hp_samples,
        # local_dir=None,
        # upload_dir=search_params.upload_dir,
        trial_name_creator=None,
        sync_config=tune.SyncConfig(upload_dir=search_params.upload_dir),
        loggers=loggers,
        # sync_to_cloud=None,
        # sync_to_driver=None,
        checkpoint_freq=2,
        checkpoint_at_end=True,
        # sync_on_checkpoint=True,
        keep_checkpoints_num=2,
        checkpoint_score_attr=search_params.search_metric,
        # global_checkpoint_period=10,
        # export_formats=None,
        # max_failures=0,
        fail_fast=False,
        # restore=None,
        # search_alg=search_alg,
        scheduler=tune_scheduler,
        # with_server=False,
        # server_port=TuneServer.DEFAULT_PORT,
        verbose=2,
        progress_reporter=reporter,
        # resume=False,
        queue_trials=False,
        # reuse_actors=False,
        # trial_executor=None,
        # raise_on_failed_trial=True,
        # return_trials=False,
        # ray_auto_init=True,
    )

    utils.save_pickle('tune_analysis.pkl', analysis)

    best_trial = analysis.get_best_trial(search_params.search_metric, "max", "last-5-avg")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result[valid_loss_name]))
    print("Best trial final search_metric: {}".format(best_trial.last_result[search_params.search_metric]))
