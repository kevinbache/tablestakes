import argparse
from pathlib import Path
from typing import *

import boto3

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import ray
from ray import tune
from ray.tune import logger as tune_logger
from ray.tune.integration import wandb as tune_wandb, pytorch_lightning as tune_pl

from chillpill import params

from tablestakes import constants, utils, load_makers
from tablestakes.ml import hyperparams, model_transformer
from tablestakes.ml import torch_helpers


def train_fn(config: Dict, checkpoint_dir=None):
    hp = hyperparams.LearningParams.from_dict(config)
    assert isinstance(hp, hyperparams.LearningParams)

    utils.set_seeds(hp.seed)

    phase_names = model_transformer.RectTransformerModule.PHASE_NAMES

    phase_metric_names = {
        name: model_transformer.RectTransformerModule.get_all_metric_names_for_phase(name) for name in phase_names
    }

    callbacks = [
        torch_helpers.ParamCounterCallback(),
        torch_helpers.LogCopierCallback(),
        tune_pl.TuneReportCallback(
            metrics=phase_metric_names[model_transformer.RectTransformerModule.TRAIN_PHASE_NAME],
            on='train_end',
        ),
        tune_pl.TuneReportCheckpointCallback(
            metrics=phase_metric_names[model_transformer.RectTransformerModule.VALID_PHASE_NAME],
            filename=checkpoint_dir or constants.CHECKPOINT_FILE_BASENAME,
            on='validation_end'
        ),
        tune_pl.TuneReportCallback(
            metrics=phase_metric_names[model_transformer.RectTransformerModule.TEST_PHASE_NAME],
            on='test_end',
        ),
    ]
    logger = pl_loggers.TensorBoardLogger(
        save_dir=tune.get_trial_dir(),
        name=hp.experiment_name,
        version=tune.get_trial_id(),
    )
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        max_epochs=hp.num_epochs,
        gpus=None if is_local_run else hp.num_gpus,
        weights_summary='full',
        accumulate_grad_batches=utils.pow2int(hp.log2_batch_size),
        profiler=True,
    )
    net = model_transformer.RectTransformerModule(hp)

    fit_out = trainer.fit(net)
    return fit_out


if __name__ == '__main__':
    # e.g. num=1000_68eb
    # dataset_name = 'num=10_40db'
    # dataset_name = 'num=1000_4d8d'
    # dataset_name = 'num=2000_56d2'
    dataset_name = 'num=10000_99e0'
    search_params = hyperparams.LearningParams(dataset_name)

    import os
    os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

    ##############
    # model
    #  embedder
    search_params.num_embedding_dim = params.Discrete([20, 20+16, 20+32])
    search_params.do_include_embeddings = params.Boolean(p_true=0.9)

    #  transformer
    search_params.pre_trans_linear_dim = None

    search_params.num_trans_enc_layers = params.Discrete([4, 6, 8, 12, 16])
    search_params.num_trans_heads = params.Discrete([2, 4, 8, 16])
    search_params.num_trans_fc_dim_mult = params.Discrete([2, 3, 4, 6])

    search_params.do_cat_x_base_before_fc = params.Boolean(p_true=0.9)

    #  fully connected
    search_params.num_fc_blocks = params.Discrete([4, 6, 8, 12, 16])
    search_params.log2num_neurons_start = params.Integer(3, 10)
    search_params.log2num_neurons_end = params.Integer(3, 10)
    search_params.num_fc_blocks_per_resid = params.Integer(1, 4)

    search_params.num_fc_layers_per_dropout = params.Discrete([1, 2, 4, 8])
    # prob of dropping each unit
    search_params.dropout_p = params.Float(0.1, 0.7)

    search_params.num_head_blocks = params.Discrete([1, 2, 4, 8])
    search_params.log2num_head_neurons = params.Integer(3, 8)
    search_params.num_head_blocks_per_resid = params.Integer(1, 3)

    search_params.num_groups_for_gn = params.Discrete([16, 32])

    ##############
    # optimization
    search_params.lr = params.Discrete([3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])

    # korv, which_kv
    search_params.korv_loss_weight = params.Discrete([0.1, 0.5, 1.0])

    search_params.num_epochs = 1000

    ##############
    # hp search
    search_params.num_hp_samples = 100
    search_params.search_metric = model_transformer.RectTransformerModule.get_valid_metric_name('acc', 'which_kv')
    search_params.search_mode = 'max'
    search_params.asha_grace_period = 4
    search_params.asha_reduction_factor = 4

    ##############
    # data
    # batch size must be 1
    search_params.batch_size_log2 = params.Integer(3, 11)
    search_params.p_valid = 0.1
    search_params.p_test = 0.1

    # for data loading
    search_params.num_workers = 4

    ##############
    # extra
    search_params.num_steps_per_histogram_log = 50
    search_params.upload_dir = 's3://kb-tester-2020-10-12'
    search_params.project_name = 'tablestakes'
    search_params.experiment_name = 'trans_v0.1.1'
    search_params.num_gpus = 1
    search_params.seed = 42

    do_test_one = True
    if do_test_one:
        dataset_name = 'num=1000_4d8d'
        search_params = hyperparams.LearningParams(dataset_name)
        search_params.num_hp_samples = 1
        search_params.num_epochs = 10
        search_params.num_gpus = 1
        print("=======================================")
        print("=======================================")
        print("============ TESTING ON 1 =============")
        print("=======================================")
        print("=======================================")

    print('search_params.upload_dir', search_params.upload_dir)
    bucket_name = search_params.upload_dir.replace('s3://', '')

    s3_res = boto3.resource('s3')
    s3_client = boto3.client('s3')
    if s3_res.Bucket(bucket_name) not in s3_client.list_buckets():
        s3_client.create_bucket(Bucket=bucket_name)
    sync_config = tune.SyncConfig(upload_dir=search_params.upload_dir)

    import socket
    hostname = socket.gethostname()
    is_local_run = hostname.endswith('.local')
    # do_fast_test = is_local_run
    do_fast_test = False

    if do_fast_test:
        search_params.num_epochs = 10
        search_params.num_hp_samples = 2
        search_params.num_trans_enc_layers = 1
        search_params.num_trans_heads = 1
        search_params.num_trans_fc_dim_mult = 1
        search_params.num_fc_blocks = 2
        search_params.num_head_blocks = 1
        search_params.log2num_head_neurons = 5
        search_params.dataset_name = 'num=10_40db'
        search_params.update_files()
        sync_config = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default='auto')
    args = parser.parse_args()

    ray.shutdown()
    ray.init(
        address=None if is_local_run else str(args.address),
        ignore_reinit_error=True,
        include_dashboard=True,
        local_mode=False,
        # num_cpus=8 if is_local_run else None,
        # num_gpus=4 if is_local_run else None,
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

    assert train_loss_name == 'train_loss_total'

    reporter_metric_cols = [
        'training_iteration',
        'time_this_iter_s',
        'time_total_s',
        model_transformer.RectTransformerModule.get_train_metric_name(metric_name='loss', output_name='korv'),
        model_transformer.RectTransformerModule.get_train_metric_name(metric_name='loss', output_name='which_kv'),
        train_loss_name,
        model_transformer.RectTransformerModule.get_valid_metric_name(metric_name='loss', output_name='korv'),
        model_transformer.RectTransformerModule.get_valid_metric_name(metric_name='loss', output_name='which_kv'),
        valid_loss_name,
        model_transformer.RectTransformerModule.get_valid_metric_name(metric_name='acc', output_name='korv'),
        search_params.search_metric,
    ]

    param_cols = [torch_helpers.ParamCounterCallback.PARAM_COUNT_NAME] + search_params.get_samplable_param_names()
    reporter = tune.CLIReporter(
        max_progress_rows=search_params.num_hp_samples,
        parameter_columns=param_cols,
        metric_columns=reporter_metric_cols,
    )

    search_dict = search_params.to_ray_tune_search_dict()
    loggers = list(tune_logger.DEFAULT_LOGGERS)

    if not do_fast_test:
        train_fn = tune_wandb.wandb_mixin(train_fn)

        search_dict['wandb'] = {
            "project": search_params.project_name,
            "api_key_file": Path('~/.wandb_api_key').expanduser().resolve(),
        }
        loggers += [tune_wandb.WandbLogger]

    # blocks until done
    print('loading or making data')
    ds = load_makers.DatasetLoadMaker(
        saved_dataset_file=search_params.dataset_file,
        input_docs_directory_for_maker=search_params.docs_dir,
    )

    resources_per_trial = {
        "cpu": 2,
    }
    if not is_local_run:
        resources_per_trial['gpu'] = search_params.num_gpus

    analysis = tune.run(
        run_or_experiment=train_fn,
        name=f'{search_params.project_name}-{search_params.experiment_name}',
        stop={"training_iteration": search_params.num_epochs},
        config=search_dict,
        resources_per_trial=resources_per_trial,
        num_samples=search_params.num_hp_samples,
        # local_dir=None,
        # upload_dir=search_params.upload_dir,
        # trial_name_creator=,
        sync_config=sync_config,
        loggers=loggers,
        log_to_file=True,
        # sync_to_cloud=None,
        # sync_to_driver=None,
        checkpoint_freq=0,
        checkpoint_at_end=False,
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

    analysis_file = constants.LOGS_DIR / f'tune_analysis_{search_params.get_short_hash(num_chars=8)}.pkl'
    print(f"Saving {analysis_file}")
    utils.save_pickle(analysis_file, analysis)

    best_trial = analysis.get_best_trial(search_params.search_metric, "max", "last-5-avg")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result[valid_loss_name]))
    print("Best trial final search_metric: {}".format(best_trial.last_result[search_params.search_metric]))

