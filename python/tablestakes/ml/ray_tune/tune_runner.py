import argparse
from pathlib import Path
from typing import *

import boto3

import pytorch_lightning as pl

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
    phase_to_metric_names = {
        pn: model_transformer.RectTransformerModule.get_all_metric_names_for_phase(pn) for pn in phase_names
    }

    pl_callbacks = [
        torch_helpers.LogCopierCallback(),
        tune_pl.TuneReportCallback(
            metrics=phase_to_metric_names[model_transformer.RectTransformerModule.TRAIN_PHASE_NAME],
            on='train_end',
        ),
        tune_pl.TuneReportCheckpointCallback(
            metrics=phase_to_metric_names[model_transformer.RectTransformerModule.VALID_PHASE_NAME],
            filename=checkpoint_dir or constants.CHECKPOINT_FILE_BASENAME,
            on='validation_end',
        ),
        tune_pl.TuneReportCallback(
            metrics=phase_to_metric_names[model_transformer.RectTransformerModule.TEST_PHASE_NAME],
            on='test_end',
        ),
    ]
    trainer = pl.Trainer(
        callbacks=pl_callbacks,
        logger=torch_helpers.get_pl_logger(hp, tune),
        max_epochs=hp.num_epochs,
        gpus=None if is_local_run else hp.num_gpus,
        weights_summary='full',
        # accumulate_grad_batches=utils.pow2int(hp.log2_batch_size),
        profiler=True,
        deterministic=True,
        log_every_n_steps=hp.num_steps_per_metric_log,
    )
    net = model_transformer.RectTransformerModule(hp)

    fit_out = trainer.fit(net)
    print('Done with this fit run')

    return fit_out


if __name__ == '__main__':
    # e.g. num=1000_68eb
    # dataset_name = 'num=10_40db'
    # dataset_name = 'num=1000_4d8d'
    # dataset_name = 'num=2000_56d2'
    dataset_name = 'num=10000_99e0'
    search_params = hyperparams.LearningParams(dataset_name)

    # this is required because different types of trial results either do or don't get certain metrics
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
    search_params.log2_batch_size = params.Integer(0, 11)

    # korv, which_kv
    search_params.korv_loss_weight = params.Discrete([0.1, 0.5, 1.0])

    search_params.num_epochs = 100

    ##############
    # hp search
    search_params.num_hp_samples = 100
    search_params.search_metric = model_transformer.RectTransformerModule.get_valid_metric_name('acc', 'which_kv')
    search_params.search_mode = 'max'
    search_params.asha_grace_period = 4
    search_params.asha_reduction_factor = 2

    ##############
    # data
    # batch size must be 1
    search_params.p_valid = 0.1
    search_params.p_test = 0.1

    # for data loading
    search_params.num_workers = 4

    ##############
    # extra
    search_params.num_steps_per_histogram_log = 100
    search_params.upload_dir = 's3://kb-tester-2020-10-14'
    search_params.project_name = 'tablestakes'
    search_params.experiment_name = 'trans_v0.1.3'
    search_params.group_name = 'log2_batch_2'
    search_params.experiment_tags = ['tune', 'testing']

    search_params.num_cpus = 2
    search_params.num_gpus = 1
    search_params.seed = 42

    do_test_one = False
    if do_test_one:
        dataset_name = 'num=1000_4d8d'
        search_params = hyperparams.LearningParams(dataset_name)
        search_params.num_hp_samples = 4
        search_params.num_epochs = 10
        search_params.num_cpus = 0.5
        search_params.num_gpus = 0.5
        search_params.experiment_tags.append('do_test_one')
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
    search_params.experiment_tags.append('local' if is_local_run else 'cluster')

    do_fast_test = is_local_run

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
        search_params.experiment_tags.append('do_fast_test')
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
        torch_helpers.PARAM_COUNT_NAME,
        torch_helpers.CURRENT_EPOCH_NAME,
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

    param_cols = search_params.get_samplable_param_names()
    reporter = tune.CLIReporter(
        max_progress_rows=search_params.num_hp_samples,
        parameter_columns=param_cols,
        metric_columns=reporter_metric_cols,
    )

    search_dict = search_params.to_ray_tune_search_dict()

    # see tune.utils.UtilMonitor
    LOG_SYS_USAGE = 'log_sys_usage'
    util_keys = [
        LOG_SYS_USAGE,
    ]
    search_dict[LOG_SYS_USAGE] = True

    tune_loggers = list(tune_logger.DEFAULT_LOGGERS)

    if not is_local_run:
        tune_loggers.append(torch_helpers.TuneNeptuneLogger)

    # blocks until done
    print('loading or making data')
    ds = load_makers.TablestakesDatasetLoadMaker(
        saved_dataset_file=search_params.dataset_file,
        input_docs_directory_for_maker=search_params.docs_dir,
    )

    resources_per_trial = {
        "cpu": search_params.num_cpus,
    }
    if not is_local_run:
        resources_per_trial['gpu'] = search_params.num_gpus

    def do_stop(trial_id, result):
        if torch_helpers.CURRENT_EPOCH_NAME in result:
            return result[torch_helpers.CURRENT_EPOCH_NAME] > search_params.num_epochs
        else:
            return False
    stopper = tune.stopper.FunctionStopper(do_stop)

    analysis = tune.run(
        run_or_experiment=train_fn,
        name=search_params.get_project_exp_name(),
        stop=stopper,
        config=search_dict,
        resources_per_trial=resources_per_trial,
        num_samples=search_params.num_hp_samples,
        # local_dir=None,
        # upload_dir=search_params.upload_dir,
        # trial_name_creator=,
        sync_config=sync_config,
        loggers=tune_loggers,
        log_to_file=False,
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

    print(f'best_trial.last_result: {best_trial.last_result}')

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result[valid_loss_name]))
    print("Best trial final search_metric: {}".format(best_trial.last_result[search_params.search_metric]))

