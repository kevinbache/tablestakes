import argparse
import os
import socket
from typing import *

import boto3

import pytorch_lightning as pl

import ray
from ray import tune
from ray.tune import logger as tune_logger

from chillpill import params

from tablestakes import constants, utils
from ray.tune.integration import pytorch_lightning as pl_tune
from tablestakes.ml2.data import data_module, tablestakes_data, datapoints
from tablestakes.ml2.factored import ts_model, opt_mod, head_mod, FactoredParams, logs_mod
from tablestakes.ml2 import data, factored


REGION = 'us-west-2'


class TuneParams(params.ParameterSet):
    asha_grace_period = 4
    asha_reduction_factor = 2
    num_hp_samples = 10
    log_to_file = False
    ray_local_mode = False


class TuneRunner:
    def __init__(
            self,
            model_hp: FactoredParams,
            tune_hp: TuneParams,
            factored_lightning_module_class: type,
            extra_pl_callbacks: Optional[List[pl.callbacks.Callback]] = None,
    ):
        self.include_gpus = None

        # if you leave this at the default false, then every call to tune.report needs to have all
        # expected metrics in it
        os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = "1"

        # self.ensure_hp_is_factored(search_params)
        super().__init__()
        self.hp = model_hp
        self.tune_hp = tune_hp
        self.search_params = self.hp

        self._model_param_class = model_hp.__class__

        assert issubclass(factored_lightning_module_class, factored.FactoredLightningModule)
        self._factored_lightning_module_class = factored_lightning_module_class

        # self._metrics_tracker_class = self._factored_lightning_module_class.get_metrics_tracker_class()

        if extra_pl_callbacks is None:
            extra_pl_callbacks = [
                logs_mod.CounterTimerLrCallback(),
            ]
        self.extra_pl_callbacks = extra_pl_callbacks

        self.create_bucket_if_not_exist(model_hp.logs.output_dir)

        # True if running on macbook laptop, False if running on cluster
        is_running_on_local_machine = self.add_local_cluster_tag(model_hp)

        ################
        #   init ray   #
        ################
        args = {
            'address': None if is_running_on_local_machine else 'auto',
            'ignore_reinit_error': True,
            'include_dashboard': True,
            'local_mode': tune_hp.ray_local_mode,
        }
        if tune_hp.ray_local_mode:
            args['num_cpus'] = model_hp.data.num_cpus
        print("About to init ray...")
        import warnings
        warnings.warn(
            'tune_runner.py hard coding node ip address to 127.0.0.1 as workaround to vpn issue.'
            'https://github.com/ray-project/ray/issues/6573'
        )
        ray.services.get_node_ip_address = lambda: '127.0.0.1'
        ray.init(**args)
        print("Done with ray init")

    @staticmethod
    def add_local_cluster_tag(search_params: factored.FactoredParams):
        hostname = socket.gethostname()
        is_local_run = hostname.endswith('.local')
        search_params.exp.experiment_tags.append('local' if is_local_run else 'cluster')
        return is_local_run

    @staticmethod
    def get_address_str_from_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--address", default='auto')
        args = parser.parse_args()
        return str(args.address)

    @staticmethod
    def get_pl_callbacks_for_tune():
        return [
            logs_mod.TuneLogCopierCallback(),
        ]

    @staticmethod
    def create_bucket_if_not_exist(upload_dir: utils.DirtyPath):
        upload_dir = str(upload_dir)
        if not upload_dir.startswith('s3://'):
            return
        bucket_name = upload_dir.replace('s3://', '')
        # s3_res = boto3.resource('s3')
        s3_client = boto3.client('s3', region_name=REGION)
        location = {'LocationConstraint': REGION}
        bucket_names = [b['Name'] for b in s3_client.list_buckets()['Buckets']]
        if bucket_name not in bucket_names:
            s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)

    def _train_fn(self, config: Dict, checkpoint_dir=None, fast_dev_run=False, include_gpus=False):
        utils.hprint('Starting train function with config:')
        utils.print_dict(config)
        print()

        hp = self._model_param_class.from_dict(config)
        assert isinstance(hp, self._model_param_class)
        print('  hp:', hp)

        if checkpoint_dir:
            # see https://docs.ray.io/en/master/tune/user-guide.html#checkpointing
            raise NotImplementedError(f"Got checkpoint_dir in trian_fn: {checkpoint_dir}")

        utils.hprint("About to create net in TuneRunner")
        net = self._factored_lightning_module_class.from_hp(hp=hp)
        # import torch.autograd.profiler as profiler
        # with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
        #     net = self._factored_lightning_module_class.from_hp(hp=hp)
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1000))

        utils.set_seeds(hp.data.seed)

        # noinspection PyTypeChecker
        trainer = pl.Trainer(
            logger=logs_mod.get_pl_logger(hp=hp.exp, tune=tune),
            default_root_dir=tune.get_trial_dir(),
            callbacks=self.extra_pl_callbacks + self.get_pl_callbacks_for_tune(),
            max_epochs=hp.opt.num_epochs,
            gpus=hp.data.num_gpus if include_gpus else None,
            weights_summary='full',
            fast_dev_run=fast_dev_run,
            accumulate_grad_batches=1,
            profiler='simple',
            deterministic=True,
            log_every_n_steps=hp.logs.num_steps_per_metric_log,
            log_gpu_memory=hp.logs.log_gpu_memory,
        )
        utils.hprint('About to start tune_runner\'s trainer.fit...')
        fit_out = trainer.fit(net, datamodule=net.dm)
        utils.hprint('Done with tune_runner._train_fn')

        return fit_out

    def _get_train_fn(self, fast_dev_run: bool, include_gpus: bool):
        """Just a closure"""

        def train_fn(config: Dict, checkpoint_dir: str):
            return self._train_fn(
                config=config,
                checkpoint_dir=checkpoint_dir,
                fast_dev_run=fast_dev_run,
                include_gpus=include_gpus,
            )

        return train_fn

    @staticmethod
    def get_resources_per_trial(search_params: "TuneFactoredParams", include_gpu=False):
        resources_per_trial = {
            "cpu": search_params.data.num_cpus,
        }
        if include_gpu:
            resources_per_trial['gpu'] = search_params.data.num_gpus
        return resources_per_trial

    @staticmethod
    def get_tune_stopper(num_epochs: int):
        def do_stop(_, result):
            if logs_mod.CURRENT_EPOCH_NAME in result:
                return result[logs_mod.CURRENT_EPOCH_NAME] > num_epochs
            else:
                return False

        return tune.stopper.FunctionStopper(do_stop)

    @staticmethod
    def get_tune_loggers():
        tune_loggers = list(tune_logger.DEFAULT_LOGGERS)
        # if not is_local_run:
        #     tune_loggers.append(torch_helpers.TuneNeptuneLogger)
        return tune_loggers

    def get_cli_reporter(self, extra_metric_cols: Optional[List[str]] = None):
        if extra_metric_cols is None:
            extra_metric_cols = []

        default_cli_reporter_metric_cols = [
            logs_mod.PARAM_COUNT_NAME,
            logs_mod.CURRENT_EPOCH_NAME,
            'time_this_iter_s',
            'time_total_s',
            'train_loss_total',
            'valid_loss_total',
            self.search_params.opt.search_metric,
        ]

        # unique
        metric_cols_set = set()  # a temporary lookup set
        metric_cols = [
            x
            for x in default_cli_reporter_metric_cols if x not in metric_cols_set and metric_cols_set.add(x) is None
        ]

        return tune.CLIReporter(
            max_progress_rows=self.tune_hp.num_hp_samples,
            parameter_columns=self.search_params.get_samplable_param_names(),
            metric_columns=metric_cols + extra_metric_cols,
        )

    @staticmethod
    def get_tune_scheduler(search_params: "TuneFactoredParams", tune_hp: TuneParams) -> tune.schedulers.TrialScheduler:
        return tune.schedulers.ASHAScheduler(
            metric=search_params.opt.search_metric,
            mode=search_params.opt.search_mode,
            grace_period=tune_hp.asha_grace_period,
            reduction_factor=tune_hp.asha_reduction_factor,
        )

    def run(self, fast_dev_run=False, use_gpus=False):
        utils.set_seeds(self.search_params.data.seed)

        search_dict = self.search_params.to_ray_tune_search_dict()
        # see tune.utils.UtilMonitor
        search_dict['log_sys_usage'] = True

        output_str = str(self.search_params.logs.output_dir)
        if output_str.startswith('s3://') or output_str.startswith('gs://') or output_str.startswith('hdfs://'):
            sync_config = tune.SyncConfig(upload_dir=self.search_params.logs.output_dir)
        else:
            sync_config = None

        analysis = tune.run(
            run_or_experiment=self._get_train_fn(fast_dev_run=fast_dev_run, include_gpus=use_gpus),
            name=self.search_params.exp.get_project_exp_name(),
            stop=self.get_tune_stopper(self.search_params.opt.num_epochs),
            config=search_dict,
            resources_per_trial=self.get_resources_per_trial(self.search_params, include_gpu=use_gpus),
            num_samples=self.tune_hp.num_hp_samples,
            sync_config=sync_config,
            loggers=self.get_tune_loggers(),
            log_to_file=self.tune_hp.log_to_file and not self.tune_hp.ray_local_mode,
            keep_checkpoints_num=2,
            checkpoint_score_attr=f'{self.search_params.opt.search_mode}-{self.search_params.opt.search_metric}',
            fail_fast=False,
            scheduler=self.get_tune_scheduler(self.search_params, self.tune_hp),
            verbose=2,
            progress_reporter=self.get_cli_reporter(),
            reuse_actors=False,
        )

        utils.hprint("done with tune.run")

        param_hash = self.search_params.get_short_hash(num_chars=8)
        analysis_file = self.search_params.logs.output_dir / f'tune_analysis_{param_hash}.cloudpickle'
        print(f"Saving {analysis_file}")
        utils.save_cloudpickle(analysis_file, analysis)

        best_trial = analysis.get_best_trial(
            self.search_params.opt.search_metric,
            self.search_params.opt.search_mode,
            "last-5-avg"
        )
        utils.hprint('best_trial.last_result', do_include_pre_break_line=True)
        utils.print_dict(best_trial.last_result)

        utils.hprint('best_trial.config', do_include_pre_break_line=True)
        utils.print_dict(best_trial.config)


if __name__ == '__main__':
    fast_dev_run = False

    dp = data_module.DataParams(
        # dataset_name='num=100_057b',
        dataset_name='num=1000_2cfc',
    )
    hp = ts_model.TotalParams(
        data=dp,
        max_seq_len=8192,
        batch_size=params.Discrete([4, 8, 16, 32, 64, 128, 256]),
    )

    hp.data.do_ignore_cached_dataset = False
    hp.data.seed = 42
    hp.data.num_workers = 4
    hp.data.num_gpus = 1
    hp.data.num_cpus = 4

    hp.opt.search_metric = 'valid/loss'
    hp.opt.search_mode = 'min'
    hp.opt.num_epochs = 4
    hp.opt.lr = params.Discrete([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    hp.opt.patience = 10

    hp.logs.num_steps_per_histogram_log = 20
    hp.logs.num_steps_per_metric_log = 5
    hp.logs.output_dir = constants.OUTPUT_DIR
    hp.logs.log_gpu_memory = 'all'

    hp.exp.project_name = 'tablestakes'
    hp.exp.experiment_name = 'korv_which'
    hp.exp.experiment_tags = ['korv_which']
    hp.exp.sources_glob_str = constants.THIS_DIR.parent.parent / '**/*.py'
    hp.exp.offline_mode = False

    base_dim = 15
    hp.embed.dim = params.Discrete([base_dim + 16, base_dim + 32, base_dim + 64, base_dim + 128, base_dim + 256])
    hp.embed.requires_grad = True
    hp.embed.position_embedding_requires_grad = False

    hp.conv.num_features = params.Discrete([32, 64, 128, 256])
    hp.conv.num_layers = params.Integer(min_value=1, max_value=3)
    hp.conv.kernel_size = 3
    hp.conv.num_groups = params.Discrete([16, 32, 64])
    hp.conv.num_blocks_per_pool = 20
    hp.conv.requires_grad = True

    hp.trans.impl = 'fast-favor'
    # neck_hp.trans.impl = 'fast'
    hp.trans.num_heads = params.Discrete([2, 4, 8, 16])
    hp.trans.num_layers = params.Integer(min_value=1, max_value=3)
    hp.trans.num_query_features = None
    hp.trans.fc_dim_mult = params.Integer(min_value=2, max_value=5)
    hp.trans.p_dropout = params.Float(min_value=0.05, max_value=0.2)
    hp.trans.p_attention_dropout = params.Float(min_value=0.05, max_value=0.2)

    hp.fc.num_features = params.Discrete([32, 64, 128, 256])
    hp.fc.num_layers = params.Integer(min_value=1, max_value=3)
    hp.fc.num_groups = params.Discrete([16, 32, 64])
    hp.fc.num_blocks_per_residual = params.Integer(min_value=1, max_value=5)
    hp.fc.num_blocks_per_dropout = params.Integer(min_value=1, max_value=5)
    hp.fc.requires_grad = True

    hp.neck.num_features = params.Discrete([32, 64, 128, 256])
    hp.neck.num_layers = params.Integer(min_value=1, max_value=3)
    hp.neck.num_groups = params.Discrete([16, 32, 64])
    hp.neck.num_blocks_per_residual = params.Integer(min_value=1, max_value=5)
    hp.neck.num_blocks_per_dropout = params.Integer(min_value=1, max_value=5)
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

    tune_hp = TuneParams()
    tune_hp.asha_grace_period = 4
    tune_hp.asha_reduction_factor = 2
    tune_hp.num_hp_samples = 2
    tune_hp.log_to_file = False
    tune_hp.ray_local_mode = False

    hostname = socket.gethostname()
    is_local_run = hostname.endswith('.local')

    utils.hprint('About to start model run:')
    utils.print_dict(hp.to_dict())

    tune_runner = TuneRunner(
        model_hp=hp,
        tune_hp=tune_hp,
        factored_lightning_module_class=ts_model.TablestakesBertConvTransTClassModel,
        extra_pl_callbacks=None,
    )
    tune_runner.run(
        fast_dev_run=False,
        use_gpus=not is_local_run,
    )

    utils.hprint("done with tune_runner.run")
