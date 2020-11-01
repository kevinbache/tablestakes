import argparse
import os
import socket
from typing import *

import boto3

import pytorch_lightning as pl

import ray
import tablestakes.ml.metrics_mod
from ray import tune
from ray.tune import logger as tune_logger
from ray.tune.integration import pytorch_lightning as tune_pl

from chillpill import params

from tablestakes import constants, utils
from tablestakes.ml import metrics_mod, data, torch_mod, factored, model_bertenc_conv_tclass


REGION = 'us-west-2'


class TuneRunner(torch_mod.Parametrized['factored.FactoredLightningModule.FactoredParams']):
    class TuneParams(params.ParameterSet):
        asha_grace_period = 4
        asha_reduction_factor = 2
        num_hp_samples = 10

    def __init__(
            self,
            search_params: "TuneFactoredParams",
            factored_lightning_module_class: type,
            extra_pl_callbacks: Optional[List[pl.callbacks.Callback]] = None,
            ray_local_mode=False,
    ):

        self.include_gpus = None

        os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = "1"

        self.ensure_hp_is_factored(search_params)
        super().__init__(search_params)
        self.search_params = self.hp
        assert hasattr(self.search_params, 'tune')
        # noinspection PyUnresolvedReferences
        assert isinstance(self.search_params.tune, self.TuneParams)

        self._param_class = search_params.__class__

        assert issubclass(factored_lightning_module_class, factored.FactoredLightningModule)
        self._factored_lightning_module_class = factored_lightning_module_class

        self._metrics_tracker_class = self._factored_lightning_module_class.get_metrics_tracker_class()

        if extra_pl_callbacks is None:
            extra_pl_callbacks = [
                metrics_mod.CounterTimerCallback(),
                metrics_mod.LogCopierCallback(),
            ]
        self.extra_pl_callbacks = extra_pl_callbacks

        self.create_bucket_if_not_exist(search_params.metrics.output_dir)

        # True if running on macbook laptop, False if running on cluster
        is_running_on_local_machine = self.add_local_cluster_tag(search_params)

        ################
        #   init ray   #
        ################
        print("about to init ray")
        ray.init(
            address=None if is_running_on_local_machine else 'auto',
            ignore_reinit_error=True,
            include_dashboard=True,
            local_mode=ray_local_mode,
        )
        print("Done with ray init")

    @staticmethod
    def add_local_cluster_tag(search_params: "TuneFactoredParams"):
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
    def ensure_hp_is_factored(hp: "TuneFactoredParams"):
        assert isinstance(hp, factored.FactoredLightningModule.FactoredParams)
        assert isinstance(hp.data, data.TablestakesDataModule.DataParams)
        assert isinstance(hp.opt, factored.OptimizersMaker.OptParams)
        assert isinstance(hp.metrics, tablestakes.ml.metrics_mod.MetricsTracker.MetricParams)
        assert isinstance(hp.exp, torch_mod.ExperimentParams)
        assert isinstance(hp.tune, TuneRunner.TuneParams)

    def get_tune_callbacks(self):
        return [
            tune_pl.TuneReportCallback(
                metrics=self._metrics_tracker_class.get_all_metric_names_for_phase(constants.TRAIN_PHASE_NAME),
                on='train_end',
            ),
            tune_pl.TuneReportCheckpointCallback(
                metrics=self._metrics_tracker_class.get_all_metric_names_for_phase(constants.VALID_PHASE_NAME),
                filename=constants.CHECKPOINT_FILE_BASENAME,
                on='validation_end',
            ),
            tune_pl.TuneReportCallback(
                metrics=self._metrics_tracker_class.get_all_metric_names_for_phase(constants.TEST_PHASE_NAME),
                on='test_end',
            ),
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

        del config['tune']
        hp = self._param_class.from_dict(config)
        assert isinstance(hp, self._param_class)

        if checkpoint_dir:
            # see https://docs.ray.io/en/master/tune/user-guide.html#checkpointing
            raise NotImplementedError(f"Got checkpoint_dir in trian_fn: {checkpoint_dir}")

        net = self._factored_lightning_module_class.from_hp(hp=hp)

        utils.set_seeds(hp.data.seed)

        # noinspection PyTypeChecker
        trainer = pl.Trainer(
            logger=torch_mod.get_pl_logger(hp=hp.exp, tune=tune, offline_mode=fast_dev_run),
            default_root_dir=tune.get_trial_dir(),
            callbacks=self.extra_pl_callbacks + self.get_tune_callbacks(),
            max_epochs=hp.opt.num_epochs,
            gpus=hp.data.num_gpus if include_gpus else None,
            weights_summary='full',
            fast_dev_run=fast_dev_run,
            accumulate_grad_batches=1,
            profiler=True,
            deterministic=True,
            log_every_n_steps=hp.metrics.num_steps_per_metric_log,
        )
        fit_out = trainer.fit(net, datamodule=net.dm)

        utils.print_dict(config)
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
            if metrics_mod.CURRENT_EPOCH_NAME in result:
                return result[metrics_mod.CURRENT_EPOCH_NAME] > num_epochs
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
            metrics_mod.PARAM_COUNT_NAME,
            metrics_mod.CURRENT_EPOCH_NAME,
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
            max_progress_rows=self.search_params.tune.num_hp_samples,
            parameter_columns=self.search_params.get_samplable_param_names(),
            metric_columns=metric_cols + extra_metric_cols,
        )

    @staticmethod
    def get_tune_scheduler(search_params: "TuneFactoredParams") -> tune.schedulers.TrialScheduler:
        return tune.schedulers.ASHAScheduler(
            metric=search_params.opt.search_metric,
            mode=search_params.opt.search_mode,
            grace_period=search_params.tune.asha_grace_period,
            reduction_factor=search_params.tune.asha_reduction_factor,
        )

    def run(self, fast_dev_run=False, use_gpus=False, log_to_file=False):
        search_dict = self.search_params.to_ray_tune_search_dict()
        # see tune.utils.UtilMonitor
        search_dict['log_sys_usage'] = True

        # noinspection PyTypeChecker
        analysis = tune.run(
            run_or_experiment=self._get_train_fn(fast_dev_run=fast_dev_run, include_gpus=use_gpus),
            name=self.search_params.exp.get_project_exp_name(),
            stop=self.get_tune_stopper(self.search_params.opt.num_epochs),
            config=search_dict,
            resources_per_trial=self.get_resources_per_trial(self.search_params, include_gpu=use_gpus),
            num_samples=self.search_params.tune.num_hp_samples,
            sync_config=tune.SyncConfig(upload_dir=self.search_params.metrics.output_dir),
            loggers=self.get_tune_loggers(),
            log_to_file=log_to_file,
            keep_checkpoints_num=2,
            checkpoint_score_attr=f'{self.search_params.opt.search_mode}-{self.search_params.opt.search_metric}',
            fail_fast=False,
            scheduler=self.get_tune_scheduler(self.search_params),
            verbose=2,
            progress_reporter=self.get_cli_reporter(),
            reuse_actors=False,
        )

        utils.hprint("done with tune.run")

        param_hash = self.search_params.get_short_hash(num_chars=8)
        analysis_file = self.search_params.metrics.output_dir / f'tune_analysis_{param_hash}.pkl'
        print(f"Saving {analysis_file}")
        utils.save_pickle(analysis_file, analysis)

        best_trial = analysis.get_best_trial(
            self.search_params.opt.search_metric,
            self.search_params.opt.search_mode,
            "last-5-avg"
        )
        print(f'best_trial.last_result: {best_trial.last_result}')
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final search_metric: {}".format(best_trial.last_result[self.search_params.opt.search_metric]))


class TuneFactoredParams(factored.FactoredLightningModule.FactoredParams):
    data = data.TablestakesDataModule.DataParams()
    opt = factored.OptimizersMaker.OptParams()
    metrics = tablestakes.ml.metrics_mod.MetricsTracker.MetricParams()
    exp = torch_mod.ExperimentParams()
    tune = TuneRunner.TuneParams()

    # noinspection PyTypeChecker
    @classmethod
    def from_factored_params(
            cls,
            hp: factored.FactoredLightningModule.FactoredParams,
            tune_params: TuneRunner.TuneParams = TuneRunner.TuneParams(),
    ) -> "TuneFactoredParams":
        # oh god this is dirty
        hp.tune = tune_params
        return hp


# noinspection DuplicatedCode
if __name__ == '__main__':
    hp = model_bertenc_conv_tclass.ModelBertEncConvTClass.Params(
        max_seq_len=2 ** 11,
        batch_size=32,
    )

    # hp.data.dataset_name = 'num=1000_02b7'
    hp.data.dataset_name = 'num=4000_9b9f'
    hp.data.do_ignore_cached_dataset = False
    hp.data.seed = 42
    hp.data.num_workers = 0
    hp.data.num_gpus = 1
    hp.data.num_cpus = 4

    hp.opt.search_metric = 'valid_loss_total'
    hp.opt.search_mode = 'min'
    hp.opt.num_epochs = 100
    hp.opt.lr = params.Discrete([1e-4, 1e-3, 1e-2, 1e-1])
    hp.opt.min_lr = 1e-6
    hp.opt.patience = 16

    hp.metrics.num_steps_per_histogram_log = 20
    hp.metrics.num_steps_per_metric_log = 5
    hp.metrics.output_dir = 's3://kb-tester-2020-10-30'

    hp.exp.project_name = 'tablestakes'
    hp.exp.experiment_name = 'korv_which'
    hp.exp.experiment_tags = ['korv_which', 'conv', 'sharp', 'search', 'v0.0.1']
    hp.exp.sources_glob_str = constants.THIS_DIR.parent.parent / '**/*.py'

    hp.embed.dim = 16
    hp.embed.requires_grad = True

    hp.heads.num_features = params.Discrete([32, 64, 128, 256])
    hp.conv.num_layers = params.Integer(2, 11)
    hp.conv.kernel_size = 3
    hp.conv.num_groups = params.Discrete([8, 16, 32, 64])
    hp.conv.num_blocks_per_pool = 20
    hp.conv.num_blocks_per_skip = 2
    hp.conv.requires_grad = True

    hp.heads.num_features = params.Discrete([32, 64, 128, 256])
    hp.fc.num_layers = params.Integer(2, 7)
    hp.fc.num_groups = params.Discrete([8, 16, 32, 64])
    hp.fc.num_blocks_per_residual = params.Integer(1, 5)
    hp.fc.num_blocks_per_dropout = params.Integer(1, 8)
    hp.fc.requires_grad = True

    hp.heads.num_features = params.Discrete([32, 64, 128, 256])
    hp.heads.num_layers = params.Integer(2, 5)
    hp.fc.num_groups = params.Discrete([8, 16, 32, 64])
    hp.heads.num_blocks_per_residual = params.Integer(1, 5)
    hp.heads.num_blocks_per_dropout = params.Integer(1, 5)
    hp.heads.requires_grad = True

    hp.tune = TuneRunner.TuneParams()
    hp.tune.asha_grace_period = 16
    hp.tune.asha_reduction_factor = 2
    hp.tune.num_hp_samples = 100

    hostname = socket.gethostname()
    is_local_run = hostname.endswith('.local')

    # noinspection PyTypeChecker
    tune_runner = TuneRunner(
        search_params=hp,
        factored_lightning_module_class=model_bertenc_conv_tclass.ModelBertEncConvTClass,
        extra_pl_callbacks=None,
        ray_local_mode=False,
    )
    tune_runner.run(
        fast_dev_run=False,
        use_gpus=not is_local_run,
        log_to_file=True,
    )

    utils.hprint("done with tune_runner.run")
