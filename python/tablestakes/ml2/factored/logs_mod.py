from argparse import Namespace
import os
import time
from collections import defaultdict
from typing import *

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.metrics.utils import _input_format_classification
from pytorch_lightning import loggers as pl_loggers

from ray import tune

from chillpill import params

from tablestakes import constants, utils
from tablestakes.ml2.data import datapoints
from tablestakes.ml2.data import data_module

CURRENT_EPOCH_NAME = 'current_epoch'
PARAM_COUNT_NAME = 'param_count'
TRAINABLE_PARAM_COUNT_NAME = 'trainable_param_count'
TIME_PERF_NAME = 'train_time_perf'
TIME_PROCESS_NAME = 'train_time_process'


class LoggingParams(params.ParameterSet):
    num_steps_per_histogram_log: int = 10
    num_steps_per_metric_log: int = 10
    output_dir: str = 'output'
    # Nona, 'all', or 'min_max'
    log_gpu_memory: Optional[str] = None


class TuneLogCopierCallback(pl.Callback):
    def __init__(self, filename='checkpoint'):
        super().__init__()
        self._filename = filename

    def _checkpoint(self, trainer):
        if trainer.running_sanity_check:
            return
        with tune.checkpoint_dir(step=trainer.global_step) as checkpoint_dir:
            trainer.save_checkpoint(
                os.path.join(checkpoint_dir, self._filename))

    @staticmethod
    def _get_metrics_dict(trainer, pl_module):
        d = trainer.logged_metrics
        d.update(trainer.callback_metrics)
        d.update(trainer.progress_bar_metrics)

        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.item()
        return d

    def _inner(self, trainer, pl_module):
        d = self._get_metrics_dict(trainer, pl_module)
        d[CURRENT_EPOCH_NAME] = trainer.current_epoch
        pid = os.getpid()
        d['pid'] = pid
        tune.report(**d)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        self._inner(trainer, pl_module)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        self._checkpoint(trainer)
        self._inner(trainer, pl_module)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        self._inner(trainer, pl_module)


class VocabLengthCallback(pl.Callback):
    VOCAB_PAD_VALUE = utils.VOCAB_PAD_VALUE

    def __init__(self, doc_dim=1):
        self._valid_counts = []
        self._invalid_counts = []

        self.doc_dim = doc_dim

    def _reset(self):
        self._valid_counts = []
        self._invalid_counts = []

    def on_train_epoch_start(self, trainer, pl_module):
        self._reset()

    def on_train_batch_start(self, trainer, pl_module: pl.LightningModule, batch: datapoints.XYMetaDatapoint, batch_idx, dataloader_idx):
        vocab = batch.x.vocab

        is_invalid = (vocab == self.VOCAB_PAD_VALUE).int()
        is_valid = 1 - is_invalid

        valid_counts = is_valid.sum(dim=self.doc_dim)
        invalid_counts = is_invalid.sum(dim=self.doc_dim)
        self._valid_counts.append(valid_counts)
        self._invalid_counts.append(invalid_counts)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        valid_counts = torch.cat(self._valid_counts)
        invalid_counts = torch.cat(self._invalid_counts)

        batch_lens = valid_counts + invalid_counts
        num_total = batch_lens.sum()

        p_valids = valid_counts / batch_lens

        trainer.logger.log_metrics(
            metrics={
                'p_tokens_is_valid': valid_counts.sum() / num_total,
                'num_tokens_total': num_total,

                'p_valids_min': p_valids.min(),
                'p_valids_median': p_valids.median(),
                'p_valids_max': p_valids.max(),

                'min_num_valid_tokens_per_doc': valid_counts.min(),
                'med_num_valid_tokens_per_doc': valid_counts.median(),
                'max_num_valid_tokens_per_doc': valid_counts.max(),

                'min_num_wasted_tokens_per_doc': invalid_counts.min(),
                'med_num_wasted_tokens_per_doc': invalid_counts.median(),
                'max_num_wasted_tokens_per_doc': invalid_counts.max(),
            },
            step=trainer.global_step,
        )


class PredictionSaver(pl.Callback):
    def __init__(
            self,
            weighted_head_params,
            dir_suffixes_to_keep: Iterable[str] = (),
            p_keep: float = 0.05,
    ):
        super().__init__()
        self.weighted_head_params = weighted_head_params
        self.meta_vals_to_keep = dir_suffixes_to_keep
        self.p_keep = p_keep
        self.dir_to_head_outputs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.field_name_to_y_col_names = {}

    def _get_meta_vals_to_keep(self, pl_module):
        kept_dirs = []
        for batch in pl_module.dm.val_dataloader():
            datapoint_dirs = batch.meta.datapoint_dir
            do_keep = np.random.rand(len(datapoint_dirs)) < self.p_keep
            keep_dirs = list(np.array(datapoint_dirs)[do_keep])
            kept_dirs.extend(keep_dirs)
        return kept_dirs

    def on_sanity_check_start(self, trainer, pl_module):
        self.field_name_to_y_col_names = pl_module.dm.get_field_name_to_y_col_names()
        if not self.meta_vals_to_keep:
            self.meta_vals_to_keep = self._get_meta_vals_to_keep(pl_module)

    def on_validation_batch_end(
            self,
            trainer,
            pl_module,
            y_hats_for_pred,
            batch: datapoints.XYMetaDatapoint,
            batch_idx,
            dataloader_idx,
    ):
        self._inner(batch, pl_module)

    def _inner(self, batch, pl_module):
        do_keep = []
        datapoint_dirs = batch.meta.datapoint_dir

        # yes, yes, terribly slow
        for dd in datapoint_dirs:
            do_keep.append(any([dd.endswith(mv) for mv in self.meta_vals_to_keep]))

        _, y_hats_for_pred = pl_module(batch.x)
        y_hats_for_pred = {k: v[do_keep] for k, v in y_hats_for_pred.items()}
        ys = {}

        keep_inds = np.where(do_keep)[0]

        for k, v in batch.y:
            if isinstance(v, list):
                v = [v[ki] for ki in keep_inds]
            elif isinstance(v, torch.Tensor):
                v = list(v.cpu().numpy().copy()[do_keep])
            ys[k] = v
        metas_kept = list(np.array(batch.meta.datapoint_dir)[do_keep])

        for meta_idx, meta_kept in enumerate(metas_kept):
            for head_name, y_hat_per_head in y_hats_for_pred.items():
                y_field_name = self.weighted_head_params.head_params[head_name].get_y_field_name()
                try:
                    y_per_head = ys[y_field_name]
                except KeyError as e:
                    print(f'PredictionSaver._inner ys: {ys}')
                    raise e
                y = y_per_head[meta_idx].copy()
                y_hat = y_hat_per_head[meta_idx].cpu().numpy().copy()
                self.dir_to_head_outputs[meta_kept][head_name]['y'] = y
                self.dir_to_head_outputs[meta_kept][head_name]['y_hats'].append(y_hat)

    def get_df_dict(self, do_sanitize_filenames=True):
        d = {}
        for datapoint_dir, heads in self.dir_to_head_outputs.items():
            if do_sanitize_filenames:
                datapoint_dir = '-'.join(datapoint_dir.split(os.path.sep)[-2:])
            dd = {}
            for head_name, head_outs in heads.items():
                y_field_name = self.weighted_head_params.get_y_field_name_from_head_name(head_name)
                df = pd.DataFrame(
                    [head_outs['y']] + head_outs['y_hats'],
                    columns=self.field_name_to_y_col_names[y_field_name]
                )
                dd[head_name] = df
            d[datapoint_dir] = dd
        return d

    def on_validation_epoch_end(self, trainer, pl_module):
        d = self.get_df_dict(do_sanitize_filenames=True)
        d = {'preds': d}
        pl_module.log_lossmetrics_dict(phase=utils.Phase.valid, d=d)

    def print_preds(self):
        for datapoint_dir_name, head_dfs in self.get_df_dict().items():
            print()
            print(datapoint_dir_name)
            for head_name, df in head_dfs.items():
                print(f'  {head_name}')
                print()
                s = str(df)
                s = s.replace('\n', '\n    ')
                print(f'    {s}')


class CounterTimerLrCallback(pl.Callback):
    def __init__(self):
        self._train_start_perf = None
        self._train_start_process = None

    @staticmethod
    def _count_params(pl_module):
        return sum(p.numel() for p in pl_module.parameters())

    @staticmethod
    def _count_trainable_params(pl_module):
        return sum(p.numel() for p in pl_module.parameters() if p.requires_grad)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, ):
        if trainer.logger is None:
            return

        d = {
            PARAM_COUNT_NAME: self._count_params(pl_module),
            TRAINABLE_PARAM_COUNT_NAME: self._count_trainable_params(pl_module),
            'pid': os.getpid(),
        }

        # noinspection PyTypeChecker
        trainer.logger.log_hyperparams(params=d)

    def _get_lr_dict(self, pl_module) -> Dict[str, float]:
        lrs = [float(param_group['lr']) for param_group in pl_module.optimizers().param_groups]
        assert (all([lr == lrs[0] for lr in lrs]))
        lr = lrs[0]

        return {
            'lrs_opt': lr,
        }

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        d = self._get_lr_dict(pl_module)
        trainer.logger.log_metrics(d, step=trainer.global_step)

        self._train_start_perf = time.perf_counter()
        self._train_start_process = time.process_time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        if trainer.logger is None:
            return

        d = self._get_lr_dict(pl_module)
        d[TIME_PERF_NAME] = time.process_time() - self._train_start_process
        d[TIME_PROCESS_NAME] = time.process_time() - self._train_start_process
        d['pid'] = os.getpid()

        trainer.logger.log_metrics(d, step=trainer.global_step)


class ClassCounterCallback(pl.Callback):
    """Count the number of datapoints belonging to each class.

    Right now, designed for binary multi-feild problems (i.e.: `SigmoidHead`s)
    """
    PORTIONS = 'portions'
    INV_PORTIONS = '1 / portions'
    CLASS_COUNTS = 'class_counts'

    def __init__(
            self,
            field_names: List[str],
            head_params: Optional[params.ParameterSet] = None,
            do_update_pos_class_weights=True,
            max_pos_class_weight=10.0,
            verbose=True,
    ):
        super().__init__()
        self.field_names = field_names
        self.verbose = verbose
        self.hp = head_params
        assert self.hp.type == 'weighted', f'bad hp type: {self.hp.type}'
        self.field_name_to_head_name = self.hp.get_field_name_to_head_name()

        self.do_update_pos_class_weights = do_update_pos_class_weights
        self.max_pos_class_weight = max_pos_class_weight

    def _y_list_to_df(self, head_name: str, y: List[torch.Tensor]) -> pd.DataFrame:
        y = torch.cat(y, dim=0)
        num_datapoints = len(y)
        counts = y.sum(dim=0).detach().numpy()
        if head_name in self.hp.head_params.keys():
            cols = list(self.hp.head_params[head_name].class_name_to_weight.keys()) or None
            assert len(cols) == len(counts)
        else:
            cols = None

        invs = [1.0 if not c else num_datapoints / c for c in counts]
        return pd.DataFrame(
            data=[counts, counts / num_datapoints, invs],
            columns=cols,
            index=['counts', self.PORTIONS, self.INV_PORTIONS],
        )

    def get_field_to_class_counts(self, dataloader: DataLoader):
        field_name_to_y_lists = defaultdict(list)
        for batch in dataloader:
            for field_name in self.field_names:
                field_name_to_y_lists[field_name].extend([batch.y[field_name]])

        field_to_class_counts = {
            field_name: self._y_list_to_df(field_name, y_list)
            for field_name, y_list in field_name_to_y_lists.items()
        }
        return field_to_class_counts

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        field_to_class_counts = self.get_field_to_class_counts(dataloader=pl_module.train_dataloader())
        if self.verbose:
            utils.hprint('ClassCounterCallback Class Counts:')
            utils.print_dict(field_to_class_counts)
            print()

        if self.hp is None:
            if self.verbose:
                print(f'  Not setting head_params.pos_class_weights because you did not pass hp to my init')
        else:
            if self.hp.type != 'weighted':
                raise NotImplementedError(
                    f'hp == {self.hp} but this is only implemented for WeightedHeadParams'
                )
            for field_name, class_counts_df in field_to_class_counts.items():
                if field_name not in self.field_name_to_head_name:
                    # we might not be using all fields in heads
                    continue
                head_name = self.field_name_to_head_name[field_name]
                head = pl_module.head.heads[head_name]

                if head.did_set_pos_class_weights:
                    pos_class_weights = head.pos_class_weights
                    if self.verbose:
                        weights_str = ', '.join([f'{e:.2f}' for e in pos_class_weights])
                        print(f'  head_params["{field_name}"].pos_class_weights was already set to [{weights_str}]')
                        print()

                else:
                    pos_class_weights = class_counts_df.loc[self.INV_PORTIONS].values
                    max_inds = np.where(pos_class_weights > self.max_pos_class_weight)[0]
                    pos_class_weights[max_inds] = self.max_pos_class_weight

                    head.set_pos_class_weights(torch.tensor(pos_class_weights, dtype=torch.float, device=pl_module.device))
                    if self.verbose:
                        weights_str = ', '.join([f'{e:.2f}' for e in pos_class_weights])
                        print(f'  Setting head_params["{field_name}"].pos_class_weights = [{weights_str}]')
                        print()

        pl_module.log_lossmetrics_dict(
            phase=utils.Phase.train,
            d={self.CLASS_COUNTS: field_to_class_counts},
            do_log_to_progbar=False,
        )


class BetterAccuracy(pl.metrics.Accuracy):
    # TODO: add class weights to init; report both
    """Like Accuracy, but better.
        - PyTorch Lightning's += lines cause warnings about transferring lots of scalars between cpu / gpu
          but " = ... +" lines are fine
        - Respect Y_VALUE_TO_IGNORE
    """
    Y_VALUE_TO_IGNORE = constants.Y_VALUE_TO_IGNORE
    # def __init__(self, class_name_to_weight: OrderedDict[str, float], print_every: Optional[int] = 100):
    def __init__(self, print_every: Optional[int] = 100):
        super().__init__()

        self.print_every = print_every
        self.counter = 1

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        do_print = self.print_every is not None and not self.counter % self.print_every
        if do_print:
            utils.hprint(f'BetterAccuracy is set to print every {self.print_every} and you at {self.counter}:')
            print(f"BetterAccuracy: preds: \n{preds}")
            print(f"BetterAccuracy: target: \n{target}")
            print()
        self.counter += 1

        # preds, target = _input_format_classification(preds, target, self.threshold)
        assert preds.shape == target.shape, f'preds.shape = {preds.shape} != target.shape = {target.shape}'

        preds = preds.argmax(dim=1)
        target = target.argmax(dim=1)

        if do_print:
            print(f"BetterAccuracy: preds post argmax: \n{preds}")
            print(f"BetterAccuracy: target post argmax: \n{target}")
            print()

        assert target.dim() == 1, f'got target of shape {target.shape}'

        eqs = preds.eq(target)

        if do_print:
            print(
                f"BetterAccuracy: new_correct: {eqs.sum()}, "
                f" numel: {target.numel()}, "
                f" shape[0]: {target.shape[0]}, "
                f" ignore: {target.eq(self.Y_VALUE_TO_IGNORE).sum()}"
            )

        self.correct = self.correct + torch.sum(eqs)
        self.total = self.total + target.shape[0]


class WeightedBetterAccuracy(pl.metrics.Accuracy):
    Y_VALUE_TO_IGNORE = constants.Y_VALUE_TO_IGNORE

    def __init__(self, class_name_to_weight: OrderedDict[str, float], print_every: Optional[int] = 100):
        super().__init__()
        self.register_buffer('class_weights', torch.tensor(list(class_name_to_weight.values()), dtype=torch.float))

        self.print_every = print_every
        self.counter = 1

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        do_print = self.print_every is not None and not self.counter % self.print_every
        if do_print:
            utils.hprint(f'WeightedBetterAccuracy is set to print every {self.print_every} and you at {self.counter}:')
            print(f"WeightedBetterAccuracy: preds: \n{preds}")
            print(f"WeightedBetterAccuracy: target: \n{target}")
            print()
        self.counter += 1

        # preds, target = _input_format_classification(preds, target, self.threshold)
        assert preds.shape == target.shape, f'preds.shape = {preds.shape} != target.shape = {target.shape}'

        preds = preds.argmax(dim=1)
        target = target.argmax(dim=1)

        if do_print:
            print(f"WeightedBetterAccuracy: preds post argmax: \n{preds}")
            print(f"WeightedBetterAccuracy: target post argmax: \n{target}")
            print()

        assert target.dim() == 1, f'got target of shape {target.shape}'

        eqs = preds.eq(target)

        batch_weights = self.class_weights[target]

        if do_print:
            print(
                f"WeightedBetterAccuracy:"
                f" batch_weights: {batch_weights}, "
                f" eqs * batch_w: {eqs * batch_weights}, "
                f" new_correct: {torch.sum(eqs * batch_weights)}, "
                f" numel: {target.numel()}, "
                f" shape[0]: {target.shape[0]}, "
                f" ignore: {target.eq(self.Y_VALUE_TO_IGNORE).sum()}"
            )

        self.correct = self.correct + torch.sum(eqs * batch_weights)
        self.total = self.total + batch_weights.sum() - target.eq(self.Y_VALUE_TO_IGNORE).sum()



class SigmoidBetterAccuracy(pl.metrics.Accuracy):
    Y_VALUE_TO_IGNORE = constants.Y_VALUE_TO_IGNORE
    THRESHOLD = 0.5

    def __init__(self):
        super().__init__(
            threshold=0.5,
            compute_on_step=True,
            dist_sync_on_step=False,
            process_group=None,
            dist_sync_fn=None,
        )
        # self.s = nn.Sigmoid()

    """PyTorch Lightning's += lines cause warnings about transferring lots of scalars between cpu / gpu"""

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds = self.s(preds)
        preds = 1. * (preds > self.THRESHOLD)
        # preds, target = _input_format_classification(preds, target, self.threshold)
        # assert preds.shape == target.shape, f"preds.shape: {preds.shape}, target.shape: {target.shape}"

        # print()
        # print(f"SigmoidBetterAccuracy: new_correct: {torch.sum(preds.eq(target))}, numel: {target.numel()}, ignore: {target.eq(self.Y_VALUE_TO_IGNORE).sum()}")
        # print(f"SigmoidBetterAccuracy: preds: \n  {preds}")
        # print(f"SigmoidBetterAccuracy: target: \n  {target}")
        try:
            self.correct = self.correct + torch.sum(preds.eq(target))
        except BaseException as e:
            print('preds:  ', preds, preds.shape)
            print('target: ', target, target.shape)
            raise e
        self.total = self.total + target.numel() - target.eq(self.Y_VALUE_TO_IGNORE).sum()


class ExperimentParams(params.ParameterSet):
    project_name: str = 'my_project'
    experiment_name: str = 'my_experiment'
    experiment_tags: List[str] = ['testing', ]
    sources_glob_str: str = '*.py'
    offline_mode: bool = False

    def get_project_exp_name(self):
        return f'{self.project_name}-{self.experiment_name}'


# noinspection PyProtectedMember
class MyLightningNeptuneLogger(pl_loggers.NeptuneLogger):
    def __init__(self, hp: ExperimentParams, version: str = '', offline_mode=False):
        self.offline_mode = offline_mode

        super().__init__(
            api_key=utils.get_logger_api_key(),
            project_name=utils.get_neptune_fully_qualified_project_name(hp.project_name),
            close_after_fit=True,
            offline_mode=offline_mode,
            experiment_name=f'pl_log-{version}',
            params=hp.to_dict(),
            tags=hp.experiment_tags,
            upload_source_files=str(hp.sources_glob_str),
        )
        self.append_tags(hp.experiment_tags)

    @pl.utilities.rank_zero_only
    def log_metrics(
            self,
            metrics: Dict[str, Union[torch.Tensor, float, pd.DataFrame]],
            step: Optional[int] = None
    ) -> None:
        assert pl.utilities.rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'
        for key, val in metrics.items():
            if isinstance(val, pd.DataFrame):
                self.log_df(key, val)
            else:
                self.log_metric(key, val, step=step)

    def log_df(self, name: str, df: pd.DataFrame):
        from neptunecontrib.api.table import log_table
        log_table(name, df, self.experiment)

    @pl.utilities.rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        params = self._sanitize_params(params)
        return self.set_properties(params)

    def set_properties(self, new_properties: Dict):
        if self.offline_mode:
            import warnings
            warnings.warn('log_mods.MyLightningNeptuneLogger is in offline mode and is skipping set_properties')
            return
        else:
            properties = self.experiment._backend.get_experiment(self.experiment.internal_id).properties
            properties = {p.key: p.value for p in properties}
            properties.update({k: str(v) for k, v in new_properties.items()})
            return self.experiment._backend.update_experiment(
                experiment=self.experiment,
                properties=properties,
            )


def get_pl_logger(hp: ExperimentParams, tune=None):
    version = 'local' if tune is None else tune.get_trial_id()
    logger = MyLightningNeptuneLogger(hp=hp, version=version, offline_mode=hp.offline_mode),

    return logger


