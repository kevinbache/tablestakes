import abc
from pathlib import Path
from typing import *

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from tablestakes import utils, constants
from tablestakes.ml2.data import datapoints
from tablestakes.ml2 import data

from chillpill import params


class DataParams(params.ParameterSet):
    dataset_name: utils.DirtyPath
    docs_root_dir: utils.DirtyPath = constants.DOCS_DIR
    dataset_root_dir: utils.DirtyPath = constants.DATASETS_DIR
    do_ignore_cached_dataset: bool = False

    max_seq_len = 4096
    batch_size = 32

    p_valid = 0.1
    p_test = 0.1
    num_workers = 4

    num_gpus = 0
    num_cpus = 4

    seed = 42

    def __init__(self, dataset_name='DUMMY_DATASET_NAME', **kwargs):
        self.dataset_name = dataset_name
        super().__init__(**kwargs)

    def __post_init__(self):
        self.docs_root_dir = Path(self.docs_root_dir)
        self.dataset_root_dir = Path(self.dataset_root_dir)

        self.dataset_file = self.dataset_root_dir / f'{self.dataset_name}.cloudpickle'
        self.docs_dir = self.docs_root_dir / self.dataset_name


class XYMetaHandlerDatasetModule(pl.LightningDataModule):
    def __init__(self, hp: DataParams, verbose=False):
        super().__init__()
        self.hp = hp
        self.verbose = verbose

        self.ds = self.get_dataset(hp)
        assert isinstance(self.ds, data.XYMetaDirHandlerDataset), f'{type(self.ds)}, hp: {hp}'
        dims = self.ds.get_num_features()

        self.num_x_base_dims = dims.x.base
        self.num_y_classes: datapoints.YDatapoint = dims.y

        self.hp.num_x_dims = dims.x
        self.hp.num_y_classes = dims.y

        self.example_input_array = self.get_example_input_array()

        self.train_dataset, self.valid_dataset, self.test_dataset = None, None, None

    @abc.abstractmethod
    def get_example_input_array(self) -> datapoints.XDatapoint:
        pass

    @abc.abstractmethod
    def get_dataset(self, hp: DataParams) -> data.XYMetaDirHandlerDataset:
        pass

    def get_y_col_names(self):
        return {name: df.columns for name, df in self.ds.datapoints[0].y}

    def setup(self, stage: Optional[str] = None):
        # called on one gpu
        self.hp.num_data_total = len(self.ds)
        self.hp.num_data_test = int(self.hp.num_data_total * self.hp.p_test)
        self.hp.num_data_valid = int(self.hp.num_data_total * self.hp.p_valid)
        self.hp.num_data_train = self.hp.num_data_total - self.hp.num_data_test - self.hp.num_data_valid

        self.train_dataset, self.valid_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset=self.ds,
            lengths=[self.hp.num_data_train, self.hp.num_data_valid, self.hp.num_data_test],
            generator=torch.Generator().manual_seed(self.hp.seed),
        )

        if self.verbose:
            print(f'module setup ds lens: '
                  f'{len(self.train_dataset)}, {len(self.valid_dataset)}, {len(self.test_dataset)}')

    def _collate_fn(self, batch: List[datapoints.XYMetaDatapoint]) -> datapoints.XYMetaDatapoint:
        return datapoints.XYMetaDatapoint.collate(batch, max_seq_len=self.hp.max_seq_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hp.batch_size,
            shuffle=True,
            num_workers=self.hp.num_workers,
            collate_fn=lambda x: self._collate_fn(x),
            pin_memory=self.hp.num_gpus > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hp.batch_size,
            shuffle=False,
            num_workers=self.hp.num_workers,
            collate_fn=lambda x: self._collate_fn(x),
            pin_memory=self.hp.num_gpus > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hp.batch_size,
            shuffle=False,
            num_workers=self.hp.num_workers,
            collate_fn=lambda x: self._collate_fn(x),
            pin_memory=self.hp.num_gpus > 0,
        )

    def transfer_batch_to_device(
            self,
            batch: datapoints.XYMetaDatapoint,
            device: torch.device,
    ) -> datapoints.XYMetaDatapoint:
        return batch.transfer_to_device(device)

    def prepare_data(self, *args, **kwargs):
        pass

