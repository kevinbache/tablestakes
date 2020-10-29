import abc
import collections
import glob
from pathlib import Path
import re
from typing import *

import numpy as np
import pandas as pd

import torch
from tablestakes.ml import param_torch_mods, factored
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from chillpill import params

from tablestakes import constants, utils, load_makers


X_PREFIX = constants.X_PREFIX
Y_PREFIX = constants.Y_PREFIX

Y_VALUE_TO_IGNORE = constants.Y_VALUE_TO_IGNORE


class XYCsvDataset(Dataset):
    """
    Expects data to look like this:

    docs_dir
        datapoint_dir_1
            x.csv
            x_2.csv
            y.csv
        datapoint_dir_2
            x.csv
            x_2.csv
            y.csv
    """
    DATAPOINT_DIR_NAME = '*'

    @classmethod
    def _find_files_matching_patterns(cls, data_dir: Union[Path, str], patterns: List[str]):
        return [sorted(glob.glob(str(data_dir / Path(cls.DATAPOINT_DIR_NAME) / Path(p)), recursive=True)) for p in
                patterns]

    @staticmethod
    def _read_dfs(files):
        return [pd.read_csv(f) for f in files]

    @classmethod
    def _find_files(cls, pattern):
        """get all files matching a glob search pattern group them into a list of dicts.
        each dict contains all the matching file names in a given parent directory
        each dict's keys
        """
        files = sorted(glob.glob(str(pattern), recursive=True))
        d = collections.defaultdict(dict)
        for this_file in files:
            this_file = Path(this_file)
            name = this_file.name
            name = name.replace('.csv', '')
            d[this_file.parent][name] = this_file
        out = d.values()
        return out

    @staticmethod
    def _read_csvs_from_dict(
            d: dict,
            key_to_base_name_fn: Callable[[str], str],
            df_postproc: Callable[[pd.DataFrame], pd.DataFrame],
    ):
        return {key_to_base_name_fn(k): df_postproc(pd.read_csv(v)) for k, v in d.items()}

    @staticmethod
    def _convert_dict_of_dfs_to_tensors(d: dict):
        try:
            return {k: torch.tensor(v.values) for k, v in d.items()}
        except BaseException as e:
            utils.print_dict(d)
            raise e

    def __init__(
            self,
            data_dir: Union[Path, str],
            x_pattern=Path('**') / f'{X_PREFIX}*.csv',
            y_pattern=Path('**') / f'{Y_PREFIX}*.csv',
            csv_filename_to_x_name: Optional[Callable[[str], str]] = None,
            csv_filename_to_y_name: Optional[Callable[[str], str]] = None,
            x_df_postproc: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
            y_df_postproc: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ):
        self.data_dir = str(Path(data_dir))
        self.x_pattern = x_pattern
        self.y_pattern = y_pattern

        drop_unnamed = lambda df: df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if x_df_postproc is None:
            x_df_postproc = drop_unnamed
        self._x_df_postproc = x_df_postproc

        if y_df_postproc is None:
            y_df_postproc = drop_unnamed
        self._y_df_postproc = y_df_postproc

        def _match_after_prefix(filename: str, prefix: str) -> str:
            m = re.match(fr'.*{prefix}(\w+)$', filename)
            return m.groups()[0]

        if csv_filename_to_x_name is None:
            csv_filename_to_x_name = lambda filename: _match_after_prefix(filename, X_PREFIX)
        self._csv_filename_to_x_name = csv_filename_to_x_name

        if csv_filename_to_y_name is None:
            csv_filename_to_y_name = lambda filename: _match_after_prefix(filename, Y_PREFIX)
        self._csv_filename_to_y_name = csv_filename_to_y_name

        self._x_filename_dicts = self._find_files(self.data_dir / self.x_pattern)
        self._y_filename_dicts = self._find_files(self.data_dir / self.y_pattern)
        if len(self._x_filename_dicts) == 0:
            raise ValueError(f"Found no files matching {self.data_dir / self.x_pattern}")
        if len(self._y_filename_dicts) == 0:
            raise ValueError(f"Found no files matching {self.data_dir / self.y_pattern}")

        df_dicts = []
        for x_filename_dict, y_filename_dict in zip(self._x_filename_dicts, self._y_filename_dicts):
            df_dicts.append((
                self._read_csvs_from_dict(
                    x_filename_dict,
                    key_to_base_name_fn=self._csv_filename_to_x_name,
                    df_postproc=self._x_df_postproc,
                ),
                self._read_csvs_from_dict(
                    y_filename_dict,
                    key_to_base_name_fn=self._csv_filename_to_y_name,
                    df_postproc=self._y_df_postproc,
                ),
            ))

        self._datapoints = []
        for x_dict, y_dict in df_dicts:
            self._datapoints.append((
                self._convert_dict_of_dfs_to_tensors(x_dict),
                self._convert_dict_of_dfs_to_tensors(y_dict),
            ))

        self.num_x_dims = {k: df.shape[1] for k, df in self._datapoints[0][0].items()}
        self.num_y_dims = {k: df.shape[1] for k, df in self._datapoints[0][1].items()}

        self.num_y_classes = {
            k: df.shape[1] if df.shape[1] > 1 else df.max().item() + 1
            for k, df in self._datapoints[0][1].items()
        }

        self.x_names = self.num_x_dims.keys()
        self.y_names = self.num_y_dims.keys()

    def __len__(self):
        return len(self._datapoints)

    def __getitem__(self, item):
        return self._datapoints[item]

    def __getstate__(self):
        return {
            'data_dir': self.data_dir,
            'x_pattern': self.x_pattern,
            'y_pattern': self.y_pattern,
            '_datapoints': self._datapoints,
            'num_x_dims': self.num_x_dims,
            'num_y_dims': self.num_y_dims,
            'num_y_classes': self.num_y_classes,
            'x_names': self.x_names,
            'y_names': self.y_names,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save(self, filename: str):
        utils.save_cloudpickle(filename, self)

    @classmethod
    def load(cls, filename: str):
        return utils.load_cloudpickle(filename)


class TablestakesDataset(XYCsvDataset):
    def __init__(self, docs_dir: Union[Path, str]):
        super().__init__(docs_dir)
        self.docs_dir = self.data_dir


class TablestakesDatasetLoadMaker(load_makers.LoadMaker[TablestakesDataset]):
    def __init__(self, saved_dataset_file: str, input_docs_directory_for_maker: str):
        super().__init__([saved_dataset_file])
        self.input_docs_directory_for_maker = input_docs_directory_for_maker

    def _load(self) -> TablestakesDataset:
        return utils.load_cloudpickle(self.files_to_check[0])

    def _makesave(self, *args, **kwargs) -> TablestakesDataset:
        print(f"Making TablestakesDataset({self.input_docs_directory_for_maker})")
        ds = TablestakesDataset(self.input_docs_directory_for_maker)
        print(f"Saving TablestakesDataset to {self.files_to_check[0]}")
        utils.save_cloudpickle(self.files_to_check[0], ds)
        return ds

    @classmethod
    def run_from_hp(cls, hp: "TablestakesDataModule.DataParams") -> TablestakesDataset:
        return TablestakesDatasetLoadMaker(
            saved_dataset_file=hp.get_dataset_file(),
            input_docs_directory_for_maker=hp.get_docs_dir(),
        ).loadmake(
            do_ignore_cache=hp.do_ignore_cached_dataset,
        )


class XYDocumentDataModule(pl.LightningDataModule):
    class DataParams(params.ParameterSet):
        dataset_name = None
        docs_root_dir =  None
        dataset_root_dir = None
        p_valid = 0.1
        p_test = 0.1
        seed = 42
        do_ignore_cached_dataset = False
        num_workers = 4
        num_gpus = 0
        max_seq_length = 1024
        batch_size = 32

        def get_dataset_file(self):
            return self.dataset_root_dir / f'{self.dataset_name}.cloudpickle'

        def get_docs_dir(self):
            return self.docs_root_dir / self.dataset_name

    def __init__(self, hp: DataParams):
        pl.LightningDataModule.__init__(self)
        self.hp = hp

        self.ds = self.get_dataset(hp)

        self.num_y_dims = self.ds.num_y_dims
        self.num_y_classes = self.ds.num_y_classes

        self.num_x_base_dims, self.num_x_vocab_dims = list(self.ds.num_x_dims.values())

        self.example_input_array = self.get_example_input_array()

        self.hp.num_x_dims = self.ds.num_x_dims
        self.hp.num_y_dims = self.ds.num_y_dims

        self.train_dataset, self.valid_dataset, self.test_dataset = None, None, None

    @abc.abstractmethod
    def get_example_input_array(self) -> Dict[str, torch.Tensor]:
        pass

    @abc.abstractmethod
    def get_dataset(self, hp: DataParams) -> XYCsvDataset:
        pass

    @abc.abstractmethod
    def _transform_xs(self, xs: Dict[str, List[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _transform_ys(self, ys: Dict[str, List[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        raise NotImplementedError()

    # def prepare_data(self):
    #     pass

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

        print(f'module setup ds lens: '
              f'{len(self.train_dataset)}, {len(self.valid_dataset)}, {len(self.test_dataset)}')

    def _collate_fn(self, batch: List[Any]) -> Any:
        """
        batch is a list of tuples like
          [
            ((x_base, x_vocab), (y_doc_class,)),
            ...
          ]
        """
        xs, ys = zip(*batch)
        xs = {k: [d[k] for d in xs] for k in xs[0]}
        ys = {k: [d[k] for d in ys] for k in ys[0]}

        xs = self._transform_xs(xs)
        ys = self._transform_ys(ys)

        new_xs = {}
        for k, v in xs.items():
            x_padded = torch.nn.utils.rnn.pad_sequence(v, batch_first=True)
            seq_len = x_padded.shape[1]
            if seq_len >= self.hp.max_seq_length:
                x_padded = x_padded.narrow(dim=1, start=0, length=self.hp.max_seq_length)
            new_xs[k] = x_padded
        xs = new_xs

        ys = {
            k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True, padding_value=Y_VALUE_TO_IGNORE)
            for k, v in ys.items()
        }

        return xs, ys

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

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        if isinstance(batch, (list, tuple)):
            for e in batch:
                e.to(device)
        elif isinstance(batch, MutableMapping):
            for k, v in batch.items():
                batch[k] = v.to(device)
        else:
            raise ValueError('')
        return batch


class TablestakesDataModule(XYDocumentDataModule):
    class DataParams(params.ParameterSet):
        # dataset_name = 'num=1000_7cda'
        dataset_name = 'num=100_d861'
        docs_root_dir = constants.DOCS_DIR
        dataset_root_dir = constants.DATASETS_DIR
        p_valid = 0.1
        p_test = 0.1
        seed = 42
        do_ignore_cached_dataset = False
        num_workers = 4
        num_cpus = 4
        num_gpus = 1
        max_seq_length = 1024
        batch_size = 32

        def get_dataset_file(self):
            return self.dataset_root_dir / f'{self.dataset_name}.cloudpickle'

        def get_docs_dir(self):
            return self.docs_root_dir / self.dataset_name

    def __init__(self, hp: DataParams):
        super().__init__(hp)

    def get_example_input_array(self) -> Dict[str, torch.Tensor]:
        num_example_batch_size = 32
        num_example_words = 1000

        return {
            constants.X_BASE_BASE_NAME:
                torch.tensor(np.random.rand(num_example_batch_size, num_example_words, self.num_x_base_dims)).float(),
            constants.X_VOCAB_BASE_NAME:
                torch.tensor(np.random.rand(num_example_batch_size, num_example_words)).long(),
        }

    def get_dataset(self, hp: DataParams) -> XYCsvDataset:
        return TablestakesDatasetLoadMaker.run_from_hp(hp)

    def _transform_xs(self, xs: Dict[str, List[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        xs[constants.X_BASE_BASE_NAME] = [x.float() for x in xs[constants.X_BASE_BASE_NAME]]
        xs[constants.X_VOCAB_BASE_NAME] = [x.long().squeeze(1) for x in xs[constants.X_VOCAB_BASE_NAME]]
        return xs

    def _transform_ys(self, ys: Dict[str, List[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        ys[constants.Y_WHICH_KV_BASE_NAME] = [y.long().squeeze(1) for y in ys[constants.Y_WHICH_KV_BASE_NAME]]
        ys[constants.Y_KORV_BASE_NAME] = [y.long().squeeze(1) for y in ys[constants.Y_KORV_BASE_NAME]]
        return ys


