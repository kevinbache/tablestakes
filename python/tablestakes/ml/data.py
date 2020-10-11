import collections
import glob
from pathlib import Path
from typing import *

import pandas as pd

import torch
from tablestakes import constants
from torch.utils.data import Dataset

X_PREFIX = constants.X_PREFIX
Y_PREFIX = constants.Y_PREFIX


class XYCsvDataset(Dataset):
    """
    Expects data to look like this:

    data_dir
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

    @staticmethod
    def all_same(x: List):
        return all(e == x[0] for e in x)

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
        lens = [len(e) for e in out]
        assert cls.all_same(lens), 'Got mismatched numbers of input files in different directories'
        return out

    @staticmethod
    def _read_csvs_from_dict(d: dict, remove_from_k: str):
        return {k.replace(remove_from_k, ''): pd.read_csv(v) for k, v in d.items()}

    @staticmethod
    def _convert_dict_of_dfs_to_tensors(d: dict):
        return {k: torch.tensor(v.values) for k, v in d.items()}

    def __init__(
            self,
            data_dir: Union[Path, str],
            x_pattern=Path('**') / f'{X_PREFIX}*.csv',
            y_pattern=Path('**') / f'{Y_PREFIX}*.csv',
    ):
        self.data_dir = str(Path(data_dir))
        self.x_pattern = x_pattern
        self.y_pattern = y_pattern

        self._x_filename_dicts = self._find_files(self.data_dir / self.x_pattern)
        self._y_filename_dicts = self._find_files(self.data_dir / self.y_pattern)

        df_dicts = []
        for x_filename_dict, y_filename_dict in zip(self._x_filename_dicts, self._y_filename_dicts):
            df_dicts.append((
                self._read_csvs_from_dict(x_filename_dict, remove_from_k=X_PREFIX),
                self._read_csvs_from_dict(y_filename_dict, remove_from_k=Y_PREFIX),
            ))

        self._datapoints = []
        for x_dict, y_dict in df_dicts:
            self._datapoints.append((
                self._convert_dict_of_dfs_to_tensors(x_dict),
                self._convert_dict_of_dfs_to_tensors(y_dict),
            ))

        self.num_x_dims = {k: df.shape[1] for k, df in df_dicts[0][0].items()}
        self.num_y_dims = {k: df.shape[1] for k, df in df_dicts[0][1].items()}

        self.x_names = [k for k in self._datapoints[0][0].keys()]
        self.y_names = [k for k in self._datapoints[0][1].keys()]

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
            'x_names': self.x_names,
            'y_names': self.y_names,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)


if __name__ == '__main__':
    from tablestakes import utils

    doc_set_name = 'num=2000_e4d0'
    with utils.Timer('ds creation'):
        ds = XYCsvDataset(constants.DOCS_DIR / doc_set_name)

    ds_file = str(constants.DOCS_DIR / (doc_set_name + '_dataset.cloudpickle'))
    with utils.Timer('ds save'):
        utils.save_cloudpickle(ds_file, ds)

    with utils.Timer('ds2 load'):
        ds2 = utils.load_cloudpickle(ds_file)
