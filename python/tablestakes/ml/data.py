import collections
import glob
from pathlib import Path
from typing import *

import pandas as pd

import torch
from tablestakes import constants
from torch.utils.data import Dataset, DataLoader

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

    @staticmethod
    def turn_inside_out(list_of_lists):
        return list(zip(*list_of_lists))

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
        self.data_dir = Path(data_dir)
        self.x_pattern = x_pattern
        self.y_pattern = y_pattern

        self._x_filename_dicts = self._find_files(self.data_dir / self.x_pattern)
        self._y_filename_dicts = self._find_files(self.data_dir / self.y_pattern)

        self._df_dicts = []
        for x_filename_dict, y_filename_dict in zip(self._x_filename_dicts, self._y_filename_dicts):
            self._df_dicts.append((
                self._read_csvs_from_dict(x_filename_dict, remove_from_k=X_PREFIX),
                self._read_csvs_from_dict(y_filename_dict, remove_from_k=Y_PREFIX),
            ))

        self._datapoints = []
        for x_dict, y_dict in self._df_dicts:
            self._datapoints.append((
                self._convert_dict_of_dfs_to_tensors(x_dict),
                self._convert_dict_of_dfs_to_tensors(y_dict),
            ))

        self.num_x_dims = {k: df.shape[1] for k, df in self._df_dicts[0][0].items()}
        self.num_y_dims = {k: df.shape[1] for k, df in self._df_dicts[0][1].items()}

        self.x_names = [k for k in self._datapoints[0][0].keys()]
        self.y_names = [k for k in self._datapoints[0][1].keys()]

    def __len__(self):
        return len(self._datapoints)

    def __getitem__(self, item):
        return self._datapoints[item]


if __name__ == '__main__':
    doc_set_name = 'num=1000_extra=0'
    ds = XYCsvDataset(constants.DOCS_DIR / doc_set_name)

    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
