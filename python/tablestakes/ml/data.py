import collections
import glob
from pathlib import Path
import re
from typing import *

import pandas as pd

import torch
from tablestakes import constants, utils
from torch.utils.data import Dataset

X_PREFIX = constants.X_PREFIX
Y_PREFIX = constants.Y_PREFIX


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
    def _read_csvs_from_dict(d: dict, key_to_base_name_fn: Callable[[str], str], df_postproc: Callable[[pd.DataFrame], pd.DataFrame]):
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
    def __init__(
            self,
            docs_dir: Union[Path, str],
    ):
        super().__init__(docs_dir)
        self.docs_dir = self.data_dir


if __name__ == '__main__':
    pass
    # from tablestakes import utils
    # this_docs_dir = constants.DOCS_DIR / doc_set_name
    #
    # tablestakes_meta = TablestakesMeta.from_metas_dir()
    #
    # with utils.Timer('ds creation'):
    #     ds = TablestakesDataset(this_docs_dir)
    #
    # ds_file = str(constants.DOCS_DIR / (doc_set_name + '_dataset.cloudpickle'))
    # with utils.Timer('ds save'):
    #     utils.save_cloudpickle(ds_file, ds)
    #
    # with utils.Timer('ds2 load'):
    #     ds2 = utils.load_cloudpickle(ds_file)
    #
