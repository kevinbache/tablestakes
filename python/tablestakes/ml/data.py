import collections
import glob
from pathlib import Path
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
        meta
            num_y_classes.json
            word_to_count.json
            word_to_id.json
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
        assert cls.all_same(lens), f'Got mismatched numbers of input files in different directories.  Lens: {lens}'
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
        if len(self._x_filename_dicts) == 0:
            raise ValueError(f"Found no files matching {self.data_dir / self.x_pattern}")
        if len(self._y_filename_dicts) == 0:
            raise ValueError(f"Found no files matching {self.data_dir / self.y_pattern}")

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


class TablestakesMeta:
    def __init__(self, word_to_count, word_to_id, num_y_classes, **kwargs):
        self.word_to_count = word_to_count
        self.word_to_id = word_to_id
        self.num_y_classes = num_y_classes
        self._others = kwargs

    def save(self, target_dir: Path):
        utils.mkdir_if_not_exist(target_dir)

        metas_dict = {
            'word_to_count': self.word_to_count,
            'word_to_id': self.word_to_id,
            'num_y_classes': self.num_y_classes,
            **self._others,
        }

        for meta_name, meta_obj in metas_dict.items():
            utils.save_json(target_dir / f'{meta_name}.json', meta_obj)

    @classmethod
    def from_metas_dir(cls, docs_dir: Path):
        meta_dir = TablestakesDataset.get_meta_dir(docs_dir)

        metas_dict = {
            meta_name: utils.load_json(meta_dir / f'{meta_name}.json')
            for meta_name in ['word_to_count', 'word_to_id', 'num_y_classes']
        }
        return cls(**metas_dict)

    @classmethod
    def from_metas_dict(cls, metas_dict: Dict[str, Union[Dict, List, str, int, float]]):
        return cls(**metas_dict)

    @classmethod
    def from_dict(cls, d):
        o = cls({}, {}, {})
        o.__dict__.update(d)
        return o

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class TablestakesDataset(XYCsvDataset):
    def __init__(
            self,
            docs_dir: Union[Path, str],
            meta: Optional[TablestakesMeta] = None,
            x_pattern: Path = Path('**') / f'{X_PREFIX}*.csv',
            y_pattern: Path = Path('**') / f'{Y_PREFIX}*.csv',
    ):
        super().__init__(docs_dir, x_pattern, y_pattern)
        self.meta = meta or TablestakesMeta.from_metas_dir(self.get_default_meta_dir())
        self.docs_dir = self.data_dir

    def get_num_vocab(self) -> int:
        return len(self.word_to_id)

    @classmethod
    def get_meta_dir(cls, docs_dir: str) -> Path:
        return Path(docs_dir) / constants.META_DIR_NAME

    def get_default_meta_dir(self) -> Path:
        return self.get_meta_dir(self.data_dir)

    def save(self, filename: str):
        utils.save_cloudpickle(filename, self)

    @classmethod
    def load(self, filename: str):
         return utils.load_cloudpickle(filename)

    def __getstate__(self):
        d =  self.__dict__
        d['meta'] = self.meta.__dict__
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.meta = TablestakesMeta.from_dict(state['meta'])



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
