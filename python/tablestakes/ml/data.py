import glob
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import torch
from tablestakes.ml.hyperparams import MyHyperparams

from torch.utils.data import Dataset, DataLoader


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

    X_NAMES = ['x.csv', 'x_vocab.csv']
    Y_NAMES = ['y.csv']
    DATAPOINT_DIR_NAME = '*'

    @classmethod
    def _find_files_matching_patterns(cls, data_dir: Union[Path, str], patterns: List[str]):
        return [glob.glob(str(data_dir / Path(cls.DATAPOINT_DIR_NAME) / Path(p)), recursive=True) for p in patterns]

    @staticmethod
    def _read_dfs(files):
        return [pd.read_csv(f) for f in files]

    @staticmethod
    def turn_inside_out(list_of_lists):
        return list(zip(*list_of_lists))

    def __init__(self, data_dir: Union[Path, str], ys_postproc: Callable):
        self.data_dir = Path(data_dir)

        self._x_filess = self._find_files_matching_patterns(self.data_dir, self.X_NAMES)
        self._y_filess = self._find_files_matching_patterns(self.data_dir, self.Y_NAMES)

        # before turn_inside_out, self._x_df_sets[0] = [
        #       [x_df_datapoint1, x_df_datapoint2, ...],
        #       [x2_df_datapoint1, x2_df_datapoint2, ...],
        # ]
        # after turn_inside_out, self._x_df_sets[0] = [
        #       [x_df_datapoint1, x2_df_datapoint1],
        #       [x_df_datapoint2, x2_df_datapoint2],
        #       ...
        # ]
        self._x_df_sets = self.turn_inside_out([self._read_dfs(file_set) for file_set in self._x_filess])
        self._y_df_sets = self.turn_inside_out([self._read_dfs(file_set) for file_set in self._y_filess])

        # convert to tensors
        self._xs = [[torch.tensor(df.values) for df in df_set] for df_set in self._x_df_sets]
        self._ys = [[torch.tensor(df.values) for df in df_set] for df_set in self._y_df_sets]

        self._ys = ys_postproc(self._ys)

        self.num_x_dims = [x_input.shape[1] for x_input in self._x_df_sets[0]]
        self.num_y_dims = [y_input.shape[1] for y_input in self._y_df_sets[0]]
        self.num_vocab = max([max(x_vocab) for _, x_vocab in self._xs]).item() + 1

        def all_same(x: List):
            return all(e == x[0] for e in x)

        # ensure num datapoints match across input types
        all_names = self.X_NAMES + self.Y_NAMES
        df_lens = [(len(x_set), len(y_set)) for x_set, y_set in zip(self._x_df_sets, self._y_df_sets)]
        if not all_same(df_lens):
            s = ', '.join([f'{name}: {l}' for name, l in zip(all_names, df_lens)])
            raise ValueError(f"Found mismatched datapoint counts across input types: {s}.")

        # ensure num rows match within each set
        for i, (x_df_set, y_df_set) in enumerate(zip(self._x_df_sets, self._y_df_sets)):
            all_dfs = x_df_set + y_df_set
            lens = [len(df) for df in all_dfs]

            if not all_same(lens):
                s = ', '.join([f'{name}: {l}' for name, l in zip(all_names, lens)])
                raise ValueError(f"Datapoint {i} had mismatched number of rows across files: {s}.")

    def __len__(self):
        return len(self._x_df_sets)

    def __getitem__(self, item):
        return self._xs[item], self._ys[item]


if __name__ == '__main__':
    ds = XYCsvDataset('../scripts/generate_ocrd_doc_2/docs')

    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
