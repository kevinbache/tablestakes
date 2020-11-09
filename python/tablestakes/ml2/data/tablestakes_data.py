from dataclasses import dataclass
from typing import *

import numpy as np
import pandas as pd
import torch

from tablestakes import utils, load_makers, constants
from tablestakes.ml2 import data
from tablestakes.ml2.data import data_module, datapoints


class TablestakesHandledDataset(data.XYMetaDirHandlerDataset):
    def __init__(
            self,
            base_dir: utils.DirtyPath,
            data_dir_glob_patterns: Optional[List[utils.DirtyPath]] = None,
            data_dir_glob_recursive=False,
    ):

        super().__init__(
            base_dir=base_dir,
            x_handler=data.BaseVocabXHandler(),
            y_handler=data.KorvWhichYHandler(),
            meta_handler=data.ShortMetaHandler(),
            data_dir_glob_patterns=data_dir_glob_patterns,
            data_dir_glob_recursive=data_dir_glob_recursive,
        )


class TablestakesHandledDatasetLoadMaker(load_makers.LoadMaker[TablestakesHandledDataset]):
    def __init__(self, saved_dataset_file: str, input_docs_directory_for_maker: str, verbose=False):
        super().__init__([saved_dataset_file], verbose=verbose)
        self.input_docs_directory_for_maker = input_docs_directory_for_maker

    def _load(self) -> TablestakesHandledDataset:
        return utils.load_cloudpickle(self.files_to_check[0])

    def _makesave(self, *args, **kwargs) -> TablestakesHandledDataset:
        if self.verbose:
            print(f"{self.__class__.__name__} making ddataset from {self.input_docs_directory_for_maker}")
        ds = TablestakesHandledDataset(self.input_docs_directory_for_maker)
        if self.verbose:
            print(f"{self.__class__.__name__} saving to {self.files_to_check[0]}")
        utils.save_cloudpickle(self.files_to_check[0], ds)
        return ds

    @classmethod
    def run_from_hp(cls, hp: "TablestakesHandlerDataModule.DataParams") -> TablestakesHandledDataset:
        return TablestakesHandledDatasetLoadMaker(
            saved_dataset_file=hp.dataset_file,
            input_docs_directory_for_maker=hp.docs_dir,
        ).loadmake(
            do_ignore_cache=hp.do_ignore_cached_dataset,
        )


class TablestakesHandlerDataModule(data_module.XYMetaHandlerDatasetModule):
    def get_example_input_array(self) -> datapoints.XYMetaDatapoint:
        num_example_batch_size = 32
        num_example_words = 10000

        return datapoints.XYMetaDatapoint(
            x=datapoints.BaseVocabDatapoint(
                base=torch.tensor(
                    np.random.rand(num_example_batch_size, num_example_words, self.num_x_base_dims),
                    dtype=torch.float
                ),
                vocab=torch.tensor(np.random.randint(num_example_batch_size, num_example_words), dtype=torch.long),
            ),
            y=datapoints.KorvWhichDatapoint(
                korv=torch.tensor(np.random.randint(num_example_batch_size, num_example_words), dtype=torch.long),
                which_kv=torch.tensor(np.random.randint(num_example_batch_size, num_example_words), dtype=torch.long),
            ),
            meta=datapoints.MetaDatapoint(
                datapoint_dir=['test_datapoint'] * num_example_batch_size,
            ),
        )

    def get_dataset(self, hp: data_module.DataParams) -> TablestakesHandledDataset:
        return TablestakesHandledDatasetLoadMaker.run_from_hp(hp)


if __name__ == '__main__':
    dataset_name = 'num=100_8163'

    hp_dm = data_module.DataParams(
        dataset_name=dataset_name,
        docs_root_dir=constants.DOCS_DIR,
        dataset_root_dir=constants.DATASETS_DIR,
        do_ignore_cached_dataset=True,
    )
    hp_dm.batch_size = 5

    dm = TablestakesHandlerDataModule(hp=hp_dm)
    dm.setup()

    for batch in dm.train_dataloader():
        break

    print(str(batch))
