import abc
from pathlib import Path
from typing import *

from tablestakes import utils
from tablestakes.ml import data as ts_data

T = TypeVar('T')


class LoadMaker(abc.ABC, Generic[T]):
    def __init__(self, files_to_load: List[str]):
        super().__init__()
        if isinstance(files_to_load, str):
            files_to_load = [files_to_load]
        elif isinstance(files_to_load, Path):
            files_to_load = [str(files_to_load)]

        self.files_to_check = files_to_load

    def loadmake(self) -> T:
        if all([Path(f).exists() for f in self.files_to_check]):
            return self._load()
        else:
            return self._make()

    @abc.abstractmethod
    def _load(self) -> T:
        pass

    @abc.abstractmethod
    def _make(self) -> T:
        pass


class DatasetLoadMaker(LoadMaker[ts_data.TablestakesDataset]):
    def __init__(self, saved_dataset_file: str, input_docs_directory_for_maker: str):
        super().__init__([saved_dataset_file])
        self.input_docs_directory_for_maker = input_docs_directory_for_maker

    def _load(self) -> ts_data.TablestakesDataset:
        return utils.load_cloudpickle(self.files_to_check[0])

    def _make(self) -> ts_data.TablestakesDataset:
        ds = ts_data.TablestakesDataset(self.input_docs_directory_for_maker)
        utils.save_cloudpickle(self.files_to_check[0], ds)
        return ds
