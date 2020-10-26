import abc
from pathlib import Path
from typing import *

import ray
from tablestakes import utils
from tablestakes.ml import data as ts_data

T = TypeVar('T')


class LoadMaker(abc.ABC, Generic[T]):
    """A LoadMaker is designed to make caching easeir.  If If the cached object isn't there, then it'll make it"""

    def __init__(self, files_to_load: Union[List[str], str]):
        super().__init__()
        if isinstance(files_to_load, str):
            files_to_load = [files_to_load]
        elif isinstance(files_to_load, Path):
            files_to_load = [str(files_to_load)]

        self.files_to_check = files_to_load

    def loadmake(self, do_ignore_cache=False, *args, **kwargs) -> T:
        if all([Path(f).exists() for f in self.files_to_check]) and not do_ignore_cache:
            return self._load()
        else:
            return self._makesave(*args, **kwargs)

    @abc.abstractmethod
    def _load(self) -> T:
        pass

    @abc.abstractmethod
    def _makesave(self, *args, **kwargs) -> T:
        pass


class RayLoadMaker(LoadMaker, abc.ABC, Generic[T]):
    def loadmake(self, do_ignore_cache=False, *args, **kwargs) -> T:
        if all([Path(f).exists() for f in self.files_to_check]) and not do_ignore_cache:
            return self._load()
        else:
            return self._makesave.remote(self)

    @ray.remote
    @abc.abstractmethod
    def _makesave(self, *args, **kwargs) -> T:
        pass


class TablestakesDatasetLoadMaker(LoadMaker[ts_data.TablestakesDataset]):
    def __init__(self, saved_dataset_file: str, input_docs_directory_for_maker: str):
        super().__init__([saved_dataset_file])
        self.input_docs_directory_for_maker = input_docs_directory_for_maker

    def _load(self) -> ts_data.TablestakesDataset:
        return utils.load_cloudpickle(self.files_to_check[0])

    @ray.remote
    def _makesave(self, *args, **kwargs) -> ts_data.TablestakesDataset:
        ds = ts_data.TablestakesDataset(self.input_docs_directory_for_maker)
        utils.save_cloudpickle(self.files_to_check[0], ds)
        return ds
