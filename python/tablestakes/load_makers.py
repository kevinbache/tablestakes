import abc
from pathlib import Path
from typing import *

import ray

T = TypeVar('T')


class LoadMaker(abc.ABC, Generic[T]):
    """A LoadMaker is designed to make caching easier.  If the cached object isn't there, then it'll make it"""

    def __init__(self, files_to_load: Union[List[str], str], verbose=False):
        super().__init__()
        if isinstance(files_to_load, str):
            files_to_load = [files_to_load]
        elif isinstance(files_to_load, Path):
            files_to_load = [str(files_to_load)]

        self.files_to_check = files_to_load
        self.verbose = verbose

    def loadmake(self, do_ignore_cache=False, *args, **kwargs) -> T:
        if all([Path(f).exists() for f in self.files_to_check]) and not do_ignore_cache:
            if self.verbose:
                print(f"{self.__class__.__name__} loading...")
            return self._load()
        else:
            if self.verbose:
                print(f"{self.__class__.__name__} make / saving...")
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
            if self.verbose:
                print(f"{self.__class__.__name__} loading...")
            return self._load()
        else:
            if self.verbose:
                print(f"{self.__class__.__name__} make / saving...")
            return self._makesave.remote(self)

    @ray.remote
    @abc.abstractmethod
    def _makesave(self, *args, **kwargs) -> T:
        pass

