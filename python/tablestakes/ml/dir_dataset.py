import abc
import itertools
from pathlib import Path
from typing import *

import pandas as pd
from tablestakes.utils import ray_prog_bar

from torch.utils.data import Dataset

import ray
from ray import remote_function

from tablestakes import utils, load_makers


class FileReference:
    """A reference to a file.  Good for ignoring parents in relative path and easy access to ext."""

    def __init__(self, p: utils.DirtyPath, root_path: Optional[utils.DirtyPath] = None):
        self.p = Path(p)
        self.root_path = Path(root_path)
        self.ext = (''.join(self.p.suffixes))[1:]
        self.parent = self.p.parent
        self.name = self.p.name

    def get_relative_path_str(self):
        return f'{str(self.p.relative_to(self.root_path))}'

    def __repr__(self):
        return f'FR({self.get_relative_path_str()})'


DirtyReference = Union[FileReference, Path, str]

FileSelectorFn = Callable[[FileReference], bool]
FileHandlerFn = Callable[[FileReference], Any]

T = TypeVar('T')


class FileHandler(abc.ABC, Generic[T]):
    def __init__(self, should_handle_fn: FileSelectorFn, handle_fn: FileHandlerFn):
        self._should_handle_fn = should_handle_fn
        self._handle_fn = handle_fn

    def should_handle(self, file: FileReference):
        return self._should_handle_fn(file)

    def handle(self, file: FileReference) -> T:
        return self._handle_fn(file)


class RayFileHandler(FileHandler, abc.ABC, Generic[T]):
    # noinspection PyMissingConstructor
    def __init__(
            self,
            should_handle_fn: FileSelectorFn,
            remote_handle_fn: remote_function.RemoteFunction,
    ):
        self._should_handle_fn = should_handle_fn
        self._handle_fn = remote_handle_fn
        assert isinstance(self._handle_fn, remote_function.RemoteFunction)

    def handle(self, file: FileReference) -> T:
        return self._handle_fn.remote(self, file)


class RayExtHandler(RayFileHandler[pd.DataFrame]):
    def __init__(self, ext: str, remote_handle_fn: remote_function.RemoteFunction):
        def _ext_eq(f: FileReference):
            return f.ext == self.ext

        super().__init__(should_handle_fn=_ext_eq, remote_handle_fn=remote_handle_fn)
        self.ext = ext


class DirectoryDatapoint:
    """Represents a directory which contains files.

    The root_path is the parent portion of the path which can be ignored.
    This helps you e.g. include filenames in your document body without
    them varying from filesystem to filesystem.
    """

    def __init__(
            self,
            p: utils.DirtyPath,
            root_path: Optional[utils.DirtyPath] = None,
    ):
        self.p = Path(p)
        self.root_path = Path(root_path) if root_path is not None else None
        self.name = self.p.name
        self.file_refs = None
        self.update_file_refs()

    def __repr__(self):
        return f'{self.__class__.__name__}({str(self.p)})'

    def update_file_refs(self):
        self.file_refs = [FileReference(f, self.root_path) for f in utils.globster(self.p / '**', recursive=True)]

    def crawl_this_dir(self, file_handlers: List[FileHandler], do_ignore_cache=True):
        outs = {}
        if do_ignore_cache:
            self.update_file_refs()
        for f in self.file_refs:
            for file_handler in file_handlers:
                if file_handler.should_handle(f):
                    outs[f] = file_handler.handle(f)

        return outs

    def count_matching_files(self, file_handlers: List[FileHandler]):
        count = 0
        for f in self.file_refs:
            for file_handler in file_handlers:
                count += int(file_handler.should_handle(f))
        return count

    def __iter__(self):
        return iter(self.file_refs)


D = TypeVar('D')


class ListDataset(Dataset, Generic[D]):
    """Represents a directory which contains files"""

    def __init__(self, datapoints: List[D]):
        self.datapoints = datapoints

    def __getitem__(self, item) -> D:
        return self.datapoints[item]

    def __len__(self):
        return len(self.datapoints)


class DirectoryDataset(ListDataset[DirectoryDatapoint]):
    def __init__(
            self,
            root_path: utils.DirtyPath,
            search_str='*',
    ):
        self.root_path = Path(root_path)
        self.load_makers = load_makers

        subdirs = utils.globster(self.root_path / search_str)
        subdirs = [Path(sd) for sd in subdirs if Path(sd).is_dir()]

        super().__init__([
            DirectoryDatapoint(f, root_path=self.root_path)
            for f in subdirs
        ])

    def __repr__(self):
        return f'{self.__class__.__name__}({self.root_path})'

    def update_file_refs(self):
        for d in self.datapoints:
            d.update_file_refs()

    def crawl_files(
            self,
            file_handlers: Union[List[RayFileHandler], RayFileHandler],
            do_ignore_cache=False,
    ):
        if isinstance(file_handlers, RayFileHandler):
            file_handlers = [file_handlers]

        outs = {}
        for dir_datapoint in self.datapoints:
            outs.update({
                str(dir_datapoint): dir_datapoint.crawl_this_dir(file_handlers, do_ignore_cache=do_ignore_cache)
            })

        return outs

    def crawl_dirs(
            self,
            dir_handler: Union[Callable[[DirectoryDatapoint], Any], remote_function.RemoteFunction],
    ):
        outs = {}
        for dir_datapoint in self.datapoints:
            if isinstance(dir_handler, remote_function.RemoteFunction):
                outs.update({str(dir_datapoint): dir_handler.remote(dir_datapoint)})
            else:
                outs.update({str(dir_datapoint): dir_handler(dir_datapoint)})

        return outs


def wait_for_crawl_outs(outs: Dict[str, Dict[str, Union[ray.ObjectRef, Any]]]):
    refs = [fileref_dict.values() for fileref_dict in outs.values()]
    refs = list(itertools.chain.from_iterable(refs))
    ray_prog_bar(refs)
    if any([isinstance(e, ray.ObjectRef) for e in refs]):
        return ray.get(refs)
    else:
        return refs


def wait_for_dir_crawl_outs(outs: Dict[str, Union[ray.ObjectRef, Any]]):
    refs = list(outs.values())
    ray_prog_bar(refs)
    if any([isinstance(e, ray.ObjectRef) for e in refs]):
        return ray.get(refs)
    else:
        return refs
