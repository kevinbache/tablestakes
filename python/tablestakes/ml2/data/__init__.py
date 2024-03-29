import abc
from collections import defaultdict
from pathlib import Path
import re
from typing import *

from torch.utils.data import Dataset

from tablestakes import constants, utils
from tablestakes.ml2.data import datapoints


D = TypeVar('D')


class ListDataset(Dataset, Generic[D]):
    """Represents a directory which contains files"""

    def __init__(self, datapoints: List[D]):
        self.datapoints = datapoints

    def __getitem__(self, item) -> D:
        return self.datapoints[item]

    def __len__(self):
        return len(self.datapoints)

    def save(self, filename: str):
        utils.save_cloudpickle(filename, self)

    @classmethod
    def load(cls, filename: str):
        return utils.load_cloudpickle(filename)


DP = TypeVar('DP')


class SubtypeCsvJsonHandler(Generic[DP], abc.ABC):
    def __init__(
            self,
            subtype_name: str,
            patterns=None,
            glob_recursive=False,
            filename_to_sub_name: Optional[Callable[[utils.DirtyPath], str]] = None,
    ):
        prefix = f'{subtype_name}_'
        if patterns is None:
            patterns = [f'{prefix}*.csv', f'{prefix}*.json']

        self.glob_recursive = glob_recursive
        self.patterns = patterns

        # eg.: filename = '/blah/blah/My Document.x_base.csv'
        #      sub_name = 'base'
        if filename_to_sub_name is None:
            filename_to_sub_name = lambda filename: self._match_between_prefix_and_ext(filename, prefix=prefix)
        self.filename_to_subtype_name = filename_to_sub_name

    def handle(self, datapoint_dir: Path) -> DP:
        files = utils.glob_multiple(datapoint_dir, self.patterns, self.glob_recursive)
        subname_to_filename = {self.filename_to_subtype_name(f): f for f in files}
        return self._files_to_subtype(subname_to_filename)

    @staticmethod
    def _match_between_prefix_and_ext(filename: str, prefix: str) -> str:
        m = re.match(fr'.*{prefix}(\w+).(csv|json)$', filename)
        after_prefix = m.groups()[0]
        if after_prefix.endswith('.csv'):
            after_prefix = after_prefix.replace('.csv', '')
        return after_prefix

    @abc.abstractmethod
    def _files_to_subtype(self, subname_to_file: Dict[str, Path]) -> DP:
        pass


class BaseVocabXHandler(SubtypeCsvJsonHandler[datapoints.BaseVocabDatapoint]):
    def __init__(self):
        super().__init__(subtype_name='x', patterns=['**/*.x_*.csv'], glob_recursive=True)

    def _files_to_subtype(self, subname_to_file: Dict[str, Path]) -> datapoints.BaseVocabDatapoint:
        # print(f'BaseVocabXHandler subname_to_file:')
        # utils.print_dict(subname_to_file)
        print('BaseVocabXHandler._files_to_subtype.subname_to_file')
        utils.print_dict(subname_to_file)
        print(' Done with print dict')
        return datapoints.BaseVocabDatapoint(
            base=utils.load_csv(subname_to_file[constants.X_BASE_BASE_NAME]),
            vocab=utils.load_csv(subname_to_file[constants.X_VOCAB_BASE_NAME]),
        )


class BaseVocabMultiXHandler(SubtypeCsvJsonHandler[datapoints.BaseVocabMultiDatapoint]):
    """Converts a directory into a datapoint.  Multiple docs per datapoint."""
    def __init__(self):
        super().__init__(subtype_name='x', patterns=['**/*x_*.csv'], glob_recursive=True)

    def _files_to_subtype(self, subname_to_file: Dict[str, Path]) -> DP:
        pass

    def handle(self, datapoint_dir: Path) -> DP:
        # subbame is e.g. 'base' or 'vocab'
        subname_to_docind_to_filename = defaultdict(dict)
        # print("BaseVocabMultiXHandler.handle:")

        # print(f'datapoint_dir: {datapoint_dir}')
        # print(f'self.patterns: {self.patterns}')

        x_files = utils.glob_multiple(datapoint_dir, self.patterns, self.glob_recursive)
        # print(f'x_files: {x_files}')
        for x_file in x_files:
            subname = self.filename_to_subtype_name(x_file)
            doc_ind = self._get_attachment_doc_ind(x_file)
            # print(f'x_file: {x_file}')
            # print(f'subname: {subname}')
            # print(f'doc_ind: {doc_ind}')
            subname_to_docind_to_filename[subname][doc_ind] = x_file

        sorter = lambda tup: tup[0]

        base_dfs = []
        filenames = []
        for doc_ind, f in sorted(subname_to_docind_to_filename[constants.X_BASE_BASE_NAME].items(), key=sorter):
            base_dfs.append(utils.load_csv(f, index_col=0))
            filenames.append(self._get_attachment_filename(f))

        # print(f'base_dfs: {base_dfs}')
        # print(f'filenames: {filenames}')

        vocab_dfs = []
        for doc_ind, f in sorted(subname_to_docind_to_filename[constants.X_VOCAB_BASE_NAME].items(), key=sorter):
            vocab_dfs.append(utils.load_csv(f, index_col=0))

        return datapoints.BaseVocabMultiDatapoint(
            base=base_dfs,
            vocab=vocab_dfs,
            filenames=filenames,
        )

    def _get_attachment_doc_ind(self, p: utils.DirtyPath):
        return int(Path(p).parent.name.replace('doc_', ''))

    def _get_attachment_filename(self, p: utils.DirtyPath):
        name = Path(p).name
        for suffix in (constants.X_BASE_NAME, constants.X_VOCAB_NAME):
            name = utils.replace_endswith(name, f'.{suffix}.csv', '')
        return name


class KorvWhichYHandler(SubtypeCsvJsonHandler[datapoints.KorvWhichDatapoint]):
    def __init__(self):
        super().__init__(subtype_name='y', patterns=['**/y_*.csv'], glob_recursive=True)

    def _files_to_subtype(self, subname_to_file: Dict[str, Path]) -> datapoints.KorvWhichDatapoint:
        return datapoints.KorvWhichDatapoint(
            korv=utils.load_csv(subname_to_file[constants.Y_KORV_BASE_NAME]),
            which_kv=utils.load_csv(subname_to_file[constants.Y_WHICH_KV_BASE_NAME]),
        )


class ShortMetaHandler(SubtypeCsvJsonHandler[datapoints.MetaDatapoint]):
    def __init__(self):
        super().__init__(subtype_name='meta', patterns=['**/meta_short.csv'], glob_recursive=True)

    def _files_to_subtype(self, subname_to_file: Dict[str, Path]) -> datapoints.MetaDatapoint:
        df = utils.load_csv(subname_to_file[constants.META_SHORT_BASE_NAME])
        d = df.iloc[0].to_dict()
        return datapoints.MetaDatapoint(datapoint_dir=d[constants.META_ORIGINAL_DATA_DIR_COL_NAME])


class DirHandlerDataset(ListDataset[DP]):
    def __init__(
            self,
            base_dir: utils.DirtyPath,
            dir_handler: Callable[[utils.DirtyPath], DP],
            data_dir_glob_patterns: Optional[List[utils.DirtyPath]] = None,
            data_dir_glob_recursive=False,
    ):
        self.base_dir = base_dir

        if data_dir_glob_patterns is None:
            data_dir_glob_patterns = ['*']

        datapoint_dirs = utils.glob_multiple(self.base_dir, data_dir_glob_patterns, data_dir_glob_recursive)
        datapoint_dirs = [Path(sd) for sd in datapoint_dirs if Path(sd).is_dir()]
        datapoints = [dir_handler(d) for d in datapoint_dirs]
        super().__init__(datapoints=datapoints)

    def get_num_features(self):
        assert len(self.datapoints) > 0
        dp = self.datapoints[0]

        return dp.get_num_features()


class XYMetaDirHandlerDataset(DirHandlerDataset[datapoints.XYMetaDatapoint]):
    def __init__(
            self,
            base_dir: utils.DirtyPath,
            x_handler: SubtypeCsvJsonHandler,
            y_handler: SubtypeCsvJsonHandler,
            meta_handler: SubtypeCsvJsonHandler,
            data_dir_glob_patterns: Optional[List[utils.DirtyPath]] = None,
            data_dir_glob_recursive=False,
    ):
        def dir_handler(d: utils.DirtyPath) -> datapoints.XYMetaDatapoint:
            return datapoints.XYMetaDatapoint.from_makers(
                data_dir=d,
                x_maker=x_handler.handle,
                y_maker=y_handler.handle,
                meta_maker=meta_handler.handle,
            )

        super().__init__(
            base_dir,
            dir_handler,
            data_dir_glob_patterns,
            data_dir_glob_recursive,
        )

