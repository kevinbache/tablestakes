import copy
import hashlib
from dataclasses import dataclass, fields, asdict
import enum
import itertools

import torch
from glob import glob
import json
import os
from pathlib import Path
import pickle
import time
from typing import *
import unicodedata
import xml.dom.minidom

from lxml import etree
import cloudpickle
from PIL import Image
import pdf2image

import numpy as np
import pandas as pd
from matplotlib.image import imread

import ray

from tablestakes import constants

StrDict = Dict[str, Union[str, int, float]]


class Phase(enum.Enum):
    train = 0
    valid = 1
    test = 2


import transformers
# too slow for contants
VOCAB_PAD_VALUE = transformers.BertTokenizer.from_pretrained(constants.BERT_MODEL_NAME).mask_token_id
print(f'utils.VOCAB_PAD_VALUE: {VOCAB_PAD_VALUE}')

def to_list(v: Any):
    if isinstance(v, str):
        v = [v]
    elif isinstance(v, Iterable):
        v = list(v)
    else:
        v = [v]
    return v


def dict_to_str(d: Dict, indent_width=2, extra_key_width=0, do_norm_key_width=True, line_end=''):
    if d is None or len(d) == 0:
        return ' ' * (extra_key_width + indent_width)

    indent = ' ' * indent_width
    lens = [len(k) for k in d.keys()]
    if do_norm_key_width:
        k_width = max(lens) + extra_key_width
        return '\n'.join(f'{indent}{k:{k_width}}: {v}{line_end}' for k, v in d.items())
    else:
        return '\n'.join(f'{indent}{k}: {v}{line_end}' for k, v in d.items())


def print_dict(d: Dict, indent_width=2, extra_key_width=0, do_norm_key_width=True, line_end=''):
    print(dict_to_str(d, indent_width, extra_key_width, do_norm_key_width, line_end))


def split_dict(dict_to_split: Dict, keys_for_first) -> Tuple[Dict, Dict]:
    d1 = {k: dict_to_split[k] for k in keys_for_first}
    d2 = {k: dict_to_split[k] for k in [k for k in dict_to_split if k not in keys_for_first]}

    return d1, d2


def globster(p, *args, **kwargs):
    return glob(str(p), *args, **kwargs)


DirtyPath = Union[Path, str]


def glob_multiple(base_dir: DirtyPath, patterns: List[DirtyPath], recursive=False):
    return sorted(itertools.chain(*[
        globster(base_dir / Path(p), recursive=recursive)
        for p in patterns
    ]))


def load_json(filename: DirtyPath):
    with open(filename, mode='r') as f:
        return json.load(f)


def save_json(filename: DirtyPath, json_obj: Any):
    parent = Path(filename).parent.resolve()
    mkdir_if_not_exist(parent)
    with open(filename, mode='w') as f:
        return json.dump(json_obj, f, default=str)


def load_txt(filename: DirtyPath, **kwargs):
    with open(filename, mode='r', **kwargs) as f:
        return f.read()


def load_btxt(filename: DirtyPath, **kwargs):
    with open(filename, mode='rb', **kwargs) as f:
        return f.read()


def load_txt_dirty(filename: DirtyPath, **kwargs):
    return str(load_btxt(filename, **kwargs), encoding='utf-8', errors='ignore')


def save_txt(filename: DirtyPath, txt: str):
    parent = Path(filename).parent.resolve()
    mkdir_if_not_exist(parent)
    with open(filename, mode='w') as f:
        return f.write(txt)


def load_pickle(filename: DirtyPath):
    with open(filename, mode='rb') as f:
        return pickle.load(f)


def save_pickle(filename: DirtyPath, obj: Any):
    parent = Path(filename).parent.resolve()
    mkdir_if_not_exist(parent)
    with open(filename, mode='wb') as f:
        pickle.dump(obj, f)


def load_cloudpickle(filename: DirtyPath):
    with open(filename, mode='rb') as f:
        return cloudpickle.load(f)


def save_cloudpickle(filename: DirtyPath, obj: Any):
    parent = Path(filename).parent.resolve()
    mkdir_if_not_exist(parent)
    with open(filename, mode='wb') as f:
        cloudpickle.dump(obj, f, protocol=4)


def load_csv(filename: DirtyPath, *args, **kwargs) -> pd.DataFrame:
    return pd.read_csv(filename, *args, **kwargs)


def save_csv(filename: DirtyPath, obj: pd.DataFrame, index=False, *args, **kwargs):
    return obj.to_csv(filename, index=index, *args, **kwargs)





def set_seeds(seed: int):
    from pytorch_lightning import seed_everything
    from faker import Faker
    seed_everything(seed)
    Faker.seed(seed)


def hprint(s: str, sep_char='=', do_include_pre_break_line=True):
    """Print header"""
    l = len(s)
    h = sep_char * (l + 4)
    if do_include_pre_break_line:
        print()
    print(h)
    print(f'  {s}  ')
    print(h)


# ref: https://stackoverflow.com/questions/1662351/problem-with-newlines-when-i-use-toprettyxml/39984422#39984422
def root_2_pretty_str(root: etree._Element):
    """Get a pretty string for the given tree."""
    xml_string = xml.dom.minidom.parseString(etree.tostring(root)).toprettyxml()
    xml_string = os.linesep.join([s for s in xml_string.splitlines() if s.strip()])  # remove the weird newline issue
    return xml_string


def print_tree(root: etree._Element):
    """Print a pretty string for the given tree."""
    print(root_2_pretty_str(root))


class Timer:
    TIME_FORMAT = '%H:%M:%S'

    def __init__(self, name: str, do_print_outputs=True):
        self.name = name
        self.name_str = '' if not self.name else f' "{self.name}"'
        self.do_print_outputs = do_print_outputs

    def __enter__(self):
        self.t = time.time()
        time_str = time.strftime(self.TIME_FORMAT, time.localtime(self.t))
        if self.do_print_outputs:
            print(f'Starting timer{self.name_str} at time {time_str}.', end=" ")

    def __exit__(self, *args):
        if self.do_print_outputs:
            print(f'Timer{self.name_str} took {time.time() - self.t:2.3g} secs.')


def levenshtein(a: str, b: str):
    na = len(a)
    nb = len(b)

    d = np.zeros((na + 1, nb + 1))

    d[0, :] = np.arange(nb + 1)
    d[:, 0] = np.arange(na + 1)

    for bi in range(1, nb + 1):
        for ai in range(1, na + 1):
            sub_cost = 0 if a[ai - 1] == b[bi - 1] else 1

            d[ai, bi] = np.min([
                d[ai - 1, bi] + 1,
                d[ai, bi - 1] + 1,
                d[ai - 1, bi - 1] + sub_cost,
            ])

    return d[na, nb]


def mkdir_if_not_exist(d: str, make_parents=True):
    p = Path(d)
    if p.exists():
        if p.is_dir():
            return
        else:
            raise ValueError(f'{d} exists but is not a directory.')
    else:
        p.mkdir(parents=make_parents)
        return


def prepend_before_extension(f: str, to_append: str, new_ext: Optional[str] = None):
    p = Path(f)
    ext = new_ext if new_ext is not None else p.suffix
    return p.parent / f'{p.stem}{to_append}{ext}'


def set_pandas_disp(width=200, max_rows=200):
    pd.set_option('display.max_columns', width)
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.width', width)


class PdfHandler:
    @classmethod
    def load_pdf_to_images(cls, pdf_filename: Union[str, Path], dpi: int) -> List[Image.Image]:
        return pdf2image.convert_from_path(str(pdf_filename), dpi=dpi)

    @classmethod
    def make_page_file_name(cls, page_ind: int) -> str:
        return f'page_{page_ind:02d}.png'

    @classmethod
    def save_page_images(
            cls,
            input_pdf_file: Union[Path, str],
            output_dir: Path,
            dpi: int,
    ):
        page_images = cls.load_pdf_to_images(input_pdf_file, dpi)
        page_filenames = []
        for page_ind, page_image in enumerate(page_images):
            page_filename = output_dir / cls.make_page_file_name(page_ind)
            page_filenames.append(page_filename)
            page_image.save(page_filename)
        return page_filenames


def load_image_files_to_arrays(filenames: List[Union[Path, str]]):
    """output is height x width x color array scaled [0, 255] """
    return [(imread(str(f)) * 255).astype('uint8') for f in filenames]


def generate_unique_color_matrix(num_colors: int) -> np.ndarray:
    """Make a maximally distant grid of colors in RGB color space.

    Note that these colors aren't perceptually dissimilar but it's a computer looking at them, not a human.
    """
    MAX = 255
    num_steps = int(np.ceil(np.power(num_colors, 1 / 3)))

    steps = np.linspace(0, MAX, num_steps).astype('uint8')
    rs, gs, bs = np.meshgrid(steps, steps, steps)

    out = np.array([rs.ravel(), gs.ravel(), bs.ravel()]).T

    out = out[:num_colors]

    return out


def generate_unique_color_strings(num_colors: int) -> List[str]:
    datapoint_by_color = generate_unique_color_matrix(num_colors)
    return ['rgb({r}, {g}, {b})'.format(r=e[0], g=e[1], b=e[2]) for e in datapoint_by_color]


def split_df_by_cols(df: pd.DataFrame, col_sets: List[List[str]], df_names=List[str], do_output_leftovers_df=True):
    if do_output_leftovers_df:
        # flat list
        used_cols = [col_name for col_set in col_sets for col_name in col_set]
        leftover_cols = [col_name for col_name in df.columns if col_name not in used_cols]
        col_sets.append(leftover_cols)

    return {name: df[col_set].copy() for name, col_set in zip(df_names, col_sets)}


def pow2int(num):
    return int(np.power(2, num))


def get_logger_api_key():
    return load_txt(constants.LOGGER_API_KEY_FILE).strip()


def get_neptune_fully_qualified_project_name(project_name):
    return f'{constants.NEPTUNE_USERNAME}/{project_name}'


def unicode_norm(s):
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')


def invert_list_of_lists(lol):
    return list(zip(*lol))


if __name__ == '__main__':
    assert levenshtein('', '') == 0
    assert levenshtein('asdf', '') == 4
    assert levenshtein('', 'asdf') == 4
    assert levenshtein('asdf', 'asdf') == 0
    assert levenshtein('asxf', 'asdf') == 1
    assert levenshtein('asf', 'asdf') == 1
    assert levenshtein('asdf', 'asf') == 1


def one_hot_to_categorical(df: pd.DataFrame, col_name) -> pd.DataFrame:
    return pd.DataFrame(np.argmax(df.values, axis=-1).astype(np.int), columns=[col_name])


def ray_prog_bar(obj_refs: List[ray.ObjectRef]):
    import tqdm

    def to_iterator(obj_refs):
        while obj_refs:
            try:
                done, obj_refs = ray.wait(obj_refs)
                yield ray.get(done[0])
            except TypeError as e:
                return

    for _ in tqdm.tqdm(to_iterator(obj_refs), total=len(obj_refs)):
        pass


@dataclass
class DataclassPlus:
    """dataclass with a few dict methods and a fancy __str__.

    Subclasess of DataclassPlus should still use the @dataclass decorator so that __init__s work.
        (Double check this.)

    Class members need type annotations in order to be dataclass fields.
    """
    def __contains__(self, item):
        return item in self.keys()

    def keys(self):
        return [f.name for f in fields(self)]

    # def items(self):
    #     d = {k: getattr(self, k) for k in self.keys()}
    #     return iter(d.items())

    def __getitem__(self, item):
        if item not in self.keys():
            raise ValueError(f'Keys for this class are: {self.keys()} but you tried to get {item}')
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        # return iter(self.items())
        d = {k: getattr(self, k) for k in self.keys()}
        return iter(d.items())

    def to_dict(self) -> Dict:
        return sanitize_tensors(self)

    def _str_inner(self, obj):
        typestr = obj.__class__.__name__
        if isinstance(obj, DataclassPlus):
            return str(obj).replace('\n', '\n  ')
        elif hasattr(obj, 'shape'):
            shape = ', '.join([str(e) for e in obj.shape])
            return f'{typestr}({shape})'
        elif isinstance(obj, list):
            l = len(obj)
            first_str = self._str_inner(obj[0]) if l else '[]'
            return f'{typestr}(len={l}, first={first_str})'
        elif isinstance(obj, dict):
            l = len(obj)
            k0 = list(obj.keys())[0]
            v0 = obj[k0]
            first_str = f'{self._str_inner(k0)}: {self._str_inner(v0)}' if l else '{}'
            return f'{typestr}([len={l}, first={first_str})'
        else:
            return f'{typestr}({str(obj)})'

    def __str__(self):
        d = {k: self._str_inner(v) for k, v in self}
        strs = [f'{k}={v}' for k, v in d.items()]
        arg_str = ',\n  '.join(strs)
        return f'{self.__class__.__name__}(\n  {arg_str}\n)'


def tensor_is_singleton(t: torch.Tensor):
    return t.view(-1).shape == torch.Size([1])


def sanitize_tensors(obj: Any) -> Any:
    if isinstance(obj, DataclassPlus):
        return {k: sanitize_tensors(v) for k, v in obj}
    elif isinstance(obj, torch.Tensor):
        if tensor_is_singleton(obj):
            return obj.detach().item()
        else:
            return obj.detach().numpy()
    elif isinstance(obj, dict):
        return {k: sanitize_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_tensors(o) for o in obj]
    else:
        return obj


def flatten_dict(dt, delimiter="/", prevent_delimiter=False):
    """Stolen from ray.tune"""
    dt = copy.deepcopy(dt)
    if prevent_delimiter and any(delimiter in key for key in dt):
        # Raise if delimiter is any of the keys
        raise ValueError(
            "Found delimiter `{}` in key when trying to flatten array."
            "Please avoid using the delimiter in your specification.")
    while any(isinstance(v, dict) for v in dt.values()):
        remove = []
        add = {}
        for key, value in dt.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    if prevent_delimiter and delimiter in subkey:
                        # Raise  if delimiter is in any of the subkeys
                        raise ValueError(
                            "Found delimiter `{}` in key when trying to "
                            "flatten array. Please avoid using the delimiter "
                            "in your specification.")
                    add[delimiter.join([key, str(subkey)])] = v
                remove.append(key)
        dt.update(add)
        for k in remove:
            del dt[k]
    return dt
