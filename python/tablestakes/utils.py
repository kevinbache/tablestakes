from pathlib import Path

import time

from typing import *
from collections.abc import Iterable
import json
import os
import xml.dom.minidom

import numpy as np
import pandas as pd
from lxml import etree

StrDict = Dict[str, Union[str, int, float]]


def to_list(v: Any):
    if isinstance(v, str):
        v = [v]
    elif isinstance(v, Iterable):
        v = list(v)
    else:
        v = [v]
    return v


def dict_to_str(d: Dict, indent_width=2, extra_key_width=0, do_norm_key_width=True, line_end=''):
    indent = ' ' * indent_width
    if do_norm_key_width:
        k_width = max(len(k) for k in d.keys()) + extra_key_width
        return '\n'.join(f'{indent}{k:{k_width}}: {v}{line_end}' for k, v in d.items())
    else:
        return '\n'.join(f'{indent}{k}: {v}{line_end}' for k, v in d.items())


def print_dict(d: Dict, indent_width=2, extra_key_width=0, do_norm_key_width=True, line_end=''):
    print(dict_to_str(d, indent_width, extra_key_width, do_norm_key_width, line_end))


def read_json(filename: str):
    with open(filename, mode='r') as f:
        return json.load(f)


def read_txt(filename: str):
    with open(filename, mode='r') as f:
        return f.read()


def save_txt(filename: str, txt: str):
    with open(filename, mode='w') as f:
        return f.write(txt)


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

    def __init__(self, name: str):
        self.name = name
        self.name_str = '' if not self.name else f' "{self.name}"'

    def __enter__(self):
        self.t = time.time()
        time_str = time.strftime(self.TIME_FORMAT, time.localtime(self.t))
        print(f'Starting timer{self.name_str} at time {time_str}.', end=" ")

    def __exit__(self, *args):
        print(f'Timer{self.name_str} took {time.time() - self.t:2.3g} secs.')


def levenshtein(a: str, b: str):
    na = len(a)
    nb = len(b)

    d = np.zeros((na+1, nb+1))

    d[0, :] = np.arange(nb + 1)
    d[:, 0] = np.arange(na + 1)

    for bi in range(1, nb+1):
        for ai in range(1, na+1):
            sub_cost = 0 if a[ai - 1] == b[bi - 1] else 1

            d[ai, bi] = np.min([
                d[ai - 1, bi] + 1,
                d[ai, bi - 1] + 1,
                d[ai - 1, bi - 1] + sub_cost,
            ])

    return d[na, nb]


def mkdir_if_not_exist(d: str):
    p = Path(d)
    if p.exists():
        if p.is_dir():
            return
        else:
            raise ValueError(f'{d} exists but is not a directory.')
    else:
        p.mkdir(parents=True)
        return


def prepend_before_extension(f: str, to_append: str, new_ext: Optional[str]=None):
    p = Path(f)
    ext = new_ext if new_ext is not None else p.suffix
    return p.parent / f'{p.stem}{to_append}{ext}'


def set_pandas_width(width=200):
    pd.set_option('display.max_columns', width)
    pd.set_option('display.width', width)


def set_seed(seed: int):
    import numpy as np
    from faker import Faker
    np.random.seed(seed)
    Faker.seed(seed)


if __name__ == '__main__':
    assert levenshtein('', '') == 0
    assert levenshtein('asdf', '') == 4
    assert levenshtein('', 'asdf') == 4
    assert levenshtein('asdf', 'asdf') == 0
    assert levenshtein('asxf', 'asdf') == 1
    assert levenshtein('asf', 'asdf') == 1
    assert levenshtein('asdf', 'asf') == 1
