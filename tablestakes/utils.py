import time

from typing import *
from collections.abc import Iterable
import json

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


