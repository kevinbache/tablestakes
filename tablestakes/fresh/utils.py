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
