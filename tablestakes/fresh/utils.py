from pathlib import Path
from typing import *
from collections.abc import Iterable


StrDict = Dict[str, str]


def to_list(v: Any):
    if isinstance(v, str):
        v = [v]
    elif isinstance(v, Iterable):
        v = list(v)
    else:
        v = [v]
    return v
