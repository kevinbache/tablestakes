import enum
from typing import *

from tablestakes.tryouts import tags

"""
param: normal row height multiplier 
"""


class KeyValueContents:
    def __init__(self, k: Text, v: Text):
        self.k = k
        self.v = v


class KeyValueStyle:
    def __init__(self):
        pass


# class KeyValue:
#     def __init__(self, contents: KeyValueContents, style: KeyValueStyle):
#         self.contents = contents
#         self.style = style


class KVDiv(enum.Enum):
    CONTAINER = 1
    KEY = 2
    VALUE = 3


StyleDict = Dict[Text, Text]


class KeyValue:
    def __init__(self, container_class: Text, key_contents: tags.TextTagList, value_contents: tags.TextTagList):
        self.container_class = container_class
        self.key_contents = key_contents
        self.value_contents = value_contents

        self.div_styles = {div: {} for div in KVDiv}

    def style(self, style: StyleDict, div: KVDiv):
        self.div_styles[div].update(style)

