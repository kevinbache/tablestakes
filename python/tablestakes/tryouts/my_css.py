import enum
from typing import Text, List, Dict, Any, Optional, Tuple


class Css:
    def __init__(self):
        self._dict = {}

    def __str__(self):
        out = ''
        for selector, values_dict in self._dict.items():
            out += f'{selector}  {self._inner_dict_to_str(values_dict)}\n\n'
        return out

    def __add__(self, other):
        self.update(other.get_dict())
        return self

    def get_dict(self):
        return self._dict

    def update(self, d: Dict[Text, Dict]):
        self._dict.update(d)

    def update_selector(self, selector: Text, values_dict: Dict[Text, Text]):
        self.update({selector: values_dict})

    @staticmethod
    def _inner_dict_to_str(d: Dict[Text, Any]):
        out = '\n  '.join([f'{str(k)}: {str(v)};' for k, v in d.items()])
        return f'{{\n  {out}\n}}'


class LabelBoxConfig(enum.Enum):
    # names are where the label is relative to the content
    # start in the upper left left, go clockwise
    # UL: Label is in the upper left.
    #    Label  col, row  Content col,  row
    UL =       ((1,   1),         (2,    2))
    U =        ((1,   1),         (1,    2))
    UR =       ((2,   1),         (1,    2))
    R =        ((2,   1),         (1,    1))
    BR =       ((2,   2),         (1,    1))
    B =        ((1,   2),         (1,    1))
    BL =       ((1,   2),         (2,    1))
    L =        ((1,   1),         (2,    1))

    def __init__(self, label_row_col: Tuple[int, int], content_row_col: Tuple[int, int]):
        self.label_col = label_row_col[0]
        self.label_row = label_row_col[1]
        self.content_col = content_row_col[0]
        self.content_row = content_row_col[1]

    def get_max_cols(self):
        return max(self.label_col, self.content_col)

    def get_max_rows(self):
        return max(self.label_row, self.content_row)


if __name__ == '__main__':
    probability = [
        0.3,
        1.0,
        0.01,
        0.1,
        0.01,
        0.2,
        0.01,
        1.0,
    ]
