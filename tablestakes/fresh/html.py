import abc
import enum
from typing import *

import yattag

from tablestakes.fresh import utils


HtmlChunk = List[Union[str, 'HtmlTag']]
# an HtmlChunk which can also be a raw str or tag rather than a list of them.
DirtyHtmlChunk = Union[HtmlChunk, str, 'HtmlTag']


def _clean_dirty_html_chunk(chunk: DirtyHtmlChunk):
    if isinstance(chunk, (str, HtmlTag)):
        chunk = [chunk]
    if not isinstance(chunk, list):
        raise ValueError(f'chunk is type: {type(chunk)}')
    return chunk


def _html_chunk_to_str(chunk: HtmlChunk):
    return ''.join([str(t) for t in chunk])


class CssSelector:
    """Raw CssSelector.  Can be a complex string."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(self.to_selector_str())

    def to_selector_str(self): return self.name


class HtmlClass(CssSelector):
    def to_selector_str(self): return f'.{self.name}'


class AbstractHtmlClasses(CssSelector, abc.ABC):
    def __init__(self, classes: List[str]):
        super().__init__('')
        del self.name

        self.classes = classes

    def __repr__(self):
        return self.classes

    @abc.abstractmethod
    def get_join_str(self): pass

    def to_selector_str(self): return self.get_join_str().join([f'.{c}' for c in self.classes])


class HtmlClassesAll(AbstractHtmlClasses):
    def get_join_str(self): return ''


class HtmlClassesNested(AbstractHtmlClasses):
    def get_join_str(self): return ' '


class HtmlId(CssSelector):
    def to_selector_str(self): return f'#{self.name}'


HtmlClassesType = Union[List[HtmlClass], HtmlClass]


class HtmlTag:
    def __init__(
            self,
            tag_name: str,
            contents: DirtyHtmlChunk,
            classes: HtmlClassesType,
            attributes: Optional[utils.StrDict] = None,
    ):
        self.tag_name = tag_name
        self.contents = _clean_dirty_html_chunk(contents)
        self.classes = utils.to_list(classes)
        self.attributes = attributes or {}

    def get_class_str(self):
        return ' '.join([str(c) for c in self.classes])

    def __str__(self):
        doc = yattag.Doc()

        with doc.tag(self.tag_name, klass=self.get_class_str(), **self.attributes):
            doc.asis(_html_chunk_to_str(self.contents))

        return yattag.indent(doc.getvalue())


class Div(HtmlTag):
    def __init__(self, contents: DirtyHtmlChunk, classes: HtmlClassesType, attributes: Optional[utils.StrDict] = None):
        super().__init__('div', contents, classes, attributes)


class KVHtml(Div):
    def __init__(
            self,
            k_tag: HtmlTag,
            v_tag: HtmlTag,
            container_classes: HtmlClassesType,
            container_attributes: Optional[utils.StrDict] = None,
    ):
        super().__init__([k_tag, v_tag], container_classes, container_attributes)
        self._k_tag = k_tag
        self._v_tag = v_tag

    @staticmethod
    def _get_container_class(kv_name: str):
        return HtmlClass(f'{kv_name}_container')

    @staticmethod
    def _get_key_class(kv_name: str):
        return HtmlClass(f'{kv_name}_key')

    @staticmethod
    def _get_value_class(kv_name: str):
        return HtmlClass(f'{kv_name}_value')

    @classmethod
    def from_strs(
            cls,
            kv_name: str,
            k_contents: DirtyHtmlChunk,
            v_contents: DirtyHtmlChunk,
    ) -> 'KVHtml':
        return cls(
            container_classes=[cls._get_container_class(kv_name)],
            k_tag=Div(k_contents, classes=[cls._get_key_class(kv_name)]),
            v_tag=Div(v_contents, classes=[cls._get_value_class(kv_name)]),
        )

    def get_container_classes(self):
        return self.classes

    def get_key_classes(self):
        return self._k_tag.classes

    def get_value_classes(self):
        return self._v_tag.classes


class KLoc(enum.Enum):
    # names are where the key is relative to the value
    # start in the upper left left, go clockwise
    # UL: Label is in the upper left.
    #    Key   col, row    Value col,  row
    UL =       ((1,   1),         (2,    2))
    U =        ((1,   1),         (1,    2))
    UR =       ((2,   1),         (1,    2))
    R =        ((2,   1),         (1,    1))
    BR =       ((2,   2),         (1,    1))
    B =        ((1,   2),         (1,    1))
    BL =       ((1,   2),         (2,    1))
    L =        ((1,   1),         (2,    1))

    def __init__(self, key_row_col: Tuple[int, int], value_row_col: Tuple[int, int]):
        self.key_col = key_row_col[0]
        self.key_row = key_row_col[1]
        self.value_col = value_row_col[0]
        self.value_row = value_row_col[1]

    @property
    def num_cols(self):
        return max(self.key_col, self.value_col)

    @property
    def num_rows(self):
        return max(self.key_row, self.value_row)


class CssChunk:
    def __init__(self, selector: CssSelector, values: utils.StrDict):
        self.selector = selector
        self.values = values

    def __str__(self):
        dict_str = utils.dict_to_str(self.values, do_norm_key_width=False, line_end=';')
        return f'''{self.selector.to_selector_str()} {{
{dict_str}
}}'''

    def __add__(self, other: 'CssChunk'):
        if other.selector != self.selector:
            raise ValueError("You can only add two chunks whose selectors are equal.  "
                             "Use the CSS class instead too aggregate multiple CSSChunks "
                             "which have different selectors.")


class Css:
    def __init__(self, chunks: Optional[List[CssChunk]] = None):
        self._selector_to_chunk = {}

        if chunks:
            for chunk in chunks:
                self._selector_to_chunk[chunk.selector] = chunk

    def _add_chunk(self, chunk: CssChunk):
        if chunk.selector in self._selector_to_chunk:
            self._selector_to_chunk[chunk.selector] += chunk
        else:
            self._selector_to_chunk[chunk.selector] = chunk
        return self

    def __add__(self, other: Union[CssChunk, 'Css']):
        if isinstance(other, CssChunk):
            self._add_chunk(other)
        elif isinstance(other, Css):
            for chunk in other:
                self._add_chunk(chunk)
        return self

    def __iter__(self):
        for chunk in self._selector_to_chunk.values():
            yield chunk

    def __repr__(self):
        """str(chunk) includes the selector"""
        return '\n\n'.join([str(chunk) for chunk in self])


if __name__ == '__main__':
    css = Css([
        CssChunk(HtmlClass('asdf'), {'style': 'grid'}),
        CssChunk(HtmlClass('2qwer'), {'style2': 'grid2'}),
    ])
    for chunk in css:
        print(chunk)

    kvh = KVHtml.from_strs('address_to', 'Address To:', '1232 Apache Ave </br>Santa Fe, NM 87505')

    print(kvh)
    print(kvh.get_container_classes())
    print(kvh.get_key_classes())
    print(kvh.get_value_classes())

    kv_name = 'address_to'
    kloc = KLoc.UL

    css = Css()

    css += CssChunk(
        HtmlClassesAll([kv_name, 'container']),
        {
            'display': 'grid',
            'grid-template-columns': ' '.join(['auto'] * kloc.num_cols),
            'grid-template-rows': ' '.join(['auto'] * kloc.num_rows),
        },
    )

    css += CssChunk(
        HtmlClassesAll([kv_name, 'key']),
        {
            'grid-column-start': kloc.key_col,
            'grid-row-start': kloc.key_row,
        },
    )

    css += CssChunk(
        HtmlClassesAll([kv_name, 'value']),
        {
            'grid-column-start': kloc.value_col,
            'grid-row-start': kloc.value_row,
        },
    )

    print('')
    print("======")
    print(" css:")
    print("======")
    print(css)
