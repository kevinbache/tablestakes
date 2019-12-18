import enum
from typing import *

from tablestakes.fresh import utils, html_css


CONTAINER_HTML_CLASS = 'kv_container'
KEY_HTML_CLASS = 'kv_key'
VALUE_HTML_CLASS = 'kv_value'


ProbDict = Dict[Any, float]


def get_all_classes_container_selector(other_class_name: Optional[str] = None):
    classes = []
    if other_class_name is not None:
        classes.append(other_class_name)
    classes.append(CONTAINER_HTML_CLASS)
    return html_css.HtmlClassesAll(classes)


def get_all_classes_key_selector(kv_name: Optional[str] = None):
    classes = []
    if kv_name is not None:
        classes.append(kv_name)
    classes.append(KEY_HTML_CLASS)
    return html_css.HtmlClassesAll(classes)


def get_all_classes_value_selector(kv_name: Optional[str] = None):
    classes = []
    if kv_name is not None:
        classes.append(kv_name)
    classes.append(VALUE_HTML_CLASS)
    return html_css.HtmlClassesAll(classes)


class KVHtml(html_css.Div):
    def __init__(
            self,
            k_tag: html_css.HtmlTag,
            v_tag: html_css.HtmlTag,
            container_classes: html_css.HtmlClassesType,
            container_attributes: Optional[utils.StrDict] = None,
    ):
        super().__init__([k_tag, v_tag], container_classes, container_attributes)
        self._k_tag = k_tag
        self._v_tag = v_tag

    @classmethod
    def from_strs(
            cls,
            k_contents: html_css.DirtyHtmlChunk,
            v_contents: html_css.DirtyHtmlChunk,
            kv_name: Optional[str] = None,
    ) -> 'KVHtml':
        return cls(
            container_classes=get_all_classes_container_selector(kv_name),
            k_tag=html_css.Div(k_contents, classes=get_all_classes_key_selector(kv_name)),
            v_tag=html_css.Div(v_contents, classes=get_all_classes_value_selector(kv_name)),
        )

    def get_container_class_list(self):
        return self.get_class_list()

    def get_key_class_list(self):
        return self._k_tag.get_class_list()

    def get_value_class_list(self):
        return self._v_tag.get_class_list()


class KLoc(enum.Enum):
    # names are where the key is relative to the value
    # start in the upper left left, go clockwise
    # UL: Label is in the upper left.
    #    Key   col, row    Value col,  row
    AL =       ((1,   1),         (2,    2))
    A =        ((1,   1),         (1,    2))
    AR =       ((2,   1),         (1,    2))
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
    def num_cols(self) -> int:
        return max(self.key_col, self.value_col)

    @property
    def num_rows(self) -> int:
        return max(self.key_row, self.value_row)

    def get_css(
            self,
            kv_name: Optional[str] = None,
            containing_class: Optional[Union[html_css.HtmlClass, str]] = None,
    ) -> html_css.Css:
        kv_css = html_css.Css()

        # TODO: clean this selector creation business up
        if kv_name is not None and containing_class is not None:
            raise ValueError(f'Only one of kv_name and containing_class may be set')
        elif containing_class is not None:
            if isinstance(containing_class, html_css.HtmlClass):
                class_name = containing_class.name
            elif isinstance(containing_class, str):
                class_name = containing_class
            else:
                raise ValueError(f'containing_class should be either an HtmlClass or str.  '
                                 f'Got {type(containing_class)}')
            container_selector = html_css.HtmlClassesNested([class_name, CONTAINER_HTML_CLASS])
            key_selector = html_css.HtmlClassesNested([class_name, KEY_HTML_CLASS])
            value_selector = html_css.HtmlClassesNested([class_name, VALUE_HTML_CLASS])
        else:
            # still fine if kv_name is None
            container_selector = get_all_classes_container_selector(kv_name)
            key_selector = get_all_classes_key_selector(kv_name)
            value_selector = get_all_classes_value_selector(kv_name)

        kv_css.add_style(html_css.CssChunk(
            container_selector,
            {
                'display': 'grid',
                'grid-template-columns': ' '.join(['auto'] * self.num_cols),
                'grid-template-rows': ' '.join(['auto'] * self.num_rows),
            },
        ))

        kv_css.add_style(html_css.CssChunk(
            key_selector,
            {
                'grid-column-start': self.key_col,
                'grid-row-start': self.key_row,
            },
        ))

        kv_css.add_style(html_css.CssChunk(
            value_selector,
            {
                'grid-column-start': self.value_col,
                'grid-row-start': self.value_row,
            },
        ))

        return kv_css


