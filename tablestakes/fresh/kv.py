import enum
from typing import *

from tablestakes.fresh import utils, html_css
from tablestakes.fresh.creators import ChoiceCreator, MultiCreator, AddressCreator, KvCssCreator, KeyValueCreator

KEY_HTML_CLASS = 'key'
VALUE_HTML_CLASS = 'value'
CONTAINER_HTML_CLASS = 'container'


ProbDict = Dict[Any, float]


def get_container_selector(kv_name: str):
    return html_css.HtmlClassesAll([kv_name, CONTAINER_HTML_CLASS])


def get_key_selector(kv_name: str):
    return html_css.HtmlClassesAll([kv_name, KEY_HTML_CLASS])


def get_value_selector(kv_name: str):
    return html_css.HtmlClassesAll([kv_name, VALUE_HTML_CLASS])


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
            kv_name: str,
            k_contents: html_css.DirtyHtmlChunk,
            v_contents: html_css.DirtyHtmlChunk,
    ) -> 'KVHtml':
        return cls(
            container_classes=get_container_selector(kv_name),
            k_tag=html_css.Div(k_contents, classes=get_key_selector(kv_name)),
            v_tag=html_css.Div(v_contents, classes=get_value_selector(kv_name)),
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
    def num_cols(self) -> int:
        return max(self.key_col, self.value_col)

    @property
    def num_rows(self) -> int:
        return max(self.key_row, self.value_row)

    def get_css(self, kv_name: str) -> html_css.Css:
        kv_css = html_css.Css()

        kv_css += html_css.CssChunk(
            get_container_selector(kv_name),
            {
                'display': 'grid',
                'grid-template-columns': ' '.join(['auto'] * kloc.num_cols),
                'grid-template-rows': ' '.join(['auto'] * kloc.num_rows),
            },
        )

        kv_css += html_css.CssChunk(
            get_key_selector(kv_name),
            {
                'grid-column-start': kloc.key_col,
                'grid-row-start': kloc.key_row,
            },
        )

        kv_css += html_css.CssChunk(
            get_value_selector(kv_name),
            {
                'grid-column-start': kloc.value_col,
                'grid-row-start': kloc.value_row,
            },
        )

        return kv_css


class KvCss(html_css.Css):
    def __init__(
            self,
            kv_name: str,
            key_css_values: utils.StrDict,
            value_css_values: utils.StrDict,
            container_css_values: utils.StrDict,
    ):
        super().__init__(chunks=[
            html_css.CssChunk(get_container_selector(kv_name), container_css_values),
            html_css.CssChunk(get_key_selector(kv_name), key_css_values),
            html_css.CssChunk(get_value_selector(kv_name), value_css_values),
        ])


if __name__ == '__main__':
    # make a style generator that uses the KLoc CSS and adds extra styling to it

    # make a kv creator loop over KVConfigs

    # make a css grid
    # pack css grid with KVs

    # kv_css = html_css.Css([
    #     html_css.CssChunk(html_css.HtmlClass('asdf'), {'style': 'grid'}),
    #     html_css.CssChunk(html_css.HtmlClass('2qwer'), {'style2': 'grid2'}),
    # ])
    # for chunk in kv_css:
    #     print(chunk)

    position_probabilities = {
        KLoc.UL: 0.3,
        KLoc.U:  1.0,
        KLoc.UR: 0.01,
        KLoc.R:  0.1,
        KLoc.BR: 0.01,
        KLoc.B:  0.2,
        KLoc.BL: 0.01,
        KLoc.L:  1.0,
    }


    KeyValueCreator(
        name='receiving_address',
        key_contents_creator=MultiCreator([
            ChoiceCreator(['Receiving', 'Receiving Address', 'Address To', 'To']),
            ChoiceCreator([':', '']),
        ]),
        value_contents_creator=AddressCreator(),
        style_creator=KvCssCreator(
        ),
    )


    kvh = KVHtml.from_strs('address_to', 'Address To:', '1232 Apache Ave </br>Santa Fe, NM 87505')

    print(kvh)
    print(kvh.get_container_classes())
    print(kvh.get_key_classes())
    print(kvh.get_value_classes())

    kv_name = 'address_to'
    kloc = KLoc.UL

    print('')
    print("======")
    print(" css:")
    print("======")
    print(kloc.get_css(kv_name))
