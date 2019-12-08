import enum
from typing import Optional, Tuple

from tablestakes.fresh import utils, constants, html_css

KEY_HTML_CLASS = 'key'
VALUE_HTML_CLASS = 'value'
CONTAINER_HTML_CLASS = 'container'


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

    @staticmethod
    def _get_container_classes(kv_name: str):
        return html_css.HtmlClassesAll([kv_name, CONTAINER_HTML_CLASS])

    @staticmethod
    def _get_key_classes(kv_name: str):
        return html_css.HtmlClassesAll([kv_name, KEY_HTML_CLASS])

    @staticmethod
    def _get_value_classes(kv_name: str):
        return html_css.HtmlClassesAll([kv_name, VALUE_HTML_CLASS])

    @classmethod
    def from_strs(
            cls,
            kv_name: str,
            k_contents: html_css.DirtyHtmlChunk,
            v_contents: html_css.DirtyHtmlChunk,
    ) -> 'KVHtml':
        return cls(
            container_classes=cls._get_container_classes(kv_name),
            k_tag=html_css.Div(k_contents, classes=cls._get_key_classes(kv_name)),
            v_tag=html_css.Div(v_contents, classes=cls._get_value_classes(kv_name)),
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


class KvCss:
    def __init__(self):
        pass

    def create(self, kv_name: str, kloc: KLoc):
        # kv_name = 'address_to'
        # kloc = KLoc.UL

        kv_css = html_css.Css()

        kv_css += html_css.CssChunk(
            html_css.HtmlClassesAll([kv_name, self.CONTAINER_CLASS]),
            {
                'display': 'grid',
                'grid-template-columns': ' '.join(['auto'] * kloc.num_cols),
                'grid-template-rows': ' '.join(['auto'] * kloc.num_rows),
            },
        )

        kv_css += html_css.CssChunk(
            html_css.HtmlClassesAll([kv_name, self.KEY_CLASS]),
            {
                'grid-column-start': kloc.key_col,
                'grid-row-start': kloc.key_row,
            },
        )

        kv_css += html_css.CssChunk(
            html_css.HtmlClassesAll([kv_name, self.VALUE_CLASS]),
            {
                'grid-column-start': kloc.value_col,
                'grid-row-start': kloc.value_row,
            },
        )

        return kv_css


if __name__ == '__main__':
    kv_css = html_css.Css([
        html_css.CssChunk(html_css.HtmlClass('asdf'), {'style': 'grid'}),
        html_css.CssChunk(html_css.HtmlClass('2qwer'), {'style2': 'grid2'}),
    ])
    for chunk in kv_css:
        print(chunk)

    kvh = KVHtml.from_strs('address_to', 'Address To:', '1232 Apache Ave </br>Santa Fe, NM 87505')

    print(kvh)
    print(kvh.get_container_classes())
    print(kvh.get_key_classes())
    print(kvh.get_value_classes())

    kv_name = 'address_to'
    kloc = KLoc.UL

    kv_css = html_css.Css()

    kv_css += html_css.CssChunk(
        html_css.HtmlClassesAll([kv_name, constants.CONTAINER_CLASS]),
        {
            'display': 'grid',
            'grid-template-columns': ' '.join(['auto'] * kloc.num_cols),
            'grid-template-rows': ' '.join(['auto'] * kloc.num_rows),
        },
    )

    kv_css += html_css.CssChunk(
        html_css.HtmlClassesAll([kv_name, constants.KEY_CLASS]),
        {
            'grid-column-start': kloc.key_col,
            'grid-row-start': kloc.key_row,
        },
    )

    kv_css += html_css.CssChunk(
        html_css.HtmlClassesAll([kv_name, constants.VALUE_CLASS]),
        {
            'grid-column-start': kloc.value_col,
            'grid-row-start': kloc.value_row,
        },
    )

    print('')
    print("======")
    print(" css:")
    print("======")
    print(kv_css)
