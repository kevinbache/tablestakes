import abc
import enum
from typing import *

from tablestakes.create_fake_data import kv, html_css as hc
from tablestakes.create_fake_data.html_css import SelectorType


class KvAlign(enum.Enum):
    LL = ('left', 'left')
    LR = ('left', 'right')
    RL = ('right', 'left')
    CC = ('center', 'center')
    CL = ('center', 'left')
    LC = ('left', 'center')
    TB = ('flex-start', 'flex-end')
    BB = ('flex-end', 'flex-end')
    TT = ('flex-start', 'flex-start')

    def __init__(self, key_alignment: str, value_alignment: str):
        self.key_alignment = key_alignment
        self.value_alignment = value_alignment

    def __repr__(self):
        return f'{self.__class__.__name__}({self.key_alignment}, {self.value_alignment})'


class KvGroup(hc.StyledHtmlTag, abc.ABC):
    def __init__(self, klass: Union[hc.HtmlClass, str]):
        super().__init__(hc.Div('', classes=[klass]), hc.Css())
        self.klass = klass
        self.kvs = []

    def get_selector(self, selector_type: SelectorType) -> hc.CssSelector:
        if selector_type == SelectorType.GROUP:
            return hc.HtmlClass(self.klass)
        elif selector_type == SelectorType.TRS_IN_GROUP:
            return hc.CssSelector(f'.{self.klass} tr')
        elif selector_type == SelectorType.TDS_IN_GROUP:
            return hc.CssSelector(f'.{self.klass} td')
        elif selector_type in (
                SelectorType.KV_CONTAINER,
                SelectorType.KEY,
                SelectorType.VALUE,
                SelectorType.KEY_OUTER,
                SelectorType.VALUE_OUTER,
                SelectorType.TABLE_KEY_HOLDER,
                SelectorType.TABLE_VALUE_HOLDER
        ):
            return hc.HtmlClassesNested([self.klass, selector_type.html_class_name])
        else:
            raise ValueError(f'Got unexpected container SelectorType: {selector_type}')

    def set_kv_horz_alignment(self, alignment: KvAlign):
        self.css.add_style(hc.Css([
            hc.CssChunk(self.get_selector(SelectorType.KEY), {
                'text-align': alignment.key_alignment,
            }),
            hc.CssChunk(self.get_selector(SelectorType.VALUE), {
                'text-align': alignment.value_alignment,
            }),
        ]))

    def set_kv_vert_alignment(self, alignment: KvAlign):
        self.css.add_style(hc.Css([
            hc.CssChunk(self.get_selector(SelectorType.KEY_OUTER), {
                'display': 'flex',
                'flex-direction': 'column',
                'justify-content': alignment.key_alignment,
                'height': '100%',
            }),
            hc.CssChunk(self.get_selector(SelectorType.VALUE_OUTER), {
                'display': 'flex',
                'flex-direction': 'column',
                'justify-content': alignment.value_alignment,
                'height': '100%',
            }),
        ]))

    def do_add_colon_to_keys(self):
        self.css.add_style(
            hc.CssChunk(f'{self.get_selector(SelectorType.KEY).to_selector_str()}:after', {
                'content': "':'",
            }),
        )

    def set_css_property(self, property: str, value: Optional[str], selector_type=SelectorType.KEY):
        if value is None:
            return
        self.css.add_style(
            hc.CssChunk(self.get_selector(selector_type), {
                property: value,
            })
        )

    def set_font_family(self, value: Optional[str] = '"Times New Roman", Times, serif', selector_type=SelectorType.KEY):
        self.set_css_property('font-family', value, selector_type)

    def set_font_style(self, value: Optional[str] = 'italics', selector_type=SelectorType.KEY):
        self.set_css_property('font-style', value, selector_type)

    def set_font_size(self, value: Optional[str] = '1em', selector_type=SelectorType.KEY):
        self.set_css_property('font-size', value, selector_type)

    def set_font_weight(self, value: Optional[str] = 'bold', selector_type=SelectorType.KEY):
        self.set_css_property('font-weight', value, selector_type)

    def set_text_transform(self, value: Optional[str] = 'uppercase', selector_type: SelectorType = SelectorType.KEY):
        self.set_css_property('text-transform', value, selector_type)

    def set_bg_color(self, value: Optional[str] = '#333333', selector_type: SelectorType = SelectorType.TABLE_KEY_HOLDER):
        self.set_css_property('background-color', value, selector_type)

    def set_color(self, value: Optional[str] = '#ffffff', selector_type: SelectorType = SelectorType.TABLE_KEY_HOLDER):
        self.set_css_property('color', value, selector_type)

    def set_padding(self, value: Optional[str] = '1px', selector_type: SelectorType = SelectorType.TDS_IN_GROUP):
        self.set_css_property('padding', value, selector_type)

    def set_position(self, value: Optional[str] = 'relative', selector_type: SelectorType = SelectorType.GROUP):
        self.set_css_property('position', value, selector_type)

    def set_left(self, value: Optional[str] = '30px', selector_type: SelectorType = SelectorType.GROUP):
        self.set_css_property('left', value, selector_type)

    def set_top(self, value: Optional[str] = '30px', selector_type: SelectorType = SelectorType.GROUP):
        self.set_css_property('top', value, selector_type)


class LColonKvGroup(KvGroup):
    def __init__(self, klass: Union[hc.HtmlClass, str]):
        super().__init__(klass)

        self.css.add_style(kv.KLoc.L.get_css())

        self.css.add_style(hc.Css([
            # group container styles
            hc.CssChunk(self.get_selector(SelectorType.GROUP), {
                'display': 'grid',
                'grid-template-columns': 'auto',
                'grid-gap': '5px 10px',  # vert, horz
            }),

            # kv key styles
            hc.CssChunk(self.get_selector(SelectorType.KEY), {
                'text-align': 'left',
            }),

            # kv value styles
            hc.CssChunk(self.get_selector(SelectorType.VALUE), {
                'text-align': 'right',
            }),
        ]))


class TableKvGroup(KvGroup, abc.ABC):
    def __init__(self, klass: Union[hc.HtmlClass, str]):
        super().__init__(klass)
        self.html_chunks = []

        css = hc.Css([
            hc.CssChunk(self.get_selector(SelectorType.GROUP), {
                'border-collapse': 'collapse',
                'height': '1px',
            }),
            hc.CssChunk(self.get_selector(SelectorType.TDS_IN_GROUP), {
                'border-style': 'solid',
                'border-width': '1px',
                'padding': '1px',
                'height': '100%',
            }),
            hc.CssChunk(self.get_selector(SelectorType.TRS_IN_GROUP), {
                'height': '100%',
            }),
        ])
        self.add_style(css)

    @abc.abstractmethod
    def get_html(self):
        pass

    def add_contents(self, html_chunk: hc.DirtyHtmlChunk):
        raise ValueError(f"This method isn't valid for this class "
                         f"(because it makes a table rather than just dumping all "
                         f"the kvs in a <div> tag)")

    def add_both(self, html_chunk: kv.KvHtml, css: hc.Css):
        self.html_chunks.append(html_chunk)
        self.add_style(css)

    def set_border_style(self, value: Optional[str] = 'solid', selector_type=SelectorType.TDS_IN_GROUP):
        self.set_css_property('border-style', value, selector_type)

    def set_border_width(self, value: Optional[str] = '1px', selector_type=SelectorType.TDS_IN_GROUP):
        self.set_css_property('border-style', value, selector_type)

    def set_width(self, value: Optional[str] = '4in', selector_type=SelectorType.GROUP):
        self.set_css_property('width', value, selector_type)

    def set_height(self, value: Optional[str] = '7in', selector_type=SelectorType.GROUP):
        self.set_css_property('height', value, selector_type)

    def set_invisible_border(self):
        self.set_border_style('none', selector_type=SelectorType.TDS_IN_GROUP)
        self.set_border_style('none', selector_type=SelectorType.TRS_IN_GROUP)
        self.set_border_width('0px', selector_type=SelectorType.TDS_IN_GROUP)
        self.set_border_width('0px', selector_type=SelectorType.TRS_IN_GROUP)


class ATableKvGroup(TableKvGroup):
    def get_html(self):
        return str(hc.Table(
            contents=[
                hc.Tr(
                    contents=[hc.Td(kv.get_key_tag()) for kv in self.html_chunks],
                    classes=[SelectorType.TABLE_KEY_HOLDER.html_class_name],
                ),
                hc.Tr(
                    contents=[hc.Td(kv.get_value_tag()) for kv in self.html_chunks],
                    classes=[SelectorType.TABLE_VALUE_HOLDER.html_class_name],
                ),
            ],
            classes=[self.klass],
        ))


class LTableKvGroup(TableKvGroup):
    def get_html(self):
        return str(hc.Table(
            contents=[
                hc.Tr(
                    contents=[
                        hc.Td(
                            contents=[kv_chunk.get_key_tag()],
                            classes=[SelectorType.TABLE_KEY_HOLDER.html_class_name]
                        ),
                        hc.Td(
                            contents=[kv_chunk.get_value_tag()],
                            classes=[SelectorType.TABLE_VALUE_HOLDER.html_class_name]
                        ),
                    ],
                    classes=[SelectorType.KV_CONTAINER.html_class_name],
                ) for kv_chunk in self.html_chunks
            ],
            classes=[self.klass],
        ))
