import abc
import enum
from typing import *

import numpy as np

from tablestakes.fresh import kv, utils, chunks, creators
from tablestakes.fresh import html_css as hc


class SelectorType(enum.Enum):
    GROUP_CONTAINER = 1
    TRS_IN_GROUP = 2
    TDS_IN_GROUP = 3

    KV_CONTAINER = 4
    KV_KEY = 5
    KV_VALUE = 6

    KEY_GROUP = 7
    VALUE_GROUP = 8


KEY_GROUP_HTML_CLASS = 'key_group'
VALUE_GROUP_HTML_CLASS = 'value_group'


class KvAlign(enum.Enum):
    LL = ('left', 'left')
    LR = ('left', 'right')
    CC = ('center', 'center')
    CL = ('center', 'left')

    def __init__(self, key_alignment: str, value_alignment: str):
        self.key_alignment = key_alignment
        self.value_alignment = value_alignment


class KvGroup(hc.StyledHtmlTag, abc.ABC):
    def __init__(self, klass: Union[hc.HtmlClass, str]):
        super().__init__(hc.Div('', classes=[klass]), hc.Css())
        self.klass = klass
        self.kvs = []

    def get_selector(self, selector_type: SelectorType) -> hc.CssSelector:
        if selector_type == SelectorType.GROUP_CONTAINER:
            return hc.HtmlClass(self.klass)
        elif selector_type == SelectorType.KV_CONTAINER:
            return hc.HtmlClassesNested([self.klass, kv.CONTAINER_HTML_CLASS])
        elif selector_type == SelectorType.KV_KEY:
            return hc.HtmlClassesNested([self.klass, kv.KEY_HTML_CLASS])
        elif selector_type == SelectorType.KV_VALUE:
            return hc.HtmlClassesNested([self.klass, kv.VALUE_HTML_CLASS])
        else:
            raise ValueError(f'Got unexpected container SelectorType: {selector_type}')

    def set_kv_horz_alignment(self, alignment: KvAlign):
        self.css.add_style(hc.Css([
            hc.CssChunk(self.get_selector(SelectorType.KV_KEY), {
                'text-align': alignment.key_alignment,
            }),
            hc.CssChunk(self.get_selector(SelectorType.KV_VALUE), {
                'text-align': alignment.value_alignment,
            }),
        ]))

    def set_kv_vert_alignment(self, alignment: KvAlign):
        self.css.add_style(hc.Css([
            hc.CssChunk(self.get_selector(SelectorType.KV_KEY), {
                'display': 'flex',
                'align-items': alignment.key_alignment,
            }),
            hc.CssChunk(self.get_selector(SelectorType.KV_VALUE), {
                'display': 'flex',
                'align-items': alignment.value_alignment,
            }),
        ]))

    def do_add_colon_to_keys(self):
        self.css.add_style(
            hc.CssChunk(f'{self.get_selector(SelectorType.KV_KEY).to_selector_str()}:after', {
                'content': "':'",
            }),
        )

    def set_css_property(self, property: str, value: str, selector_type=SelectorType.KV_KEY):
        self.css.add_style(
            hc.CssChunk(self.get_selector(selector_type), {
                property: value,
            })
        )

    def set_font_family(self, value: str = '"Times New Roman", Times, serif', selector_type=SelectorType.KV_KEY):
        self.set_css_property('font-family', value, selector_type)

    def set_font_style(self, value: str = 'italics', selector_type=SelectorType.KV_KEY):
        self.set_css_property('font-style', value, selector_type)

    def set_font_weight(self, value: str = 'bold', selector_type=SelectorType.KV_KEY):
        self.set_css_property('font-weight', value, selector_type)

    def set_text_transform(self, value: str = 'uppercase', selector_type: SelectorType = SelectorType.KV_KEY):
        self.set_css_property('text-transform', value, selector_type)

    def set_bg_color(self, value: str = '#333333', selector_type: SelectorType = SelectorType.KEY_GROUP):
        self.set_css_property('background-color', value, selector_type)

    def set_color(self, value: str = '#ffffff', selector_type: SelectorType = SelectorType.KEY_GROUP):
        self.set_css_property('color', value, selector_type)


class LColonKvGroup(KvGroup):
    def __init__(self, klass: Union[hc.HtmlClass, str]):
        super().__init__(klass)

        self.css.add_style(kv.KLoc.L.get_css())

        self.css.add_style(hc.Css([
            # group container styles
            hc.CssChunk(self.get_selector(SelectorType.GROUP_CONTAINER), {
                'display': 'grid',
                'grid-template-columns': 'auto',
                'grid-gap': '5px 10px',  # vert, horz
            }),

            # kv key styles
            hc.CssChunk(self.get_selector(SelectorType.KV_KEY), {
                'text-align': 'left',
            }),

            # kv value styles
            hc.CssChunk(self.get_selector(SelectorType.KV_VALUE), {
                'text-align': 'right',
            }),
        ]))


class ATableKvGroup(KvGroup):
    def __init__(self, klass: Union[hc.HtmlClass, str]):
        super().__init__(klass)
        self.html_chunks = []

        self.css.add_style(hc.Css([
            hc.CssChunk(self.get_selector(SelectorType.GROUP_CONTAINER), {
                'border-collapse': 'collapse',
            }),

            hc.CssChunk(self.get_selector(SelectorType.TDS_IN_GROUP), {
                'border-style': 'solid',
                'border-width': '1px',
                'padding': '1px',
            }),
        ]))

    def get_selector(self, selector_type: SelectorType) -> hc.CssSelector:
        if selector_type == SelectorType.KEY_GROUP:
            return hc.HtmlClassesNested([self.klass, KEY_GROUP_HTML_CLASS])
        elif selector_type == SelectorType.TRS_IN_GROUP:
            return hc.CssSelector(f'.{self.klass} tr')
        elif selector_type == SelectorType.TDS_IN_GROUP:
            return hc.CssSelector(f'.{self.klass} td')
        elif selector_type == SelectorType.VALUE_GROUP:
            return hc.HtmlClassesNested([self.klass, VALUE_GROUP_HTML_CLASS])
        elif selector_type == SelectorType.VALUE_GROUP:
            return hc.HtmlClassesNested([self.klass, VALUE_GROUP_HTML_CLASS])
        else:
            return super().get_selector(selector_type)

    def get_html(self):
        return str(hc.Table(
            contents=[
                hc.Tr(
                    contents=[hc.Td(kv.get_key_tag()) for kv in self.html_chunks],
                    classes=[KEY_GROUP_HTML_CLASS],
                ),
                hc.Tr(
                    contents=[hc.Td(kv.get_value_tag()) for kv in self.html_chunks],
                    classes=[VALUE_GROUP_HTML_CLASS],
                ),
            ],
            classes=[self.klass],
        ))

    def add_contents(self, html_chunk: hc.DirtyHtmlChunk):
        raise ValueError(f"This method isn't value for this class "
                         f"(because it makes a table rather than just dumping all "
                         f"the kvs in a <div> tag)")

    def add_both(self, html_chunk: kv.KvHtml, css: hc.Css):
        self.html_chunks.append(html_chunk)
        self.add_style(css)

    def set_padding(self, amount='1px'):
        self.css.add_style(
            hc.CssChunk(self.get_selector(SelectorType.TDS_IN_GROUP), {
                'padding': amount,
            }),
        )


if __name__ == '__main__':
    np.random.seed(42)

    my_date_creator = creators.DateCreator()
    kv_creators = [
        creators.KvCreator(
            name='to_address',
            key_contents_creator=creators.ChoiceCreator(['Receiving', 'Receiving Address', 'Sent To', 'To']),
            value_contents_creator=creators.AddressCreator(),
        ),
        creators.KvCreator(
            name='sale_address',
            key_contents_creator=creators.ChoiceCreator(['Sale Address', 'Sold To']),
            value_contents_creator=creators.AddressCreator(),
        ),
        creators.KvCreator(
            name='from_address',
            key_contents_creator=creators.ChoiceCreator(['Shipping', 'Shipping Address', 'From', 'Address From']),
            value_contents_creator=creators.AddressCreator(),
        ),
        creators.KvCreator(
            name='date_sent',
            key_contents_creator=creators.ChoiceCreator(['Sent', 'Date Sent', 'Statement Date']),
            value_contents_creator=my_date_creator,
        ),
        creators.KvCreator(
            name='date_received',
            key_contents_creator=creators.ChoiceCreator(['Received', 'Date Received']),
            value_contents_creator=my_date_creator,
        ),
        creators.KvCreator(
            name='invoice_number',
            key_contents_creator=creators.ChoiceCreator(['Invoice', 'Invoice number', 'Account']),
            value_contents_creator=creators.IntCreator(),
        ),
    ]

    # group = LColonKvGroup('maingroup')
    # group.set_font_weight()
    # group.do_add_colon_to_keys()
    # group.set_kv_horz_alignment(KvAlign.LR)

    group = ATableKvGroup('maingroup')
    group.set_font_weight()
    group.do_add_colon_to_keys()
    group.set_kv_horz_alignment(KvAlign.CL)
    group.set_padding('4px')
    group.set_bg_color()
    group.set_color()

    for kvc in kv_creators:
        group.add_both(*kvc())

    doc = hc.Document()
    doc.add_styled_html(group)
    hc.open_html_str(str(doc), do_print_too=True)
