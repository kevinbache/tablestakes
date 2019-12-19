import abc
import enum
from typing import *

import numpy as np

from tablestakes.fresh import kv, utils, chunks, creators
from tablestakes.fresh import html_css as hc


class SelectorType(enum.Enum):
    GROUP_CONTAINER = 1
    KV_CONTAINER = 2
    KV_KEY = 3
    KV_VALUE = 4


class KvAlign(enum.Enum):
    LL = ('left', 'left')
    LR = ('left', 'right')
    CC = ('center', 'center')

    def __init__(self, key_alignment: str, value_alignment: str):
        self.key_alignment = key_alignment
        self.value_alignment = value_alignment


class KvGroup(hc.StyledHtmlTag, abc.ABC):
    def __init__(self, klass: Union[hc.HtmlClass, str]):
        super().__init__(hc.Div('', classes=[klass]), hc.Css())
        self.klass = klass
        self.kvs = []

    # TODO: reduce hc.StyledHtmlTag.add_both vs add_kv apis
    def add_kv(self, kv: hc.StyledHtmlTag):
        self.kvs.append(kv)

    def get_selector(self, type: SelectorType) -> hc.CssSelector:
        if type == SelectorType.GROUP_CONTAINER:
            return hc.HtmlClass(self.klass)
        elif type == SelectorType.KV_CONTAINER:
            return hc.HtmlClassesNested([self.klass, kv.CONTAINER_HTML_CLASS])
        elif type == SelectorType.KV_KEY:
            return hc.HtmlClassesNested([self.klass, kv.KEY_HTML_CLASS])
        elif type == SelectorType.KV_VALUE:
            return hc.HtmlClassesNested([self.klass, kv.VALUE_HTML_CLASS])
        else:
            raise ValueError(f'Got unexpected container SelectorType: {type}')

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

    def set_key_font_style(self, style: str = 'italics', selector_type=SelectorType.KV_KEY):
        self.css.add_style(
            hc.CssChunk(self.get_selector(selector_type), {
                'font-style': style,
            })
        )

    def set_font_weight(self, weight: str = 'bold', selector_type=SelectorType.KV_KEY):
        self.css.add_style(
            hc.CssChunk(self.get_selector(selector_type), {
                'font-weight': weight,
            })
        )

    def do_add_colon_to_keys(self):
        self.css.add_style(
            hc.CssChunk(f'{self.get_selector(SelectorType.KV_KEY).to_selector_str()}:after', {
                'content': "':'",
            }),
        )


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

        # self.css.add_style(kv.KLoc.A.get_css())

        self.css.add_style(hc.Css([
            # group container styles
            hc.CssChunk(self.get_selector(SelectorType.GROUP_CONTAINER), {
                'display': 'grid',
                'grid-template-columns': 'auto auto auto auto auto',
                # 'grid-gap': '5px 10px',  # vert, horz
            }),

            # # kv key styles
            # hc.CssChunk(self.get_selector(SelectorType.KV_CONTAINER), {
            #     'border-style': 'solid',
            #     'border-width': '1px',
            # }),

            # kv key styles
            hc.CssChunk(self.get_selector(SelectorType.KV_KEY), {
                'border-style': 'solid',
                'border-width': '1px',
            }),

            # # kv value styles
            # hc.CssChunk(self.get_selector(SelectorType.KV_VALUE), {
            #     'border-style': 'solid',
            #     'border-width': '1px',
            # }),
        ]))

    def get_html(self):
        hc.Table([
            hc.Tr([kv.k for kv in self.kvs]),
            hc.Tr([]),
        ])


if __name__ == '__main__':
    np.random.seed(42)

    my_date_creator = creators.DateCreator()
    kv_creators = [
        creators.KeyValueCreator(
            name='to_address',
            key_contents_creator=creators.ChoiceCreator(['Receiving', 'Receiving Address', 'Sent To', 'To']),
            value_contents_creator=creators.AddressCreator(),
        ),
        creators.KeyValueCreator(
            name='sale_address',
            key_contents_creator=creators.ChoiceCreator(['Sale Address', 'Sold To']),
            value_contents_creator=creators.AddressCreator(),
        ),
        creators.KeyValueCreator(
            name='from_address',
            key_contents_creator=creators.ChoiceCreator(['Shipping', 'Shipping Address', 'From', 'Address From']),
            value_contents_creator=creators.AddressCreator(),
        ),
        creators.KeyValueCreator(
            name='date_sent',
            key_contents_creator=creators.ChoiceCreator(['Sent', 'Date Sent', 'Statement Date']),
            value_contents_creator=my_date_creator,
        ),
        creators.KeyValueCreator(
            name='date_received',
            key_contents_creator=creators.ChoiceCreator(['Received', 'Date Received']),
            value_contents_creator=my_date_creator,
        ),
        creators.KeyValueCreator(
            name='invoice_number',
            key_contents_creator=creators.ChoiceCreator(['Invoice', 'Invoice number', 'Account']),
            value_contents_creator=creators.IntCreator(),
        ),
    ]

    # group = LColonKvGroup('maingroup')
    group = ABoxKvGroup('maingroup')
    for kvc in kv_creators:
        group.add_both(*kvc())

    group.set_font_weight()
    group.do_add_colon_to_keys()

    doc = hc.Document()
    doc.add_styled_html(group)
    hc.open_html_str(str(doc), do_print_too=True)
