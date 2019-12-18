import abc
from typing import *

from tablestakes.fresh import kv, utils, chunks, creators
from tablestakes.fresh import html_css as hc

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

style_css = hc.Css()


class KvGroup(hc.StyledHtmlTag, abc.ABC):
    def __init__(self, klass: Union[hc.HtmlClass, str]):
        super().__init__(hc.Div('', classes=[klass]), hc.Css())
        self.klass = klass
        self.kvs = []

    def add_kv(self, kv: hc.StyledHtmlTag):
        self.kvs.append(kv)


class LColonKvGroup(KvGroup):
    def __init__(self, klass: Union[hc.HtmlClass, str]):
        super().__init__(klass)
        self.css.add_style(
            # group container styles
            hc.CssChunk(f'div.{self.klass}', {
                '': '',
            }),
            # container styles
            hc.CssChunk(hc.HtmlClassesNested([self.klass, kv.CONTAINER_HTML_CLASS]), {
                '': '',
            }),
            # key styles
            hc.CssChunk(hc.HtmlClassesNested([self.klass, kv.KEY_HTML_CLASS]), {
                '': '',
            }),
            # value styles
            hc.CssChunk(hc.HtmlClassesNested([self.klass, kv.VALUE_HTML_CLASS]), {
                '': '',
            }),
        )



# grid = hc.Grid(classes=['maingrid'], num_rows=4, num_cols=4)
# for kvc in kv_creators:
#     grid.add_both(*kvc())

doc = hc.Document()
doc.add_styled_html(group)
doc.add_style(style_css)


hc.open_html_str(str(doc), do_print_too=True)

