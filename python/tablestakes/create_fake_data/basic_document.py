import random

from tablestakes import utils
from tablestakes.create_fake_data import kv_styles, html_css as hc, creators
from tablestakes.ml.hyperparams import DocGenParams


def make_doc(
        seed: int,
        doc_config: DocGenParams,
) -> hc.Document:
    utils.set_seeds(seed)

    doc_config = doc_config.sample()
    assert isinstance(doc_config, DocGenParams) # for pycharm autocomplete

    # create the complex creators up here so they'll use consistent formatting throughout.
    date_creator = creators.DateCreator()
    phone_creator = creators.PhoneCreator()
    dollars_creator = creators.DollarsCreator()

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
            value_contents_creator=date_creator,
        ),
        creators.KvCreator(
            name='date_received',
            key_contents_creator=creators.ChoiceCreator(['Received', 'Date Received']),
            value_contents_creator=date_creator,
        ),
        creators.KvCreator(
            name='invoice_number',
            key_contents_creator=creators.ChoiceCreator(['Invoice', 'Invoice number', 'Account']),
            value_contents_creator=creators.IntCreator(),
        ),
        creators.KvCreator(
            name='total',
            key_contents_creator=creators.ChoiceCreator([
                'Total',
                'Total Due',
                'Charges',
                'Total Charges',
                'Amount',
                'Amount Due',
                'Balance',
                'Balance Due',
                'After tax',
            ]),
            value_contents_creator=dollars_creator,
        ),
        creators.KvCreator(
            name='subtotal',
            key_contents_creator=creators.ChoiceCreator(['Subtotal', 'Subtotal Due', 'Sans Tax']),
            value_contents_creator=dollars_creator,
        ),
        creators.KvCreator(
            name='phone',
            key_contents_creator=creators.ChoiceCreator(['Phone', 'Phone Number', 'Phone No', 'Call']),
            value_contents_creator=phone_creator,
        ),
        creators.KvCreator(
            name='fax',
            key_contents_creator=creators.ChoiceCreator(['Fax', 'Fax Number']),
            value_contents_creator=phone_creator,
        ),
    ]
    for ind in range(doc_config.num_extra_fields):
        kv_creators.append(
            creators.KvCreator(
                name=f'field_{ind}',
                key_contents_creator=creators.RandomStrCreator(min_words=1, max_words=2),
                value_contents_creator=creators.RandomStrCreator(min_words=1, max_words=10),
            )
        )

    group = kv_styles.LTableKvGroup('ltable_group')
    if doc_config.do_set_invisible_border:
        group.set_invisible_border()
    else:
        group.set_border_width(value='1px', selector_type=hc.SelectorType.GROUP)

    font_size = doc_config.font_size_px
    group.set_font_size(f'{font_size}px', selector_type=hc.SelectorType.KEY)
    if random.random() < doc_config.do_regen_font_val_size:
        font_size = doc_config.val_font_size_px
    group.set_font_size(f'{font_size}px', selector_type=hc.SelectorType.VALUE)

    if doc_config.do_bold_keys:
        group.set_font_weight('bold', selector_type=hc.SelectorType.KEY)

    if doc_config.do_add_colon_to_keys:
        group.do_add_colon_to_keys()

    group.set_kv_horz_alignment(doc_config.horz_alignment)
    group.set_kv_vert_alignment(doc_config.vert_alignment)

    group.set_padding(f'{doc_config.table_cell_padding_px}px', selector_type=hc.SelectorType.TDS_IN_GROUP)

    group.set_font_family('Verdana, Arial, Helvetica, sans-serif;', hc.SelectorType.GROUP)
    group.set_text_transform()

    if doc_config.group_offset_in > 0:
        offset = doc_config.group_offset_in * 72
        group.set_position('relative')
        group.set_left(f'{offset}px')
        group.set_top(f'{offset}px')

    if doc_config.do_randomize_field_order:
        random.shuffle(kv_creators)

    for kvc in kv_creators:
        group.add_both(*kvc())

    doc = hc.Document()
    doc.add_styled_html(group)

    return doc


if __name__ == '__main__':
    seed = 42
    doc_config = DocGenParams()
    for i in range(10):
        doc = make_doc(seed + i, doc_config)
        doc.save_pdf(f'blah_{i}.pdf')
