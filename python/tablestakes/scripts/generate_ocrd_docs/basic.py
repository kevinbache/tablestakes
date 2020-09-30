import random

from tablestakes import creators, kv_styles, html_css as hc, utils


def make_doc(seed: int, num_extra_fields=50, do_randomize_field_order=True):
    utils.set_seed(seed)

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
    for ind in range(num_extra_fields):
        kv_creators.append(
            creators.KvCreator(
                name=f'field_{ind}',
                key_contents_creator=creators.RandomStrCreator(min_words=1, max_words=2),
                value_contents_creator=creators.RandomStrCreator(min_words=1, max_words=10),
            )
        )

    group = kv_styles.LTableKvGroup('ltable_group')
    group.set_invisible_border()
    group.set_font_weight()
    # group.do_add_colon_to_keys()
    group.set_kv_horz_alignment(kv_styles.KvAlign.LL)
    group.set_kv_vert_alignment(kv_styles.KvAlign.TT)
    group.set_padding('4px')
    group.set_font_family('Verdana, Arial, Helvetica, sans-serif;', hc.SelectorType.GROUP)
    group.set_text_transform()

    if do_randomize_field_order:
        random.shuffle(kv_creators)

    for kvc in kv_creators:
        group.add_both(*kvc())

    doc = hc.Document()
    doc.add_styled_html(group)

    return doc