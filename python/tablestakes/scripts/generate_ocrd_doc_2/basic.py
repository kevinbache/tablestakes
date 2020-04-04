from tablestakes import creators, kv_styles, html_css as hc


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

group = kv_styles.LTableKvGroup('ltable_group')
group.set_invisible_border()
group.set_font_weight()
# group.do_add_colon_to_keys()
group.set_kv_horz_alignment(kv_styles.KvAlign.LL)
group.set_kv_vert_alignment(kv_styles.KvAlign.TT)
group.set_padding('4px')
group.set_font_family('Verdana, Arial, Helvetica, sans-serif;', hc.SelectorType.GROUP)
group.set_text_transform()

for kvc in kv_creators:
    group.add_both(*kvc())

doc = hc.Document()
doc.add_styled_html(group)

