from tablestakes import kv_styles, creators, utils, html_css as hc
from tablestakes.html_css import SelectorType


if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)

    with utils.Timer('hc creation'):
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

        # group = LColonKvGroup('lcolon_group')
        # group.set_font_weight()
        # group.do_add_colon_to_keys()
        # group.set_kv_horz_alignment(KvAlign.LR)
        # # group.set_bg_color(selector_type=SelectorType.KV_KEY)
        # # group.set_color(selector_type=SelectorType.KV_KEY)
        # group.set_kv_horz_alignment(KvAlign.RL)

        # group = ATableKvGroup('atable_group')
        # group.set_invisible_border()
        # group.set_font_weight()
        # group.do_add_colon_to_keys()
        # group.set_kv_horz_alignment(KvAlign.CL)
        # group.set_kv_vert_alignment(KvAlign.TT)
        # group.set_padding('4px')
        # group.set_bg_color()
        # group.set_color()
        # group.set_font_family('Verdana, Arial, Helvetica, sans-serif;', SelectorType.GROUP)
        # group.set_text_transform()

        group = kv_styles.LTableKvGroup('ltable_group')
        group.set_invisible_border()
        group.set_font_weight()
        group.do_add_colon_to_keys()
        group.set_kv_horz_alignment(kv_styles.KvAlign.LL)
        group.set_kv_vert_alignment(kv_styles.KvAlign.TT)
        group.set_padding('4px')
        group.set_font_family('Verdana, Arial, Helvetica, sans-serif;', SelectorType.GROUP)
        group.set_text_transform()

        for kvc in kv_creators:
            group.add_both(*kvc())

        doc = hc.Document()
        doc.add_styled_html(group)
        hc.open_html_str(str(doc), do_print_too=True)

        doc.save_pdf('generated.pdf')
        print('done')
