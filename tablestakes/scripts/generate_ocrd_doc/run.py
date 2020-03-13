import re

from tablestakes import kv_styles, creators, utils, html_css as hc, ocr
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

    html_filename = 'doc.html'
    with utils.Timer('save html'):
        doc.save_html(html_filename)

    c = doc.contents[0]
    from lxml import etree

    word_id = 0
    doc = etree.fromstring(c, parser=etree.HTMLParser())
    re_white = re.compile(r'(\s+)')
    for tag in doc.iter():
        print(tag)
        if tag.text and tag.text.strip():
            words = re.split(re_white, tag.text.strip())
            annotated_words = []
            for word in words:
                if word.strip():
                    annotated_words.append(f'<w id=word{word_id}>{word}</w>')
                    word_id += 1
                else:
                    annotated_words.append(word)
                    continue
            tag.text = ''.join(annotated_words)
    new_contents = etree.tostring(doc, pretty_print=True)
    print(new_contents)


# https://stackoverflow.com/questions/44215381/wrap-text-within-element-lxml
# deal with el.text, then el.child.tail for each child

    ###########
    # KEEP ME #
    ###########
    # pdf_filename = 'doc.pdf'
    # dpi = 400
    # margins = '1in'
    # with utils.Timer('save pdf'):
    #     doc.save_pdf(pdf_filename)
    #     print('done')
    #
    # ocr_raw_filename = 'doc_ocr_df.pkl'
    # with utils.Timer('TesseractOcrProvider.ocr'):
    #     ocr_doc = ocr.TesseractOcrProvider().ocr(
    #         input_pdf=pdf_filename,
    #         dpi=dpi,
    #         save_raw_ocr_output_location=ocr_raw_filename,
    #     )
    ###########
    # KEEP ME #
    ###########
