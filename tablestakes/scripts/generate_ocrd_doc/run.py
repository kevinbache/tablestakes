import pandas as pd

from tablestakes import kv_styles, creators, utils, html_css as hc, ocr, word_wrap
from tablestakes.html_css import SelectorType
from tablestakes.scripts.generate_ocrd_doc import sel_ocr_word_match


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
        # group.do_add_colon_to_keys()
        group.set_kv_horz_alignment(kv_styles.KvAlign.LL)
        group.set_kv_vert_alignment(kv_styles.KvAlign.TT)
        group.set_padding('4px')
        group.set_font_family('Verdana, Arial, Helvetica, sans-serif;', SelectorType.GROUP)
        group.set_text_transform()

        for kvc in kv_creators:
            group.add_both(*kvc())

        doc = hc.Document()
        doc.add_styled_html(group)


    # class NodeModifier:
    #     def __init__(self, find_str: str, fn: Callable):
    #         self.find_str = find_str
    #         self.fn = fn
    #
    #     def modify(self, root: etree._Element):
    #         # TODO
    #         return root
    #
    #     def modify_str(self, contents: str):
    #         root = etree.fromstring(text=contents, parser=etree.HTMLParser())
    #         root = self.modify(root)
    #         return etree.tostring(root, encoding='unicode')

    base_filename = 'doc_unwrapped'
    with utils.Timer('save unwrapped'):
        doc.save_html(f'{base_filename}.html')
        doc.save_pdf(f'{base_filename}.pdf')
        # doc.open_in_browser()

    with utils.Timer('wrap contents'):
        wrapper = word_wrap.WordWrapper()
        doc.contents = [wrapper.wrap_words_in_str(doc.contents[0])]

    base_filename = 'doc_wrapped'
    pdf_filename = f'{base_filename}.pdf'
    with utils.Timer('save wrapped'):
        doc.save_html(f'{base_filename}.html')
        doc.save_pdf(pdf_filename)
        # doc.open_in_browser()

    dpi = 500
    window_width_px = dpi * 8.5
    window_height_px = dpi * 11.00

    html_file = "file:///Users/kevin/projects/tablestakes/tablestakes/scripts/generate_ocrd_doc/doc_wrapped.html"
    ##############################################
    # use selenium to get locations of all words #
    ##############################################
    word_id_2_word = wrapper.get_used_id_to_word()
    sel_df = sel_ocr_word_match.get_word_pixel_locations(
        html_file,
        word_id_to_word=word_id_2_word,
        window_width_px=window_width_px,
        window_height_px=window_height_px,
    )
    sel_df.to_csv('selenium_word_locations.csv')
    print(sel_df)

    ####################################
    # ocr the previously saved pdf doc #
    ####################################
    ocr_raw_filename = 'doc_ocr_df.pkl'
    with utils.Timer('TesseractOcrProvider.ocr'):
        ocr_doc = ocr.TesseractOcrProvider().ocr(
            input_pdf=pdf_filename,
            dpi=dpi,
            save_raw_ocr_output_location=ocr_raw_filename,
        )
    print(ocr_doc)

    #########################################
    # create a df of the ocr word locations #
    #########################################
    ocrd_words = []
    for word in ocr_doc.get_words():
        ocrd_word = {
            'text': word.text,
            'bbox': word.bbox,
            'word_type': word.word_type,
        }
        ocrd_words.append(ocrd_word)

    ocr_df = pd.DataFrame(ocrd_words)
    ocr_df.to_csv('ocr_word_locations.csv')

    ############################################################################
    # match up one word by text between selenium and ocr versions of the words #
    # use that word to come up with a transformation for all of them           #
    ############################################################################
    srow = sel_df[sel_df['text'].str.lower() == 'shipping']
    orow = ocr_df[ocr_df['text'].str.lower() == 'shipping']

    sbbox = sel_ocr_word_match.selenium_row_to_bbox(srow)
    obbox = orow.bbox.values[0]

    x_line, y_line = sel_ocr_word_match.get_lines(obbox, sbbox)
    ocr_text_lowered = ocr_df['text'].str.lower()
    sel_df['closest_ocr_word_id'] = \
        sel_df['text'].apply(lambda stext: np.argmin([utils.levenshtein(stext.lower(), t) for t in ocr_text_lowered]))
    sel_df['ocr_bbox'] = ocr_df.loc[sel_df.closest_ocr_word_id].bbox.reset_index(drop=True)
    sel_df['ocr_text'] = ocr_df.loc[sel_df.closest_ocr_word_id].text.reset_index(drop=True)
    sel_df['sel_bbox'] = sel_df.apply(lambda row: sel_ocr_word_match.selenium_row_to_bbox(row), axis=1)
    sel_df['sel_bbox_converted'] = \
        sel_df.sel_bbox.apply(lambda sel_bbox: sel_ocr_word_match.convert_bbox(x_line, y_line, sel_bbox))

    pd.set_option('display.max_columns', 200)
    pd.set_option('display.width', 200)
    print(sel_df[['text', 'ocr_text', 'ocr_bbox', 'sel_bbox_converted', 'sel_bbox']])

    print()
    print(f'x_line: {x_line}')
    print(f'y_line: {y_line}')


    ############################################################################
    # since the words are already matched by text, figure out the statistical  #
    # version of the transformation and apply that.                            #
    # seems to be about 1/5 in off MAE                                         #
    # need to visualize to diagnose                                            #
    #   html                                                                   #
    #   background image for page                                              #
    #   boxes for each word.  how to place? y count up? down?                  #
    ############################################################################
    from sklearn.linear_model import LassoCV
    model = LassoCV(fit_intercept=True)

    sbox_array = np.array(list(sel_df['sel_bbox'].apply(lambda bbox: bbox.to_array())))
    obox_array = np.array(list(sel_df['ocr_bbox'].apply(lambda bbox: bbox.to_array())))

    for target_ind in range(4):
        x = sbox_array[:, target_ind:target_ind+1]
        # x = sbox_array
        y = obox_array[:, target_ind]
        model.fit(x, y)
        y_hat = model.predict(x)
        mae = np.mean(np.abs(y_hat - y))
        print(target_ind, model.intercept_, model.coef_, mae)

