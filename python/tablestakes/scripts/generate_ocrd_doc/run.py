from pathlib import Path

import numpy as np
import pandas as pd

from tablestakes import kv_styles, creators, utils, ocr, word_wrap
from tablestakes import html_css as hc
from tablestakes.html_css import SelectorType
from tablestakes.scripts.generate_ocrd_doc import sel_ocr_word_match


if __name__ == '__main__':
    ######################################################
    # randomly create html representing a print document #
    ######################################################
    np.random.seed(42)
    margin = '2in'

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

    base_filename = 'doc_unwrapped'
    with utils.Timer('save unwrapped'):
        doc.save_html(f'{base_filename}.html')
        doc.save_pdf(f'{base_filename}.pdf')
        # doc.open_in_browser()

    with utils.Timer('wrap contents'):
        wrapper = word_wrap.WordWrapper()
        doc.contents = [wrapper.wrap_words_in_str(doc.contents[0])]

    base_filename = 'doc_wrapped'
    html_filename = f'{base_filename}.html'
    pdf_filename = f'{base_filename}.pdf'
    with utils.Timer('save wrapped'):
        doc.save_html(html_filename)
        doc.save_pdf(pdf_filename, page_size=hc.PageSize.LETTER, margin=margin)
        # doc.open_in_browser()

    ##############################################
    # use selenium to get locations of all words #
    ##############################################
    dpi = 1000
    window_width_px = dpi * 8.5
    window_height_px = dpi * 11.00
    output_dir = Path('.') / 'docs' / 'doc_01'
    utils.mkdir_if_not_exist(output_dir)

    word_id_2_word = wrapper.get_used_id_to_word()
    sel_df = sel_ocr_word_match.get_word_pixel_locations(
        html_file=f'file:///Users/kevin/projects/tablestakes/python/tablestakes/scripts/generate_ocrd_doc/{html_filename}',
        word_id_to_word=word_id_2_word,
        window_width_px=window_width_px,
        window_height_px=window_height_px,
    )
    sel_df.to_csv(output_dir / 'sel_words.csv')
    print(sel_df)

    ####################
    # save page images #
    ####################
    page_images = ocr.OcrProvider.load_pdf_to_images(pdf_filename, dpi)
    for page_ind, page_image in enumerate(page_images):
        page_file = output_dir / utils.prepend_before_extension(pdf_filename, f'_page_{page_ind}', new_ext='.png')
        page_image.save(page_file)

    ####################################
    # ocr the previously saved pdf doc #
    ####################################
    ocr_raw_filename = 'doc_ocr_df.pkl'
    ocr_provider = ocr.TesseractOcrProvider()
    with utils.Timer('TesseractOcrProvider.ocr'):
        ocr_doc = ocr_provider.ocr(
            input_pdf=pdf_filename,
            dpi=dpi,
            save_raw_ocr_output_location=ocr_raw_filename,
        )
    words = ocr_doc.get_words()
    # print([w for w in words if not w.bbox.xmax - w.bbox.xmin])
    # print(ocr_doc)

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
    ocr_df.to_csv(output_dir / 'ocr_words.csv')

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
    sel_df['sel_bbox_line_converted'] = \
        sel_df.sel_bbox.apply(lambda sel_bbox: sel_ocr_word_match.convert_bbox(x_line, y_line, sel_bbox))

    pd.set_option('display.max_columns', 200)
    pd.set_option('display.width', 200)
    print(sel_df[['text', 'ocr_text', 'ocr_bbox', 'sel_bbox_line_converted', 'sel_bbox']])

    print()
    print(f'x_line: {x_line}')
    print(f'y_line: {y_line}')
    print(f'dpi:    {dpi}')
    print(f'margin: {margin}')

    ############################################################################
    # since the words are already matched by text, figure out the statistical  #
    # version of the transformation and apply that.                            #
    # seems to be about 1/5 of an inch off MAE                                 #
    # need to visualize to diagnose                                            #
    #   html                                                                   #
    #   background image for page                                              #
    #   boxes for each word.  how to place? y count up? down?                  #
    ############################################################################
    from sklearn.linear_model import LassoCV
    model = LassoCV(fit_intercept=True)

    sbox_array = np.array(list(sel_df['sel_bbox'].apply(lambda bbox: bbox.to_array())))
    obox_array = np.array(list(sel_df['ocr_bbox'].apply(lambda bbox: bbox.to_array())))

    to_write = ['[self.xmin, self.xmax, self.ymin, self.ymax]']
    for target_ind in range(4):
        x = sbox_array[:, target_ind:target_ind+1]
        # x = sbox_array
        y = obox_array[:, target_ind]
        model.fit(x, y)
        y_hat = model.predict(x)
        mae = np.mean(np.abs(y_hat - y))
        print(target_ind, model.intercept_, model.coef_, mae)

        conversion_model_str = ' '.join([str(e) for e in [target_ind, model.intercept_, model.coef_, mae]])
        to_write.append(conversion_model_str)

    to_write.append(f'x_line: {x_line}')
    to_write.append(f'y_line: {y_line}')

    to_write.append(f'dpi:    {dpi}')
    to_write.append(f'margin: {margin}')

    utils.save_txt(output_dir / 'conversion_factors.txt', '\n'.join(to_write))
