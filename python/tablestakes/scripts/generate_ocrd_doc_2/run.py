from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image as pil_image

from tablestakes import utils, html_css as hc, etree_modifiers, ocr
from tablestakes.scripts.generate_ocrd_doc_2 import basic


if __name__ == '__main__':
    ##############
    # Parameters #
    ##############
    seed = 42
    dpi = 500
    margin = '1in'
    page_size = hc.PageSize.LETTER

    window_width_px = dpi * page_size.width
    window_height_px = dpi * page_size.height

    doc = basic.make_doc(seed)

    #########
    # setup #
    #########
    doc_ind = 1
    output_dir = Path('.') / 'docs' / f'doc_{doc_ind:02d}'
    utils.mkdir_if_not_exist(output_dir)

    params_file = output_dir / 'params.txt'
    params_dict = {
        'seed': seed,
        'dpi': dpi,
        'margin': margin,
        'page_size': page_size.name,
    }
    utils.save_txt(params_file, str(params_dict))

    ########################################
    # postproc doc to add word_ids, labels #
    ########################################
    df_saver = etree_modifiers.SaveWordAttribsToDataFrame()
    post_proc_stack = etree_modifiers.EtreeModifierStack(
        modifiers=[
            etree_modifiers.WordWrapper(),
            etree_modifiers.SetIsKeyOnWordsModifier(),
            etree_modifiers.SetIsValueOnWordsModifier(),
            etree_modifiers.ConvertParentClassNamesToWordAttribsModifier(),
            etree_modifiers.CopyWordTextToAttribModifier(),
            etree_modifiers.WordColorizer(),
            df_saver,
        ],
    )
    doc = post_proc_stack(doc)

    ###################################
    # save html, pdf, words.csv files #
    ###################################
    doc_html_file = output_dir / 'doc.html'
    doc_pdf_file = output_dir / 'doc.pdf'
    doc.save_html(doc_html_file)
    page_image_files = doc.save_pdf(
        doc_pdf_file,
        do_save_page_images_too=True,
        dpi=dpi,
    )

    words_df_file = output_dir / 'words.csv'
    words_df = df_saver.get_df()
    words_df.to_csv(words_df_file)

    ############################################
    # paint all words with solid colored boxes #
    ############################################
    doc = etree_modifiers.WordColorDocCssAdder(doc)(doc)

    ###################################
    # save colored html and pdf files #
    ###################################
    colored_doc_html_file = output_dir / 'doc_colored.html'
    colored_doc_pdf_file = output_dir / 'doc_colored.pdf'
    doc.save_html(colored_doc_html_file)
    colored_page_image_files = doc.save_pdf(
        colored_doc_pdf_file,
        do_save_page_images_too=True,
        dpi=dpi, pages_dirname='pages_colored',
    )

    ########################################
    # ocr the non-colored page image files #
    ########################################
    ocr_df = ocr.TesseractOcrProvider().ocr(
        page_image_files=page_image_files,
        save_raw_ocr_output_location=output_dir / 'ocr_raw.csv',
    )
    ocr_df_file = output_dir / 'ocr.csv'
    ocr_df.to_csv(ocr_df_file)

    #####################################################
    # find the median color block under each OCR'd word #
    #####################################################
    def get_colors_under_word(colored_page_image_arrays: List[np.ndarray], row: pd.Series):
        """row from the ocr df (ocr.csv) representing a word and including fields for page_num, LRTB"""
        page_num = row[ocr.TesseractOcrProvider.PAGE_NUM_COL_NAME]
        if page_num > len(colored_page_image_arrays):
            raise ValueError(f"page_num: {page_num}, len(page_image_arrays): {len(colored_page_image_arrays)}")

        page = colored_page_image_arrays[page_num]

        word_slice = page[row[ocr.TesseractOcrDfFactory.TOP]:row[ocr.TesseractOcrDfFactory.BOTTOM], :]
        word_slice = word_slice[:, row[ocr.TesseractOcrDfFactory.LEFT]:row[ocr.TesseractOcrDfFactory.RIGHT]]

        return {
            'median': np.median(word_slice, axis=(0, 1)),
            'mean': np.mean(word_slice, axis=(0, 1)),
        }

    colored_page_image_arrays = utils.load_image_files_to_arrays(colored_page_image_files)

    color_stats_of_ocr_boxes = []
    for ind, row in ocr_df.iterrows():
        color_stats_of_ocr_boxes.append(get_colors_under_word(colored_page_image_arrays, row))

    #############################
    # get canonical word colors #
    #############################
    word_colors_df = words_df[etree_modifiers.WordColorizer.RGB]
    word_colors_df = word_colors_df.apply(pd.to_numeric)

    #########################################
    # write color-matched words into ocr_df #
    #########################################
    word_ids = []
    dists = []
    for ocr_word_ind, color_stat_of_ocr_box in enumerate(color_stats_of_ocr_boxes):
        distances_from_current_ocr_word_to_each_words_color = \
            np.linalg.norm(color_stat_of_ocr_box['mean'] - word_colors_df, ord=1, axis=1)
        index_of_closest_color = np.argmin(distances_from_current_ocr_word_to_each_words_color)
        mae_to_closest_color = distances_from_current_ocr_word_to_each_words_color[index_of_closest_color]
        # TODO: Factor out id str
        min_word_id = words_df.iloc[index_of_closest_color]['id']
        print(f'min_index: {index_of_closest_color}, min_mae: {mae_to_closest_color}, min_word_id: {min_word_id}')
        word_ids.append(min_word_id)
        dists.append(mae_to_closest_color)

    ocr_df['closest_color_word_id'] = word_ids
    ocr_df['closest_color_dist'] = dists

    ocr_df.to_csv(ocr_df_file)


