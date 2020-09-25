import multiprocessing
from functools import partial
from joblib import Parallel, delayed
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

from tablestakes import utils, html_css as hc, etree_modifiers, ocr, color_matcher, df_modifiers
from tablestakes.scripts.generate_ocrd_doc_2 import basic


def make_and_ocr_docs(doc_ind, settings):
    print(f"STARTING TO CREATE DOC {doc_ind}")
    seed = settings.seed_start + doc_ind
    doc = basic.make_doc(seed, settings.num_extra_fields)
    output_dir = settings.docs_dir / f'doc_{doc_ind:02d}'
    utils.mkdir_if_not_exist(output_dir)

    params_file = output_dir / 'params.txt'
    params_dict = {
        'seed': seed,
        'dpi': settings.dpi,
        'margin': settings.margin,
        'page_size': settings.page_size.name,
    }
    utils.save_json(params_file, params_dict)

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
        dpi=settings.dpi,
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
        dpi=settings.dpi,
        pages_dirname='pages_colored',
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

    return doc_ind, ocr_df, ocr_df_file, words_df, colored_page_image_files, output_dir


def _split_df_by_cols(df: pd.DataFrame, col_sets: List[List[str]], do_output_leftovers_df=True):
    if do_output_leftovers_df:
        # flat list
        used_cols = [col_name for col_set in col_sets for col_name in col_set]
        leftover_cols = [col_name for col_name in df.columns if col_name not in used_cols]
        col_sets.append(leftover_cols)

    return [df[col_set].copy() for col_set in col_sets]


def create_and_save_xy_csvs(ocr_df, ocr_df_file, words_df, colored_page_image_files, output_dir):
    ##########################################
    # match ocr words to true words by color #
    ##########################################
    joined_df = color_matcher.WordColorMatcher.get_joined_df(ocr_df, words_df, colored_page_image_files)
    joined_df.to_csv(output_dir / 'joined.csv')
    # ocr_df gets the join word_id columns added in WordColorMatcher.get_joined_df
    ocr_df.to_csv(ocr_df_file)
    # Tokenizer change the number of rows of the DF if there are any rows with multi-word text
    joined_df = df_modifiers.DfModifierStack([
        df_modifiers.Tokenizer(),
        vocabulizer,
        df_modifiers.CharCounter(),
        df_modifiers.DetailedOtherCharCounter(),
    ])(joined_df)
    x_cols = [
        ocr.OcrDfFactory.LEFT,
        ocr.OcrDfFactory.RIGHT,
        ocr.OcrDfFactory.TOP,
        ocr.OcrDfFactory.BOTTOM,
        ocr.OcrDfFactory.CONFIDENCE,
    ]
    x_cols += [c for c in joined_df.columns if c.startswith(df_modifiers.CharCounter.PREFIX)]
    x_vocab_cols = [df_modifiers.Vocabulizer.VOCAB_NAME]
    # y_do_include_key_value_cols = True
    # y_do_include_field_id_cols = True
    y_korv_cols = [
        etree_modifiers.SetIsKeyOnWordsModifier.KEY_NAME,
        etree_modifiers.SetIsValueOnWordsModifier.KEY_NAME
    ]
    is_kv_prefix = etree_modifiers.ConvertParentClassNamesToWordAttribsModifier.TAG_PREFIX
    y_which_kv_cols = [c for c in joined_df.columns if c.startswith(is_kv_prefix)]

    x_df, x_vocab_df, y_korv_df, y_which_kv_df, meta_df = _split_df_by_cols(
        joined_df,
        [x_cols, x_vocab_cols, y_korv_cols, y_which_kv_cols],
        do_output_leftovers_df=True,
    )
    x_df.to_csv(output_dir / 'x.csv', index=False)
    x_vocab_df.to_csv(output_dir / 'x_vocab.csv', index=False)
    y_korv_df.to_csv(output_dir / 'y_korv.csv', index=False)
    y_which_kv_df.to_csv(output_dir / 'y_which_kv.csv', index=False)
    meta_df.to_csv(output_dir / 'meta.csv', index=False)

    # torch.cross_entropy expects ys to be 1 dim categorical
    y_korv_vect = np.argmax(y_korv_df.values, axis=-1).astype(np.int)
    y_which_kv_vect = np.argmax(y_which_kv_df.values, axis=-1).astype(np.int)
    np.savetxt(output_dir / 'y_korv_vect.csv', y_korv_vect, fmt='%d', delimiter=',')
    np.savetxt(output_dir / 'y_which_kv_vect.csv', y_which_kv_vect, fmt='%d', delimiter=',')


if __name__ == '__main__':
    ##############
    # Parameters #
    ##############
    class DocSettings:
        seed_start = 42
        dpi = 500
        margin = '1in'
        page_size = hc.PageSize.LETTER

        window_width_px = dpi * page_size.width
        window_height_px = dpi * page_size.height

        num_docs = 30
        num_extra_fields = 20

        docs_dir = Path('.') / 'docs'

    doc_settings = DocSettings()

    fast_test = False
    if fast_test:
        doc_settings.dpi = 100
        doc_settings.num_extra_fields = 2
        doc_settings.num_docs = 4

    # run ocr in parallel because it's slow
    num_jobs = multiprocessing.cpu_count() - 1
    print(f'num_jobs: {num_jobs}')
    ocr_func = lambda doc_ind: make_and_ocr_docs(doc_ind, doc_settings)
    ocr_outputs = Parallel(n_jobs=num_jobs)(delayed(ocr_func)(doc_ind) for doc_ind in range(doc_settings.num_docs))

    # run the rest serially so vocabulizer is consistent across docs
    vocabulizer = df_modifiers.Vocabulizer()
    for ocr_output in ocr_outputs:
        # print('*****************************')
        # print('*****************************')
        # print('*****************************')
        # print(ocr_output)
        # print('-------')
        # for ind, e in enumerate(ocr_output):
        #     print(type(e))
        doc_ind, ocr_df, ocr_df_file, words_df, colored_page_image_files, output_dir = ocr_output
        print(doc_ind)
        create_and_save_xy_csvs(ocr_df, ocr_df_file, words_df, colored_page_image_files, output_dir)
    utils.save_json(doc_settings.docs_dir / 'word_to_id.txt', vocabulizer.word_to_id)
    utils.save_json(doc_settings.docs_dir / 'word_to_count.txt', vocabulizer.word_to_count)



