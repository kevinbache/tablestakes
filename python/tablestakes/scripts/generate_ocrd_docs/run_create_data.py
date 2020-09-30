import multiprocessing
import os

from tablestakes.constants import Y_KORV_NAME, Y_WHICH_KV_NAME, X_BASIC_NAME, X_VOCAB_NAME

num_jobs = multiprocessing.cpu_count() - 1
print(f'num_jobs: {num_jobs}')

# https://tesseract-ocr.github.io/tessdoc/FAQ#can-i-increase-speed-of-ocr
# doesn't do much since we're already doing process parallelizations
# 7.5 sec / page -> 6.5 sec / page
os.environ["OMP_THREAD_LIMIT"] = f'{num_jobs}'

from joblib import Parallel, delayed

from tablestakes import utils, html_css as hc, etree_modifiers, ocr, color_matcher, df_modifiers, constants
from tablestakes.scripts.generate_ocrd_docs import basic


def make_and_ocr_docs(doc_ind, settings):
    print(f"STARTING TO CREATE DOC {doc_ind}")
    seed = settings.seed_start + doc_ind
    doc = basic.make_doc(seed, settings.num_extra_fields, do_randomize_field_order=settings.do_randomize_field_order)
    this_doc_dir = settings.docs_dir / f'doc_{doc_ind:02d}'
    utils.mkdir_if_not_exist(this_doc_dir)

    params_file = this_doc_dir / 'params.txt'
    params_dict = {
        'seed': seed,
        'dpi': settings.dpi,
        'margin': settings.margin,
        'page_size': settings.page_size.name,
        'num_extra_fields': settings.num_extra_fields,
        'min_count_to_keep_word': settings.min_count_to_keep_word,
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
    raw_dir = this_doc_dir / '0.raw'
    utils.mkdir_if_not_exist(raw_dir)
    doc_html_file = raw_dir / 'doc.html'
    doc_pdf_file = raw_dir / 'doc.pdf'
    doc.save_html(doc_html_file)
    page_image_files = doc.save_pdf(
        doc_pdf_file,
        do_save_page_images_too=True,
        dpi=settings.dpi,
    )
    words_df_file = raw_dir / 'words.csv'
    words_df = df_saver.get_df()
    words_df.to_csv(words_df_file)

    ############################################
    # paint all words with solid colored boxes #
    ############################################
    doc = etree_modifiers.WordColorDocCssAdder(doc)(doc)

    ###################################
    # save colored html and pdf files #
    ###################################
    colored_dir = this_doc_dir / '1.colored'
    utils.mkdir_if_not_exist(colored_dir)
    colored_doc_html_file = colored_dir / 'doc_colored.html'
    colored_doc_pdf_file = colored_dir / 'doc_colored.pdf'
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
    ocr_dir = this_doc_dir / '2.ocr'
    utils.mkdir_if_not_exist(ocr_dir)
    ocr_df = ocr.TesseractOcrProvider().ocr(
        page_image_files=page_image_files,
        save_raw_ocr_output_location=ocr_dir / 'ocr_raw.csv',
    )
    ocr_df_file = ocr_dir / 'ocr.csv'
    ocr_df.to_csv(ocr_df_file)

    return doc_ind, ocr_df, ocr_df_file, words_df, colored_page_image_files, this_doc_dir


def create_and_save_xy_csvs(
        ocr_df,
        ocr_df_file,
        words_df,
        colored_page_image_files,
        this_doc_dir,
        vocabulizer: df_modifiers.Vocabulizer,
        rare_word_eliminator: df_modifiers.RareWordEliminator,
):
    ##########################################
    # match ocr words to true words by color #
    ##########################################
    joined_dir = this_doc_dir / '3.joined'
    utils.mkdir_if_not_exist(joined_dir)
    joined_df = color_matcher.WordColorMatcher.get_joined_df(ocr_df, words_df, colored_page_image_files)
    joined_df.to_csv(joined_dir / 'joined.csv')
    # ocr_df gets the join word_id columns added in WordColorMatcher.get_joined_df
    ocr_df.to_csv(ocr_df_file)
    # Tokenizer changes the number of rows of the DF if there are any rows with multi-word text

    joined_df = df_modifiers.DfModifierStack([
        df_modifiers.Tokenizer(),
        df_modifiers.CharCounter(),
        df_modifiers.DetailedOtherCharCounter(),
        df_modifiers.TokenPostProcessor(),
        vocabulizer,
    ])(joined_df)

    joined_df = df_modifiers.DfModifierStack([
        rare_word_eliminator,
    ])(joined_df)

    x_base_cols = [
        ocr.TesseractOcrProvider.PAGE_NUM_COL_NAME,
        ocr.OcrDfNames.LEFT,
        ocr.OcrDfNames.RIGHT,
        ocr.OcrDfNames.TOP,
        ocr.OcrDfNames.BOTTOM,
        ocr.OcrDfNames.CONFIDENCE,
        color_matcher.WordColorMatcher.PAGE_HEIGHT_COL_NAME,
        color_matcher.WordColorMatcher.PAGE_WIDTH_COL_NAME,
        color_matcher.WordColorMatcher.NUM_PAGES_COL_NAME,
        df_modifiers.RareWordEliminator.WORD_WAS_ELIMINATED_COL_NAME,
    ]
    x_base_cols += [c for c in joined_df.columns if c.startswith(df_modifiers.CharCounter.PREFIX)]

    x_vocab_cols = [df_modifiers.RareWordEliminator.VOCAB_ID_COL_NAME]

    y_korv_cols = [
        etree_modifiers.SetIsKeyOnWordsModifier.KEY_NAME,
        etree_modifiers.SetIsValueOnWordsModifier.KEY_NAME
    ]
    which_kv_prefix = etree_modifiers.ConvertParentClassNamesToWordAttribsModifier.TAG_PREFIX
    y_which_kv_cols = [c for c in joined_df.columns if c.startswith(which_kv_prefix)]

    final_data_dir = this_doc_dir / '4.data'
    utils.mkdir_if_not_exist(final_data_dir)

    data_dfs = utils.split_df_by_cols(
        joined_df,
        [x_base_cols, x_vocab_cols, y_korv_cols, y_which_kv_cols],
        do_output_leftovers_df=True,
        names=[X_BASIC_NAME, X_VOCAB_NAME, Y_KORV_NAME, Y_WHICH_KV_NAME, 'meta'],
    )
    Y_PREFIX = 'y_'
    for name, df in data_dfs.items():
        if name.startswith(Y_PREFIX):
            utils.one_hot_to_categorical(df, name).to_csv(final_data_dir / f'{name}.csv', index=False)
            short_name = name[len(Y_PREFIX):]
            df.to_csv(final_data_dir / f'{short_name}_onehot.csv', index=False)
        else:
            df.to_csv(final_data_dir / f'{name}.csv', index=False)

    return len(y_korv_cols), len(y_which_kv_cols), vocabulizer, rare_word_eliminator


if __name__ == '__main__':
    # https://tesseract-ocr.github.io/tessdoc/FAQ#can-i-increase-speed-of-ocr

    ##############
    # Parameters #
    ##############
    class DocSettings:
        seed_start = 42
        dpi = 500
        margin = '1in'
        page_size = hc.PageSize.LETTER

        min_count_to_keep_word = 2

        window_width_px = dpi * page_size.width
        window_height_px = dpi * page_size.height

        num_docs = 100
        num_extra_fields = 0
        do_randomize_field_order = True

        docs_dir = constants.DOCS_DIR

    do_regen_docs = True

    doc_settings = DocSettings()

    fast_test = False
    if fast_test:
        doc_settings.dpi = 100
        doc_settings.num_extra_fields = 2
        doc_settings.num_docs = 10

    doc_settings.docs_dir /= f'num={doc_settings.num_docs}_extra={doc_settings.num_extra_fields}'
    print(f'Saving to {str(doc_settings.docs_dir)}')

    ocr_func = lambda doc_ind: make_and_ocr_docs(doc_ind, doc_settings)
    ocr_outputs = Parallel(n_jobs=num_jobs)(delayed(ocr_func)(doc_ind) for doc_ind in range(doc_settings.num_docs))

    # run the rest serially so vocabulizer is consistent across docs
    num_korv_classes, num_which_kv_classes = None, None
    vocabulizer = df_modifiers.Vocabulizer()
    rare_word_eliminator = \
        df_modifiers.RareWordEliminator(vocabulizer=vocabulizer, min_count=doc_settings.min_count_to_keep_word)
    for ocr_output in ocr_outputs:
        doc_ind, ocr_df, ocr_df_file, words_df, colored_page_image_files, this_doc_dir = ocr_output
        print(f'Starting to postproc doc {doc_ind}')
        num_korv_classes, num_which_kv_classes, vocabulizer, rare_word_eliminator = \
            create_and_save_xy_csvs(
                ocr_df,
                ocr_df_file,
                words_df,
                colored_page_image_files,
                this_doc_dir,
                vocabulizer,
                rare_word_eliminator,
            )

    utils.save_json(doc_settings.docs_dir / 'word_to_id_pre_elimination.json', vocabulizer.word_to_id)
    utils.save_json(doc_settings.docs_dir / 'word_to_count_pre_elimination.json', vocabulizer.word_to_count)
    utils.save_json(doc_settings.docs_dir / 'word_to_count.json', rare_word_eliminator.word_to_count)
    utils.save_json(doc_settings.docs_dir / 'word_to_id.json', rare_word_eliminator.word_to_id)

    utils.save_json(
        doc_settings.docs_dir / 'num_y_classes.json',
        {
            'korv': num_korv_classes,
            'which_kv': num_which_kv_classes,
        },
    )

    print()
    print(f'Saved to {str(doc_settings.docs_dir)}')