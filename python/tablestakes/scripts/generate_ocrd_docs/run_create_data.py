import multiprocessing
import os

from joblib import Parallel, delayed
from tablestakes.constants import Y_KORV_NAME, Y_WHICH_KV_NAME, X_BASIC_NAME, X_VOCAB_NAME
from tablestakes.ml import hyperparams

import ray
ray.init(ignore_reinit_error=True)

# https://tesseract-ocr.github.io/tessdoc/FAQ#can-i-increase-speed-of-ocr
# doesn't do much since we're already doing process parallelizations
# 7.5 sec / page -> 6.5 sec / page
num_jobs = multiprocessing.cpu_count()

os.environ["OMP_THREAD_LIMIT"] = f'{num_jobs}'

from tablestakes import utils, etree_modifiers, ocr, color_matcher, df_modifiers
from tablestakes.scripts.generate_ocrd_docs import basic


@ray.remote
def make_and_ocr_docs(doc_ind, doc_set_params: hyperparams.DocSetParams):
    print(f"STARTING TO CREATE DOC {doc_ind}")
    doc_gen_params = doc_set_params.doc_gen_params.sample()
    assert isinstance(doc_gen_params, hyperparams.DocGenParams)  # for pycharm autocomplete

    doc = basic.make_doc(
        seed=doc_set_params.seed_start + doc_ind,
        doc_config=doc_gen_params,
    )
    this_doc_dir = doc_set_params.docs_dir / f'doc_{doc_ind:02d}'
    utils.mkdir_if_not_exist(this_doc_dir)

    import pprint
    from chillpill import params
    print(params.__file__)
    # pprint.pprint(doc_gen_params.to_dict())
    # print()
    # for k, v in doc_gen_params.__dict__.items():
    #     print(type(k), type(v), k, v)
    utils.save_json(this_doc_dir / 'params.txt', doc_gen_params.to_dict())

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
        dpi=doc_gen_params.dpi,
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
        dpi=doc_gen_params.dpi,
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
    do_regen_docs = True

    doc_settings = hyperparams.DocSetParams(
        doc_gen_params=hyperparams.DocGenParams(),
        doc_prep_params=hyperparams.DocPrepParams(),
    )

    fast_test = True
    if fast_test:
        doc_settings.dpi = 100
        doc_settings.num_extra_fields = 1
        doc_settings.num_docs = 10

    doc_settings.set_docs_dir()

    print(f'Saving to {str(doc_settings.docs_dir)}')
    # ocr_func = lambda doc_ind: make_and_ocr_docs(doc_ind, doc_settings)
    # ocr_outputs = Parallel(n_jobs=num_jobs)(delayed(ocr_func)(doc_ind) for doc_ind in range(doc_settings.num_docs))
    ocr_outputs = []
    for doc_ind in range(doc_settings.num_docs):
        ocr_outputs.append(make_and_ocr_docs.remote(doc_ind, doc_settings))

    # run the rest serially so vocabulizer is consistent across docs
    num_korv_classes, num_which_kv_classes = None, None
    vocabulizer = df_modifiers.Vocabulizer()
    rare_word_eliminator = \
        df_modifiers.RareWordEliminator(
            vocabulizer=vocabulizer,
            min_count=doc_settings.doc_prep_params.min_count_to_keep_word,
        )

    for ocr_output in ocr_outputs:
        doc_ind, ocr_df, ocr_df_file, words_df, colored_page_image_files, this_doc_dir = ray.get(ocr_output)
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

