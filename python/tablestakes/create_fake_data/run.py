import multiprocessing
import os

# https://tesseract-ocr.github.io/tessdoc/FAQ#can-i-increase-speed-of-ocr
# doesn't do much since we're already doing process parallelizations
# 7.5 sec / page -> 6.5 sec / page
num_jobs = multiprocessing.cpu_count() - 2
os.environ["OMP_THREAD_LIMIT"] = f'{num_jobs}'

import ray

from tablestakes import utils, ocr, df_modifiers, constants
from tablestakes.create_fake_data import basic_document, color_matcher, etree_modifiers
from tablestakes.ml import hyperparams
from tablestakes.ml import data


@ray.remote
def make_and_ocr_docs(doc_ind, doc_set_params: hyperparams.DocSetParams):
    print(f"Starting to create doc {doc_ind}")
    doc_gen_params = doc_set_params.doc_gen_params.sample()
    assert isinstance(doc_gen_params, hyperparams.DocGenParams)  # for pycharm autocomplete

    doc = basic_document.make_doc(
        seed=doc_set_params.seed_start + doc_ind,
        doc_config=doc_gen_params,
    )
    this_doc_dir = doc_set_params.docs_dir / f'doc_{doc_ind:02d}'
    utils.mkdir_if_not_exist(this_doc_dir)

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
        do_use_timers=False,
    )
    doc = post_proc_stack(doc)

    #####################################
    # 0.save html, pdf, words.csv files #
    #####################################
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
    doc = etree_modifiers.WordColorDocCssAdder(doc)(doc, do_use_timers=False)

    #####################################
    # 1.save colored html and pdf files #
    #####################################
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

    ##########################################
    # 2.ocr the non-colored page image files #
    ##########################################
    ocr_dir = this_doc_dir / '2.ocr'
    utils.mkdir_if_not_exist(ocr_dir)
    ocr_df = ocr.TesseractOcrProvider().ocr(
        page_image_files=page_image_files,
        save_raw_ocr_output_location=ocr_dir / 'ocr_raw.csv',
    )
    ocr_df_file = ocr_dir / 'ocr.csv'
    ocr_df.to_csv(ocr_df_file)

    return doc_ind, ocr_df, ocr_df_file, words_df, colored_page_image_files, this_doc_dir


@ray.remote
def join_and_split_and_save_dfs(
        ocr_df,
        ocr_df_file,
        words_df,
        colored_page_image_files,
        this_doc_dir,
):
    print(f'Starting join_and_create_vocab on {this_doc_dir}')

    ############################################
    # 3.match ocr words to true words by color #
    ############################################
    joined_dir = this_doc_dir / '3.joined'
    utils.mkdir_if_not_exist(joined_dir)
    joined_df = color_matcher.WordColorMatcher.get_joined_df(ocr_df, words_df, colored_page_image_files)
    joined_df.to_csv(joined_dir / 'joined.csv')
    # ocr_df gets the join word_id columns added in WordColorMatcher.get_joined_df
    ocr_df.to_csv(ocr_df_file)

    # MyBertTokenizer changes the number of rows of the DF if there are any rows with multi-word text
    joined_df = df_modifiers.DfModifierStack(
        modifiers=[
            df_modifiers.MyBertTokenizer(),
            df_modifiers.CharCounter(),
        ],
        do_use_timers=False,
    )(joined_df)

    x_base_cols = [
        constants.ColNames.PAGE_NUM,
        constants.ColNames.LEFT,
        constants.ColNames.RIGHT,
        constants.ColNames.TOP,
        constants.ColNames.BOTTOM,
        constants.ColNames.CONFIDENCE,
        constants.ColNames.PAGE_WIDTH,
        constants.ColNames.PAGE_HEIGHT,
        constants.ColNames.NUM_PAGES,
    ]
    x_base_cols += constants.ColNames.CHAR_COUNT_COLS

    x_vocab_cols = [constants.ColNames.TOKEN_ID]

    y_korv_cols = [
        etree_modifiers.SetIsKeyOnWordsModifier.KEY_NAME,
        etree_modifiers.SetIsValueOnWordsModifier.KEY_NAME
    ]
    which_kv_prefix = etree_modifiers.ConvertParentClassNamesToWordAttribsModifier.TAG_PREFIX
    y_which_kv_cols = [c for c in joined_df.columns if c.startswith(which_kv_prefix)]

    ###################
    # 4.Save data dfs #
    ###################
    final_data_dir = this_doc_dir / '4.data'
    utils.mkdir_if_not_exist(final_data_dir)

    data_dfs = utils.split_df_by_cols(
        joined_df,
        [x_base_cols, x_vocab_cols, y_korv_cols, y_which_kv_cols],
        do_output_leftovers_df=True,
        df_names=[
            constants.X_BASE_NAME,
            constants.X_VOCAB_NAME,
            constants.Y_KORV_NAME,
            constants.Y_WHICH_KV_NAME,
            'meta',
        ],
    )
    Y_PREFIX = 'y_'
    for name, df in data_dfs.items():
        if name.startswith(Y_PREFIX):
            utils.one_hot_to_categorical(df, name).to_csv(final_data_dir / f'{name}.csv', index=False)
            # short_name = name[len(Y_PREFIX):]
            # df.to_csv(final_data_dir / f'{short_name}_onehot.csv', index=False)
        else:
            df.to_csv(final_data_dir / f'{name}.csv', index=False)

    return len(y_korv_cols), len(y_which_kv_cols)


if __name__ == '__main__':
    # https://tesseract-ocr.github.io/tessdoc/FAQ#can-i-increase-speed-of-ocr
    ray.init(
        ignore_reinit_error=True,
        local_mode=False,
    )

    ##############
    # Parameters #
    ##############
    do_regen_docs = True

    doc_gen_params = hyperparams.DocGenParams()

    doc_prep_params = hyperparams.DocPrepParams()
    doc_prep_params.min_count_to_keep_word = 4

    doc_settings = hyperparams.DocSetParams(
        doc_gen_params=doc_gen_params,
        doc_prep_params=doc_prep_params,
    )
    doc_settings.num_docs = 100
    doc_settings.doc_gen_params.num_extra_fields = 0

    fast_test = False
    if fast_test:
        doc_settings.num_docs = 20
        doc_settings.doc_gen_params.dpi = 100
        doc_settings.doc_gen_params.num_extra_fields = 1

    doc_settings.set_docs_dir()

    # PHASE 1
    print(f'Saving to {str(doc_settings.docs_dir)}')
    ocr_outputs = []
    for doc_ind in range(doc_settings.num_docs):
        ocr_outputs.append(make_and_ocr_docs.remote(doc_ind, doc_settings))

    utils.ray_prog_bar(ocr_outputs)

    # PHASE 2
    joined_dfs = []
    doc_dirs = []
    for ocr_output in ocr_outputs:
        doc_ind, ocr_df, ocr_df_file, words_df, colored_page_image_files, this_doc_dir = ray.get(ocr_output)
        joined_df = join_and_split_and_save_dfs.remote(
            ocr_df,
            ocr_df_file,
            words_df,
            colored_page_image_files,
            this_doc_dir,
        )
        joined_dfs.append(joined_df)
        doc_dirs.append(this_doc_dir)

    utils.ray_prog_bar(joined_dfs)

    with utils.Timer('Reading csvs into Dataset'):
        dataset = data.TablestakesDataset(docs_dir=doc_settings.docs_dir)

    dataset_file = doc_settings.get_dataset_file()
    with utils.Timer(f'Saving Dataset of type {type((dataset))} to {dataset_file}'):
        dataset.save(dataset_file)

    print()
    print(f'Saved to {str(doc_settings.docs_dir)} and {dataset_file}')
    print(f'Settings:')
    utils.print_dict((doc_settings.to_dict()))
