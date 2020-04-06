from pathlib import Path

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
    page_image_files = doc.save_pdf(doc_pdf_file, do_save_page_images_too=True, dpi=dpi)

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
    doc_colored_html_file = output_dir / 'doc_colored.html'
    doc_colored_pdf_file = output_dir / 'doc_colored.pdf'
    doc.save_html(doc_colored_html_file)
    page_image_colored_files = doc.save_pdf(doc_colored_pdf_file, do_save_page_images_too=True, dpi=dpi)


    ############################
    # ocr the page image files #
    ############################
    ocr_df = ocr.TesseractOcrProvider().ocr(
        page_image_files=page_image_files,
        save_raw_ocr_output_location=output_dir / 'ocr_raw.csv',
    )
    ocr_df_file = output_dir / 'ocr.csv'
    ocr_df.to_csv(ocr_df_file)

    # calibration_df = ocr.calibrate_conversion_parameters(words_df, ocr_df)
    # calibration_df.to_csv(output_dir / 'calibration.csv')
    # print(calibration_df)

    # now each ocr word looks up the modal color in it's region.
    # done.  perfect.

    ocr_df_file.apply(lambda row: row['page_num'])