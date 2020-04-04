from pathlib import Path

import numpy as np

from tablestakes import utils, html_css as hc, etree_modifiers
from tablestakes.scripts.generate_ocrd_doc_2.basic import doc


if __name__ == '__main__':
    ##############
    # Parameters #
    ##############
    np.random.seed(42)
    dpi = 500
    margin = '1in'
    page_size = hc.PageSize.LETTER

    window_width_px = dpi * page_size.width
    window_height_px = dpi * page_size.height

    doc_ind = 1

    output_dir = Path('.') / 'docs' / f'doc_{doc_ind:02d}'
    utils.mkdir_if_not_exist(output_dir)

    ########################################
    # postproc doc to add word_ids, labels #
    ########################################
    wrapper = etree_modifiers.WordWrapper()
    df_saver = etree_modifiers.SaveWordAttribsToDataFrame()
    post_proc_stack = etree_modifiers.EtreeModifierStack(
        modifiers=[
            wrapper,
            etree_modifiers.SetIsKeyOnWordsModifier(),
            etree_modifiers.SetIsValueOnWordsModifier(),
            etree_modifiers.ConvertParentClassNamesToWordAttribsModifier(),
            etree_modifiers.CopyWordTextToAttribModifier(),
            etree_modifiers.SeleniumWordLocator(),
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
    doc.save_pdf(doc_pdf_file)

    words_df = output_dir / 'words.csv'
    df_saver.get_df().to_csv(words_df)
