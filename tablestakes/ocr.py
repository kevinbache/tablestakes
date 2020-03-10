import abc
from typing import List

import pandas as pd
import pdf2image
import pytesseract

from tablestakes import doc


class OcrProvider(abc.ABC):
    def __init__(self, dpi=400):
        self.dpi = dpi

    @classmethod
    def load_pdf_to_images(cls, pdf_filename: str, dpi: int):
        return pdf2image.convert_from_path(pdf_filename, dpi=dpi)

    @abc.abstractmethod
    def ocr(self, pdf_filename: str) -> doc.Document:
        pass


class TesseractOcrProvider(OcrProvider):
    def ocr(self, pdf_filename: str) -> doc.Document:
        pyt_data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DATAFRAME,
        )
        # import pandas as pd
        # df = pd.DataFrame()
        # df.to_pickle()
        pass


class TesseractDocumentFactory:
    pass


if __name__ == '__main__':
    fullfile = '../output/sample_weasy_output.pdf'
    resolution = 400

    images = pdf2image.convert_from_path(fullfile, dpi=resolution)
    image = images[0]

    pyt_df = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
    # for (page_num, block_num, par_num), group_df in pyt_df.groupby(['page_num', 'block_num', 'par_num']):
    #     print(page_num, block_num, par_num)
    #     print(group_df)
    #     print()

    def _row_to_bbox(row: pd.Series):
        return doc.BBox(
            xmin=row['left'],
            xmax=row['left'] + row['width'],
            ymin=row['top'],
            ymax=row['top'] + row['height'],
        )

    def _get_words_from_para_df(para_df: pd.DataFrame) -> List[doc.Word]:
        words = []
        for id, row in para_df.iterrows():
            level = row['level']
            if level != 5:
                # only process words
                continue
            words.append(doc.Word(
                text=row['text'],
                bbox=_row_to_bbox(row),
                confidence=row['confidence'],
            ))
        return words

    BLOCK_TYPE_TEXT_STR = 'TEXT'

    document = doc.Document(pages=[])
    bbox = None
    for row_id, row, in pyt_df.iterrows():
        page_index = row['page_num'] - 1
        block_index = row['block_num'] - 1
        par_index = row['par_num'] - 1

        print(row_id, row['level'], page_index, block_index, par_index)
        prev_bbox = bbox
        bbox = _row_to_bbox(row)
        if row['level'] == 1:
            print('  appending new page')
            document.pages.append(
                doc.Page(blocks=[], bbox=bbox)
            )
        elif row['level'] == 2:
            print('  appending new block')
            document.pages[page_index].blocks.append(
                doc.Block(paragraphs=[], bbox=bbox, block_type=BLOCK_TYPE_TEXT_STR)
            )
        elif row['level'] == 3:
            print('  appending new paragraph')
            document.pages[page_index].blocks[block_index].paragraphs.append(
                doc.Paragraph(words=[], bbox=bbox)
            )
        elif row['level'] == 4:
            # ignore tesseract lines
            prev_bbox.xmax = prev_bbox.xmin
            document.pages[page_index].blocks[block_index].paragraphs[par_index].words.append(
                doc.Word(text='\n', bbox=prev_bbox, word_type=doc.Word.WordType.LINEBREAK, confidence=row['conf'])
            )
        elif row['level'] == 5:
            print('  appending new word')
            document.pages[page_index].blocks[block_index].paragraphs[par_index].words.append(
                doc.Word(text=row['text'], bbox=bbox, confidence=row['conf'])
            )
        else:
            raise ValueError("Invalid row level")

    print(document)

    # # pd.options.display.width = 120
    # pd.set_option('display.width', 140)

    # for (pageid, blockid, paraid), gdf in pyt_df.groupby(by=['page_num', 'block_num', 'par_num']):
    #     print(pageid, blockid, paraid)
    #     print(gdf)
    #     print()

    # pages = []
    # # skip the first row because it's a special case
    # for page_id, page_df in pyt_df.groupby(by='page_num'):
    #     print(f'page_id: {page_id}')
    #     print(page_df)
    #     print()
    #     blocks = []
    #     for block_id, block_df in page_df.groupby(by='block_num'):
    #         print(f'block_id: {block_id}')
    #         print(block_df)
    #         print()
    #         paragraphs = []
    #
    #         for par_id, para_df in block_df.groupby(by='par_num'):
    #             print(f'par_id: {par_id}')
    #             print(para_df)
    #             print()
    #             words = _get_words_from_para_df(para_df)
    #
    #             # row = para_df.iloc[2]
    #             # print(f'para row: {row}')
    #             # assert row['level'] == 3
    #             # paragraphs.append(doc.Paragraph(
    #             #     words=words,
    #             #     bbox=_row_to_bbox(row),
    #             # ))
    #
    #
    #         # row = block_df.iloc[1]
    #         # print(f'block row: {row}')
    #         # assert row['level'] == 2
    #         # blocks.append(doc.Block(
    #         #     paragraphs=paragraphs,
    #         #     bbox=_row_to_bbox(block_df.iloc[0]),
    #         #     block_type=BLOCK_TYPE_TEXT_STR,
    #         # ))
    #         # paragraphs = []
    #
    #     # row = page_df.iloc[0]
    #     # print(f'page row: {row}')
    #     # assert row['level'] == 1
    #     # pages.append(doc.Page(
    #     #     blocks=blocks,
    #     #     bbox=_row_to_bbox(row),
    #     # ))
    #     # blocks = []
    # document = doc.Document(pages=pages)
    # pages = []

    # print(document)

    print('done')


# tesseract box/line painting example
# https://nanonets.com/blog/ocr-with-tesseract/#installingtesseract

