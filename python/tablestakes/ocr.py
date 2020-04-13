import abc
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import pandas as pd
from PIL import Image
import pytesseract

from tablestakes import utils


class OcrProvider(abc.ABC):
    def __init__(self, do_print_timing=True):
        self.do_print_timing = do_print_timing

    def ocr(
            self,
            page_image_files: List[str],
            save_raw_ocr_output_location: Union[Path, str],
    ) -> pd.DataFrame:
        ocr_page_outputs = []
        for page_ind, page_image_file in enumerate(page_image_files):
            with utils.Timer(f"OCRing page {page_ind} of {len(page_image_files)}"):
                page_image = Image.open(page_image_file)
                ocr_page_outputs.append(self._ocr_page_image(page_image))
        ocrd_pages_combined = self._combine_ocr_output(ocr_page_outputs)
        self._save_ocr_output(ocrd_pages_combined, save_raw_ocr_output_location)
        return self._ocr_output_2_ocr_df(ocrd_pages_combined)

    @abc.abstractmethod
    def _ocr_page_image(self, image: Image) -> Any:
        pass

    @abc.abstractmethod
    def _combine_ocr_output(self, ocr_page_outputs: List[Any]):
        pass

    @abc.abstractmethod
    def _save_ocr_output(self, ocr_output: pd.DataFrame, save_raw_ocr_output_location: Union[str, Path]):
        pass

    # @abc.abstractmethod
    # def _ocr_output_2_doc(self, ocr_output: Any) -> doc.Document:
    #     pass

    @abc.abstractmethod
    def _ocr_output_2_ocr_df(self, ocr_output: Any) -> pd.DataFrame:
        pass


class TesseractOcrProvider(OcrProvider):
    PAGE_NUM_COL_NAME = 'page_num'

    def _ocr_page_image(self, image: Image) -> pd.DataFrame:
        return pytesseract.image_to_data(
            image=image,
            output_type=pytesseract.Output.DATAFRAME,
        )

    def _combine_ocr_output(self, ocr_page_outputs: List[pd.DataFrame]):
        for page_idx, df in enumerate(ocr_page_outputs):
            df[self.PAGE_NUM_COL_NAME] = page_idx
        return pd.concat(ocr_page_outputs, axis=0)

    def _save_ocr_output(self, ocr_output: pd.DataFrame, save_raw_ocr_output_location: Union[str, Path]):
        ocr_output.to_csv(save_raw_ocr_output_location)

    # def _ocr_output_2_doc(self, ocr_output: pd.DataFrame) -> doc.Document:
    #     return TesseractDocumentFactory.tesseract_df_2_document(ocr_output)

    def _ocr_output_2_ocr_df(self, ocr_output: pd.DataFrame) -> pd.DataFrame:
        return TesseractOcrDfFactory.from_tesseract_df(ocr_output)


# class TesseractDocumentFactory:
#     BLOCK_TYPE_TEXT_STR = 'TEXT'
#
#     @staticmethod
#     def _tesseract_row_to_bbox(row: pd.Series):
#         return doc.BBox(
#             left=row['left'],
#             right=row['left'] + row['width'],
#             top=row['top'],
#             bottom=row['top'] + row['height'],
#         )
#
#     @classmethod
#     def tesseract_df_2_document(cls, pyt_df: pd.DataFrame) -> doc.Document:
#         document = doc.Document(pages=[])
#         bbox = None
#         for row_id, row, in pyt_df.iterrows():
#             page_index = row['page_num'] - 1
#             block_index = row['block_num'] - 1
#             par_index = row['par_num'] - 1
#
#             prev_bbox = copy.copy(bbox)
#             bbox = cls._tesseract_row_to_bbox(row)
#             if row['level'] == 1:
#                 document.pages.append(
#                     doc.Page(blocks=[], bbox=bbox)
#                 )
#             elif row['level'] == 2:
#                 document.pages[page_index].blocks.append(
#                     doc.Block(paragraphs=[], bbox=bbox, block_type=cls.BLOCK_TYPE_TEXT_STR)
#                 )
#             elif row['level'] == 3:
#                 document.pages[page_index].blocks[block_index].paragraphs.append(
#                     doc.Paragraph(words=[], bbox=bbox)
#                 )
#             elif row['level'] == 4:
#                 # tesseract lines become "\n" words
#                 prev_bbox.right = prev_bbox.left
#                 document.pages[page_index].blocks[block_index].paragraphs[par_index].words.append(
#                     doc.Word(text='\n', bbox=prev_bbox, word_type=doc.Word.WordType.LINEBREAK, confidence=row['conf'])
#                 )
#             elif row['level'] == 5:
#                 document.pages[page_index].blocks[block_index].paragraphs[par_index].words.append(
#                     doc.Word(text=row['text'], bbox=bbox, confidence=row['conf'])
#                 )
#             else:
#                 raise ValueError("Invalid row level")
#
#         return document


class TesseractOcrDfFactory:
    """
    example Tesseract DataFrame:

    ,level,page_num,block_num,par_num,line_num,word_num,left,top,width,height,conf,text
    0,1,0,0,0,0,0,0,0,8500,11000,-1,
    1,2,0,1,0,0,0,2107,2067,624,101,-1,
    2,3,0,1,1,0,0,2107,2067,624,101,-1,
    3,4,0,1,1,1,0,2107,2067,624,101,-1,
    4,5,0,1,1,1,1,2107,2067,379,101,95,SENT
    5,5,0,1,1,1,2,2536,2067,195,101,96,TO
    """
    LEFT = 'left'
    RIGHT = 'right'
    TOP = 'top'
    BOTTOM = 'bottom'

    @classmethod
    def from_tesseract_df(cls, pyt_df: pd.DataFrame):
        df = pyt_df[pyt_df['level'] == 5].copy()
        df['right'] = df['left'] + df['width']
        df['bottom'] = df['top'] + df['height']
        df.rename(columns={'conf': 'confidence'}, inplace=True)
        df = df[[
            TesseractOcrProvider.PAGE_NUM_COL_NAME,
            cls.LEFT,
            cls.RIGHT,
            cls.TOP,
            cls.BOTTOM,
            'confidence',
            'text',
        ]]
        return df


def calibrate_conversion_parameters(words_df: pd.DataFrame, ocr_df: pd.DataFrame):
    ocr_df = ocr_df.copy()
    word_text_lowered = words_df['text'].str.lower()

    closest_word_indices = \
        ocr_df['text'].apply(lambda ot: np.argmin([utils.levenshtein(ot.lower(), wt) for wt in word_text_lowered]))

    words_df_matched = words_df.loc[closest_word_indices]

    from sklearn.linear_model import LassoCV
    models = []
    for col in ('left', 'right', 'top', 'bottom'):
        x = words_df_matched[col].to_numpy().reshape(-1, 1)
        y = ocr_df[col]

        model = LassoCV(fit_intercept=True)
        model.fit(x, y)
        y_hat = model.predict(x)
        mae = np.mean(np.abs(y_hat - y))

        models.append({
            'col': col,
            'intercept': model.intercept_,
            'coef': model.coef_[0],
            'mean_absolute_error': mae,
        })

    return pd.DataFrame(models)


if __name__ == '__main__':
    import pandas as pd

    ocr_csv_file = '/python/tablestakes/scripts/generate_ocrd_doc_2/docs/doc_01_bak/ocr.csv'
    words_csv_file = '/python/tablestakes/scripts/generate_ocrd_doc_2/docs/doc_01_bak/words.csv'

    ocr_df = pd.read_csv(ocr_csv_file)
    words_df = pd.read_csv(words_csv_file)

    models = calibrate_conversion_parameters(words_df, ocr_df)
    print(models)
