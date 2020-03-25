import copy
from typing import Any, List, Optional

from PIL import Image
import abc

import pandas as pd
import pdf2image
import pytesseract

from tablestakes import doc, utils


class OcrProvider(abc.ABC):
    def __init__(self, do_print_timing=True):
        self.do_print_timing = do_print_timing

    @classmethod
    def load_pdf_to_images(cls, pdf_filename: str, dpi: int):
        return pdf2image.convert_from_path(pdf_filename, dpi=dpi)

    def ocr(
            self,
            input_pdf: str,
            dpi=400,
            save_raw_ocr_output_location: Optional[str] = None,
    ) -> doc.Document:
        page_images = self.load_pdf_to_images(input_pdf, dpi)
        for page_ind, page_image in enumerate(page_images):
            ocr_page_outputs = []
            with utils.Timer(f"OCRing page {page_ind} of {len(page_images)}"):
                ocr_page_outputs.append(self._ocr_page_image(page_image))
            ocrd_pages_combined = self._combine_ocr_output(ocr_page_outputs)
            # TODO: remove this line when done debugging
            self.ocr_page_outputs = ocr_page_outputs
            self._save_ocr_output(ocrd_pages_combined, save_raw_ocr_output_location)
            return self._ocr_output_2_doc(ocrd_pages_combined)

    @abc.abstractmethod
    def _ocr_page_image(self, image: Image) -> Any:
        pass

    @abc.abstractmethod
    def _combine_ocr_output(self, ocr_page_outputs: List[Any]):
        pass

    @abc.abstractmethod
    def _save_ocr_output(self, ocr_output: Any, save_raw_ocr_output_location: Optional[str] = None):
        pass

    @abc.abstractmethod
    def _ocr_output_2_doc(self, ocr_output: Any) -> doc.Document:
        pass


class TesseractOcrProvider(OcrProvider):
    def _ocr_page_image(self, image: Image) -> pd.DataFrame:
        return pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DATAFRAME,
        )

    def _combine_ocr_output(self, ocr_page_outputs: List[pd.DataFrame]):
        for page_idx, df in enumerate(ocr_page_outputs):
            df['page_num'] = page_idx
        return pd.concat(ocr_page_outputs, axis=0)

    def _save_ocr_output(self, ocr_output: pd.DataFrame, save_raw_ocr_output_location: Optional[str] = None):
        ocr_output.to_pickle(save_raw_ocr_output_location)

    def _ocr_output_2_doc(self, ocr_output: pd.DataFrame) -> doc.Document:
        return TesseractDocumentFactory.tesseract_df_2_document(ocr_output)


class TesseractDocumentFactory:
    BLOCK_TYPE_TEXT_STR = 'TEXT'

    @staticmethod
    def _row_to_bbox(row: pd.Series):
        return doc.BBox(
            xmin=row['left'],
            xmax=row['left'] + row['width'],
            ymin=row['top'],
            ymax=row['top'] + row['height'],
        )

    @classmethod
    def tesseract_df_2_document(cls, pyt_df: pd.DataFrame) -> doc.Document:
        document = doc.Document(pages=[])
        bbox = None
        for row_id, row, in pyt_df.iterrows():
            page_index = row['page_num'] - 1
            block_index = row['block_num'] - 1
            par_index = row['par_num'] - 1

            prev_bbox = copy.copy(bbox)
            bbox = cls._row_to_bbox(row)
            if row['level'] == 1:
                document.pages.append(
                    doc.Page(blocks=[], bbox=bbox)
                )
            elif row['level'] == 2:
                document.pages[page_index].blocks.append(
                    doc.Block(paragraphs=[], bbox=bbox, block_type=cls.BLOCK_TYPE_TEXT_STR)
                )
            elif row['level'] == 3:
                document.pages[page_index].blocks[block_index].paragraphs.append(
                    doc.Paragraph(words=[], bbox=bbox)
                )
            elif row['level'] == 4:
                # tesseract lines become "\n" words
                prev_bbox.xmax = prev_bbox.xmin
                document.pages[page_index].blocks[block_index].paragraphs[par_index].words.append(
                    doc.Word(text='\n', bbox=prev_bbox, word_type=doc.Word.WordType.LINEBREAK, confidence=row['conf'])
                )
            elif row['level'] == 5:
                document.pages[page_index].blocks[block_index].paragraphs[par_index].words.append(
                    doc.Word(text=row['text'], bbox=bbox, confidence=row['conf'])
                )
            else:
                raise ValueError("Invalid row level")

        return document


if __name__ == '__main__':
    fullfile = '../output/sample_weasy_output.pdf'
    resolution = 400

    images = pdf2image.convert_from_path(fullfile, dpi=resolution)
    image = images[0]

    pyt_df = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
    document = TesseractDocumentFactory.tesseract_df_2_document(pyt_df)

    print(document)
    print('done')
