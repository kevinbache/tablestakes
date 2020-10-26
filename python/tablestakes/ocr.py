import abc
from pathlib import Path
from typing import Any, List, Union

import pandas as pd
from PIL import Image
import pytesseract

from tablestakes import utils, constants


class OcrProvider(abc.ABC):
    def __init__(self, do_print_timing=True):
        self.do_print_timing = do_print_timing

    def ocr(
            self,
            page_image_files: List[str],
            save_raw_ocr_output_location: Union[Path, str],
            do_print_timings=False,
    ) -> pd.DataFrame:
        ocr_page_outputs = []
        for page_ind, page_image_file in enumerate(page_image_files):
            with utils.Timer(f"OCRing page {page_ind} of {len(page_image_files)}", do_print_outputs=do_print_timings):
                page_image = Image.open(page_image_file)
                ocr_page_outputs.append(self._ocr_page_image(page_image))
        ocrd_pages_combined = self._combine_ocr_output(ocr_page_outputs)
        self._save_ocr_output(ocrd_pages_combined, save_raw_ocr_output_location)
        return self._ocr_output_2_ocr_df(ocrd_pages_combined)

    @classmethod
    def add_page_stats_to_df(cls, df, page_images):
        for page_num, page_array in enumerate(page_images):
            this_page_rows_selector = df[constants.ColNames.PAGE_NUM] == page_num
            # create page height / width columns
            df.loc[this_page_rows_selector, [constants.ColNames.PAGE_HEIGHT, constants.ColNames.PAGE_WIDTH]] = \
                int(page_array.size[0]), int(page_array.size[1])
        # create num_pages column
        df[constants.ColNames.NUM_PAGES] = len(page_images)

    @classmethod
    def ocr_from_pdf(
            cls,
            pdf_file: utils.DirtyPath,
            dpi=400,
            do_print_timings=False,
    ) -> pd.DataFrame:
        with utils.Timer(f"Loading pdf to images: '{pdf_file}'", do_print_outputs=do_print_timings):
            page_images = utils.PdfHandler.load_pdf_to_images(pdf_filename=pdf_file, dpi=dpi)
        with utils.Timer("OCRing pages", do_print_outputs=do_print_timings):
            ocr_page_outputs = [cls._ocr_page_image(page_image) for page_image in page_images]
            ocrd_pages_combined = cls._combine_ocr_output(ocr_page_outputs)
            df = cls._ocr_output_2_ocr_df(ocrd_pages_combined)
            cls.add_page_stats_to_df(df, page_images)
            return df

    @classmethod
    @abc.abstractmethod
    def _ocr_page_image(cls, image: Image) -> Any:
        pass

    @classmethod
    @abc.abstractmethod
    def _combine_ocr_output(cls, ocr_page_outputs: List[Any]):
        pass

    @classmethod
    @abc.abstractmethod
    def _save_ocr_output(cls, ocr_output: pd.DataFrame, save_raw_ocr_output_location: Union[str, Path]):
        pass

    @classmethod
    @abc.abstractmethod
    def _ocr_output_2_ocr_df(cls, ocr_output: Any) -> pd.DataFrame:
        pass


class TesseractOcrProvider(OcrProvider):
    @classmethod
    def _ocr_page_image(cls, image: Image) -> pd.DataFrame:
        return pytesseract.image_to_data(
            image=image,
            # neither lang nor config gives a huge speedup.  8.5 --> 7.5 sec/full page
            lang='eng',
            config=r'-c tessedit_do_invert=0',
            output_type=pytesseract.Output.DATAFRAME,
        )

    @classmethod
    def _combine_ocr_output(cls, ocr_page_outputs: List[pd.DataFrame]):
        for page_idx, df in enumerate(ocr_page_outputs):
            df[constants.ColNames.PAGE_NUM] = page_idx
        return pd.concat(ocr_page_outputs, axis=0)

    @classmethod
    def _save_ocr_output(cls, ocr_output: pd.DataFrame, save_raw_ocr_output_location: Union[str, Path]):
        ocr_output.to_csv(save_raw_ocr_output_location)

    @classmethod
    def _ocr_output_2_ocr_df(cls, ocr_output: pd.DataFrame) -> pd.DataFrame:
        return TesseractOcrDfFactory.from_tesseract_df(ocr_output)


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

    @classmethod
    def from_tesseract_df(cls, pyt_df: pd.DataFrame):
        df = pyt_df[pyt_df['level'] == 5].copy()
        df['right'] = df['left'] + df['width']
        df['bottom'] = df['top'] + df['height']
        df.rename(columns={'conf': constants.ColNames.CONFIDENCE}, inplace=True)
        df = df[[
            constants.ColNames.PAGE_NUM,
            constants.ColNames.LEFT,
            constants.ColNames.RIGHT,
            constants.ColNames.TOP,
            constants.ColNames.BOTTOM,
            constants.ColNames.CONFIDENCE,
            constants.ColNames.TEXT,
        ]]
        return df
