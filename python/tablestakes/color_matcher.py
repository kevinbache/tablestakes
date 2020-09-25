from pathlib import Path
from typing import List, Union, Dict

from tablestakes import ocr, utils, etree_modifiers

import numpy as np
import pandas as pd


class WordColorMatcher:
    """Matches ocr words to true words via parallel color lookup images"""

    @classmethod
    def get_colors_under_word(cls, colored_page_image_arrays: List[np.ndarray], row: pd.Series):
        """row from the ocr df (ocr.csv) representing a word and including fields for page_num, LRTB"""
        page_num = row[ocr.TesseractOcrProvider.PAGE_NUM_COL_NAME]
        if page_num > len(colored_page_image_arrays):
            raise ValueError(f"page_num: {page_num}, len(page_image_arrays): {len(colored_page_image_arrays)}")

        page = colored_page_image_arrays[page_num]

        word_slice = page[row[ocr.TesseractOcrDfFactory.TOP]:row[ocr.TesseractOcrDfFactory.BOTTOM], :]
        word_slice = word_slice[:, row[ocr.TesseractOcrDfFactory.LEFT]:row[ocr.TesseractOcrDfFactory.RIGHT]]

        return {
            'median': np.median(word_slice, axis=(0, 1)),
            'mean': np.mean(word_slice, axis=(0, 1)),
        }

    @classmethod
    def _get_color_block_stats_under_ocr_words(
            cls,
            ocr_df: pd.DataFrame,
            colored_page_image_files: List[Union[Path, str]],
    ):
        colored_page_image_arrays = utils.load_image_files_to_arrays(colored_page_image_files)

        color_stats_of_ocr_boxes = [
            cls.get_colors_under_word(colored_page_image_arrays, row)
            for _, row in ocr_df.iterrows()
        ]

        return color_stats_of_ocr_boxes

    @classmethod
    def _identify_ocr_words_by_color(cls, words_df: pd.DataFrame, color_stats_of_ocr_boxes: List[Dict]):
        #############################
        # get canonical word colors #
        #############################
        word_colors_df = words_df[etree_modifiers.WordColorizer.RGB]
        word_colors_df = word_colors_df.apply(pd.to_numeric)

        #########################################
        # find matching words for each ocr word #
        #########################################
        word_ids = []
        dists = []
        for ocr_word_ind, color_stat_of_ocr_box in enumerate(color_stats_of_ocr_boxes):
            # Todo: save mean and median?  median will be all 0s, mean is useful for diagnosis. maybe track differences.
            distances_from_current_ocr_word_to_each_words_color = \
                np.linalg.norm(color_stat_of_ocr_box['median'] - word_colors_df, ord=1, axis=1)
            index_of_closest_color = np.argmin(distances_from_current_ocr_word_to_each_words_color)
            mae_to_closest_color = distances_from_current_ocr_word_to_each_words_color[index_of_closest_color]
            # TODO: Factor out id str
            min_word_id = words_df.iloc[index_of_closest_color][etree_modifiers.WordWrapper.WORD_ID_ATTRIB_NAME]
            # print(f'min_index: {index_of_closest_color}, min_mae: {mae_to_closest_color}, min_word_id: {min_word_id}')
            word_ids.append(min_word_id)
            dists.append(mae_to_closest_color)

        return word_ids, dists

    @classmethod
    def get_joined_df(
            cls,
            ocr_df: pd.DataFrame,
            words_df: pd.DataFrame,
            colored_page_image_files: List[Union[Path, str]],
    ) -> pd.DataFrame:
        """DOES modify dataframes"""
        color_stats_of_ocr_boxes = cls._get_color_block_stats_under_ocr_words(ocr_df, colored_page_image_files)
        word_ids, dists = cls._identify_ocr_words_by_color(words_df, color_stats_of_ocr_boxes)

        # col names
        CLOSEST_WORD_ID = 'closest_color_word_id'
        CLOSEST_DIST = 'closest_color_dist'

        WORD_ID_NAME = etree_modifiers.WordWrapper.WORD_ID_ATTRIB_NAME

        ocr_df[CLOSEST_WORD_ID] = word_ids
        ocr_df[CLOSEST_DIST] = dists

        joined_df = pd.merge(
            ocr_df,
            words_df,
            how='outer',
            left_on=CLOSEST_WORD_ID,
            right_on=WORD_ID_NAME,
        )

        # todo: more formal error checking for unmatched rows.  this will fail if there are any.
        # try:
        #     joined_df['text_lev_distance'] = \
        #         joined_df.apply(
        #             lambda row: utils.levenshtein(
        #                 row[f'{TEXT_NAME}{cls.OCR_SUFFIX}'].lower(),
        #                 row[f'{TEXT_NAME}{cls.WORDS_SUFFIX}'].lower()
        #             ),
        #             axis=1,
        #         )
        # except BaseException as e:
        #     raise e

        return joined_df
