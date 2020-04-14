import abc
from typing import List

import pandas as pd

from tablestakes import color_matcher, ocr, utils, etree_modifiers


class DfModifier(abc.ABC):
    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def get_x_name(name: str):
        return color_matcher.WordColorMatcher.get_x_name(name)

    @staticmethod
    def get_y_name(name: str):
        return color_matcher.WordColorMatcher.get_y_name(name)


class DfModifierStack:
    def __init__(self, modifiers: List[DfModifier], do_use_timers=True):
        self.modifiers = modifiers

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for mod in self.modifiers:
            with utils.Timer(f'Df Modifier {mod.__class__.__name__}'):
                df = mod(df)

        return df


class ColSelector(DfModifier, abc.ABC):
    def __init__(self):
        self.output_col_names = []
        self.original_col_names = []


class XMaker(ColSelector):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        self.output_col_names = [
            ocr.OcrDfFactory.LEFT,
            ocr.OcrDfFactory.RIGHT,
            ocr.OcrDfFactory.TOP,
            ocr.OcrDfFactory.BOTTOM,
            ocr.OcrDfFactory.CONFIDENCE,
            ocr.OcrDfFactory.TEXT,
        ]
        self.original_col_names = [self.get_x_name(c) for c in self.output_col_names]

        # TODO: shared replace logic?
        df = df[self.original_col_names].copy()
        df.rename(columns=lambda name: name.replace(color_matcher.WordColorMatcher.OCR_SUFFIX, ''), inplace=True)

        return df


class YMaker(ColSelector):
    def __init__(self, do_include_key_value_cols=True, do_include_field_id_cols=True):
        super().__init__()
        self.do_include_field_id_cols = do_include_field_id_cols
        self.do_include_key_value_cols = do_include_key_value_cols

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.do_include_key_value_cols:
            # e.g.: 'isKey__y'
            self.original_col_names.append(self.get_y_name(etree_modifiers.SetIsKeyOnWordsModifier.KEY_NAME))

            # e.g.: 'isValue__y'
            self.original_col_names.append(self.get_y_name(etree_modifiers.SetIsValueOnWordsModifier.KEY_NAME))

        if self.do_include_field_id_cols:
            # e.g.: 'kv_is_'.  for selecting 'kv_is_to_address__y', 'kv_is_sale_address__y', etc.
            kv_is_prefix = etree_modifiers.ConvertParentClassNamesToWordAttribsModifier.TAG_PREFIX
            self.original_col_names.extend([c for c in df.columns if c.startswith(kv_is_prefix)])

        # TODO: shared replace logic?
        self.output_col_names = [
            name.replace(color_matcher.WordColorMatcher.WORDS_SUFFIX, '')
            for name in self.original_col_names
        ]

        df = df[self.original_col_names].copy()
        df.rename(columns=lambda name: name.replace(color_matcher.WordColorMatcher.WORDS_SUFFIX, ''), inplace=True)

        return df


class MetaMaker(ColSelector):
    def __init__(self, x_cols: List[str], y_cols: List[str]):
        super().__init__()
        self.x_cols = x_cols
        self.y_cols = y_cols

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in df.columns if c not in self.x_cols and c not in self.y_cols]

        df = df[cols].copy()
        df = df.rename(columns=lambda name: name.replace(color_matcher.WordColorMatcher.OCR_SUFFIX, ''))
        df = df.rename(columns=lambda name: name.replace(color_matcher.WordColorMatcher.WORDS_SUFFIX, ''))

        return df


class CharCounter(DfModifier):
    LOWER_COUNT_NAME = 'num_chars_lower'
    UPPER_COUNT_NAME = 'num_chars_upper'
    NUMBER_COUNT_NAME = 'num_chars_number'
    OTHER_COUNT_NAME = 'num_chars_other'

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        text_name = ocr.OcrDfFactory.TEXT

        df[self.LOWER_COUNT_NAME] = df.apply(lambda row: sum(c.islower() for c in row[text_name]), axis=1)
        totals = df[self.LOWER_COUNT_NAME].copy()

        df.loc[:, self.UPPER_COUNT_NAME] = df.apply(lambda row: sum(c.isupper() for c in row[text_name]), axis=1)
        totals += df[self.UPPER_COUNT_NAME].copy()

        df.loc[:, self.NUMBER_COUNT_NAME] = df.apply(lambda row: sum(c.isdigit() for c in row[text_name]), axis=1)
        totals += df[self.NUMBER_COUNT_NAME].copy()

        df.loc[:, self.OTHER_COUNT_NAME] = df.apply(lambda row: len(row[text_name]), axis=1) - totals

        return df


class DetailedOtherCharCounter(DfModifier):
    COUNT_TEMPLATE = 'num_chars_{}'
    CHARS_TO_CHECK = r"""~!@#$%^&*()_+{}[]\|;:"',.?<>/"""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        text_name = ocr.OcrDfFactory.TEXT

        for c in self.CHARS_TO_CHECK:
            attrib_name = self.COUNT_TEMPLATE.format(c)
            # attrib_name = html.escape(attrib_name)
            # attrib_name = attrib_name.replace('/', r'backslash')

            df[attrib_name] = df.apply(lambda row: sum(t == c for t in row[text_name]), axis=1)

        return df
