import abc
from typing import List
import re

import pandas as pd
from nltk.tokenize import WordPunctTokenizer

from tablestakes import color_matcher, ocr, utils, etree_modifiers


class DfModifier(abc.ABC):
    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DfModifierStack:
    def __init__(self, modifiers: List[DfModifier], do_use_timers=True):
        self.modifiers = modifiers
        self.do_use_timers = do_use_timers

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for mod in self.modifiers:
            if self.do_use_timers:
                with utils.Timer(f'Df Modifier {mod.__class__.__name__}'):
                    df = mod(df)
            else:
                df = mod(df)

        return df


class ColSelector(DfModifier, abc.ABC):
    def __init__(self):
        self.col_names = []

    @staticmethod
    def _get_df(df, col_names):
        return df[col_names].copy()


class XMaker(ColSelector):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # don't include TEXT -- we're gonna use vocab id instead from the Vocabulizer but it's going in a separate df
        self.col_names = [
            ocr.OcrDfFactory.LEFT,
            ocr.OcrDfFactory.RIGHT,
            ocr.OcrDfFactory.TOP,
            ocr.OcrDfFactory.BOTTOM,
            ocr.OcrDfFactory.CONFIDENCE,
        ]
        self.col_names += [c for c in df.columns if c.startswith(CharCounter.PREFIX)]

        return self._get_df(df, self.col_names)


class XVocabMaker(ColSelector):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        self.col_names = [Vocabulizer.VOCAB_NAME]
        return self._get_df(df, self.col_names)


class YMaker(ColSelector):
    def __init__(self, do_include_key_value_cols=True, do_include_field_id_cols=True):
        super().__init__()
        self.do_include_field_id_cols = do_include_field_id_cols
        self.do_include_key_value_cols = do_include_key_value_cols

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.do_include_key_value_cols:
            # e.g.: 'isKey'
            self.col_names.append(self.get_y_name(etree_modifiers.SetIsKeyOnWordsModifier.KEY_NAME))

            # e.g.: 'isValue'
            self.col_names.append(self.get_y_name(etree_modifiers.SetIsValueOnWordsModifier.KEY_NAME))

        if self.do_include_field_id_cols:
            # e.g.: 'kv_is_'.  for selecting 'kv_is_to_address, 'kv_is_sale_address', etc.
            kv_is_prefix = etree_modifiers.ConvertParentClassNamesToWordAttribsModifier.TAG_PREFIX
            self.col_names.extend([c for c in df.columns if c.startswith(kv_is_prefix)])

        return self._get_df(df, self.col_names)


class MetaMaker(ColSelector):
    def __init__(self, ignore_cols: List[str]):
        super().__init__()
        self.ignore_cols = ignore_cols

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        self.col_names = [c for c in df.columns if c not in self.ignore_cols]
        return self._get_df(df, self.col_names)


class CharCounter(DfModifier):
    PREFIX = 'num_chars_'
    LOWER_COUNT_NAME = f'{PREFIX}lower'
    UPPER_COUNT_NAME = f'{PREFIX}upper'
    NUMBER_COUNT_NAME = f'{PREFIX}number'
    OTHER_COUNT_NAME = f'{PREFIX}other'
    TOTAL_COUNT_NAME = f'{PREFIX}total'

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        text_name = ocr.OcrDfFactory.TEXT

        df[self.LOWER_COUNT_NAME] = df.apply(lambda row: sum(c.islower() for c in row[text_name]), axis=1)
        totals = df[self.LOWER_COUNT_NAME].copy()

        df.loc[:, self.UPPER_COUNT_NAME] = df.apply(lambda row: sum(c.isupper() for c in row[text_name]), axis=1)
        totals += df[self.UPPER_COUNT_NAME].copy()

        df.loc[:, self.NUMBER_COUNT_NAME] = df.apply(lambda row: sum(c.isdigit() for c in row[text_name]), axis=1)
        totals += df[self.NUMBER_COUNT_NAME].copy()

        df.loc[:, self.TOTAL_COUNT_NAME] = df.apply(lambda row: len(row[text_name]), axis=1)

        df.loc[:, self.OTHER_COUNT_NAME] = df.apply(lambda row: row[self.TOTAL_COUNT_NAME], axis=1) - totals

        return df


class DetailedOtherCharCounter(DfModifier):
    COUNT_TEMPLATE = f'{CharCounter.PREFIX}{{}}'
    CHARS_TO_CHECK = r"""~!@#$%^&*()_+{}[]\|;:"',.?<>/"""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        text_name = ocr.OcrDfFactory.TEXT

        for c in self.CHARS_TO_CHECK:
            attrib_name = self.COUNT_TEMPLATE.format(c)
            # attrib_name = html.escape(attrib_name)
            # attrib_name = attrib_name.replace('/', r'backslash')

            df[attrib_name] = df.apply(lambda row: sum(t == c for t in row[text_name]), axis=1)

        return df


class Tokenizer(DfModifier):
    """Break whitespace separated strings into multiple rows."""
    NUMBER_REGEX = r'\d+'
    NUMBER_TOKEN = r'<NUMBER>'
    ORIGINAL_INDEX_COL_NAME = 'pre_tokenized_row_index'

    def __init__(self):
        self.tokenizer = WordPunctTokenizer()

    def _tokenize(self, s: str) -> List[str]:
        tokens = self.tokenizer.tokenize(s)
        return [
            self.NUMBER_TOKEN if re.match(self.NUMBER_REGEX, token)
            else token
            for token in tokens
        ]

    @staticmethod
    def _break_up_bounding_box(bbox, tokens):
        # dirty heuristic: break up bbox assuming each character has equal width
        left, right, top, bottom = bbox
        if len(tokens) == 1:
            return [(left, right, top, bottom), ]

        # num characters including dividing spaces
        num_total_chars = sum(len(token) for token in tokens) + len(tokens) - 1

        each_char_width = (right - left) / num_total_chars
        token_widths = [len(token) * each_char_width for token in tokens]
        token_widths = [int(tw) for tw in token_widths]

        current_left = left
        bboxes = []
        for token_width in token_widths:
            current_right = current_left + token_width
            bboxes.append((current_left, current_right, top, bottom))
            current_left = int(current_right + each_char_width)

        return bboxes

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        text_name = ocr.OcrDfFactory.TEXT
        bbox_names = [
            ocr.OcrDfFactory.LEFT,
            ocr.OcrDfFactory.RIGHT,
            ocr.OcrDfFactory.TOP,
            ocr.OcrDfFactory.BOTTOM,
        ]

        tokenized_rows = []
        for index, row in df.iterrows():
            tokens = self._tokenize(row[text_name])
            bbox = tuple(n for n in row[bbox_names])
            bboxes = self._break_up_bounding_box(bbox, tokens)
            for token, bbox in zip(tokens, bboxes):
                row_copy = row.copy()
                row_copy[text_name] = token
                row_copy[bbox_names] = bbox
                row_copy[self.ORIGINAL_INDEX_COL_NAME] = index
                tokenized_rows.append(row_copy)

        out_df = pd.DataFrame(tokenized_rows)
        return out_df


class Vocabulizer(DfModifier):
    """Convert tokens to vocab ids."""

    TEXT_NAME = ocr.OcrDfFactory.TEXT
    VOCAB_NAME = 'vocab_id'

    def __init__(self):
        self.vocab = {}

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for token in df[self.TEXT_NAME]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        df[self.VOCAB_NAME] = df[self.TEXT_NAME].apply(lambda token: self.vocab[token])
        return df

    def get_vocab_size(self):
        return len(self.vocab)


if __name__ == '__main__':
    t = Tokenizer()
    df = pd.DataFrame([
        {'left': 2000, 'right': 2200, 'top': 100, 'bottom': 150, 'text': 'blah'},
        {'left': 1000, 'right': 1200, 'top': 100, 'bottom': 150, 'text': 'asdf 4'},
        {'left': 2000, 'right': 2200, 'top': 100, 'bottom': 150, 'text': 'blah'},
    ])

    out = t(df)
    print(out)

    v = Vocabulizer()
    out = v(out)
    print(out)
