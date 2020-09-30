import abc
from typing import List
import re

import numpy as np
import pandas as pd
from nltk.tokenize import WordPunctTokenizer

from tablestakes import ocr, utils


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


class CharCounter(DfModifier):
    PREFIX = 'num_chars_'
    LOWER_COUNT_NAME = f'{PREFIX}lower'
    UPPER_COUNT_NAME = f'{PREFIX}upper'
    NUMBER_COUNT_NAME = f'{PREFIX}number'
    OTHER_COUNT_NAME = f'{PREFIX}other'
    TOTAL_COUNT_NAME = f'{PREFIX}total'

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        TOKEN_COL_NAME = Tokenizer.TOKEN_NON_LOWERCASED_COL_NAME

        df[self.LOWER_COUNT_NAME] = df.apply(lambda row: sum(c.islower() for c in row[TOKEN_COL_NAME]), axis=1)
        totals = df[self.LOWER_COUNT_NAME].copy()

        df.loc[:, self.UPPER_COUNT_NAME] = df.apply(lambda row: sum(c.isupper() for c in row[TOKEN_COL_NAME]), axis=1)
        totals += df[self.UPPER_COUNT_NAME].copy()

        df.loc[:, self.NUMBER_COUNT_NAME] = df.apply(lambda row: sum(c.isdigit() for c in row[TOKEN_COL_NAME]), axis=1)
        totals += df[self.NUMBER_COUNT_NAME].copy()

        df.loc[:, self.TOTAL_COUNT_NAME] = df.apply(lambda row: len(row[TOKEN_COL_NAME]), axis=1)

        df.loc[:, self.OTHER_COUNT_NAME] = df.apply(lambda row: row[self.TOTAL_COUNT_NAME], axis=1) - totals

        return df


class DetailedOtherCharCounter(DfModifier):
    COUNT_TEMPLATE = f'{CharCounter.PREFIX}{{}}'
    CHARS_TO_CHECK = r"""~!@#$%^&*()_+{}[]\|;:"',.?<>/"""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        TOKEN_COL_NAME = Tokenizer.TOKEN_NON_LOWERCASED_COL_NAME

        for c in self.CHARS_TO_CHECK:
            attrib_name = self.COUNT_TEMPLATE.format(c)
            # attrib_name = html.escape(attrib_name)
            # attrib_name = attrib_name.replace('/', r'backslash')

            df[attrib_name] = df.apply(lambda row: sum(t == c for t in row[TOKEN_COL_NAME]), axis=1)

        return df


class Tokenizer(DfModifier):
    """Break whitespace separated strings into multiple rows."""
    ORIGINAL_INDEX_COL_NAME = 'pre_tokenized_row_index'

    ORIGINAL_TEXT_COL_NAME = ocr.OcrDfNames.TEXT
    TOKEN_NON_LOWERCASED_COL_NAME = 'token_non_lowercased'

    def __init__(self):
        self.tokenizer = WordPunctTokenizer()

    def _tokenize(self, s: str) -> List[str]:
        return self.tokenizer.tokenize(s)

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
        bbox_names = [
            ocr.OcrDfNames.LEFT,
            ocr.OcrDfNames.RIGHT,
            ocr.OcrDfNames.TOP,
            ocr.OcrDfNames.BOTTOM,
        ]

        tokenized_rows = []
        for index, row in df.iterrows():
            tokens = self._tokenize(row[self.ORIGINAL_TEXT_COL_NAME])
            bbox = tuple(n for n in row[bbox_names])
            bboxes = self._break_up_bounding_box(bbox, tokens)
            for token, bbox in zip(tokens, bboxes):
                row_copy = row.copy()
                row_copy[self.TOKEN_NON_LOWERCASED_COL_NAME] = token
                row_copy[bbox_names] = bbox
                row_copy[self.ORIGINAL_INDEX_COL_NAME] = index
                tokenized_rows.append(row_copy)

        out_df = pd.DataFrame(tokenized_rows)
        return out_df


class TokenPostProcessor(DfModifier):
    TOKEN_NON_LOWERCASED_COL_NAME = Tokenizer.TOKEN_NON_LOWERCASED_COL_NAME
    TOKEN_COL_NAME = 'token'

    NUMBER_REGEX = r'\d+'
    NUMBER_TOKEN = r'<NUMBER>'

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.TOKEN_COL_NAME] = df[self.TOKEN_NON_LOWERCASED_COL_NAME].apply(
            lambda token: self.NUMBER_TOKEN if re.match(self.NUMBER_REGEX, token) else token.lower()
        )
        return df


class Vocabulizer(DfModifier):
    """Convert tokens to vocab ids."""
    TOKEN_COL_NAME = TokenPostProcessor.TOKEN_COL_NAME
    VOCAB_ID_PRE_ELIMINATION_COL_NAME = 'vocab_id_pre_elimination'

    def __init__(self):
        self.word_to_id = {}
        self.word_to_count = {}

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for token in df[self.TOKEN_COL_NAME]:
            if token not in self.word_to_id:
                self.word_to_id[token] = len(self.word_to_id)
                self.word_to_count[token] = 0
            self.word_to_count[token] += 1
        df[self.VOCAB_ID_PRE_ELIMINATION_COL_NAME] = \
            df[self.TOKEN_COL_NAME].apply(lambda token: self.word_to_id[token])

        return df

    def get_vocab_size(self):
        return len(self.word_to_id)


class RareWordEliminator(DfModifier):
    UNKNOWN_TOKEN = '<UNKNOWN>'
    WORD_WAS_ELIMINATED_COL_NAME = 'was_removed_rare_token'
    VOCAB_ID_COL_NAME = 'vocab_id'

    def __init__(self, vocabulizer: Vocabulizer, min_count=2):
        self.min_count = min_count
        self.vocabulizer = vocabulizer

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        TOKEN_COL_NAME = TokenPostProcessor.TOKEN_COL_NAME
        VOCAB_COL_NAME = self.VOCAB_ID_COL_NAME

        counts = self.vocabulizer.word_to_count
        self.word_to_count = {k: v for k, v in sorted(counts.items()) if v >= self.min_count}
        self.word_to_id = {
            k: index
            for index, k, in enumerate(self.word_to_count.keys())
            if k in self.word_to_count
        }

        unknown_id = len(self.word_to_id)
        self.word_to_id[self.UNKNOWN_TOKEN] = unknown_id
        self.word_to_count[self.UNKNOWN_TOKEN] = \
            sum([v for v in self.vocabulizer.word_to_count.values() if v >= self.min_count])

        df[VOCAB_COL_NAME] = df[TOKEN_COL_NAME].apply(
            lambda token: self.word_to_id[token] if token in self.word_to_id else unknown_id,
        )

        df[self.WORD_WAS_ELIMINATED_COL_NAME] = (df[VOCAB_COL_NAME] == unknown_id).astype(np.int)

        return df


if __name__ == '__main__':
    t = Tokenizer()
    df = pd.DataFrame([
        {'left': 2000, 'right': 2200, 'top': 100, 'bottom': 150, 'text': 'blah'},
        {'left': 1000, 'right': 1200, 'top': 100, 'bottom': 150, 'text': 'asdf 4'},
        {'left': 2000, 'right': 2200, 'top': 100, 'bottom': 150, 'text': 'blah'},
    ])

    out = t(df)
    print(out)

    vocabulizer = Vocabulizer()
    out = vocabulizer(out)
    print(out)
