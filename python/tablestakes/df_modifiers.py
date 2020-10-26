import abc
import collections
import unicodedata
from typing import List, Dict
import re

import nltk
import numpy as np
import pandas as pd

import ray

from tablestakes import ocr, utils, constants

from transformers import BertTokenizerFast


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
                with utils.Timer(f'df modifier {mod.__class__.__name__}'):
                    if isinstance(mod, ray.actor.ActorHandle):
                        df = mod.__call__.remote(df)
                    else:
                        df = mod(df)
            else:
                if isinstance(mod, ray.actor.ActorHandle):
                    df = mod.__call__.remote(df)
                else:
                    df = mod(df)

        return df


class MyWordPunctTokenizer(nltk.RegexpTokenizer):
    def __init__(self):
        # nltk.RegexpTokenizer.__init__(self, r"""!@#$%^&*()_+\-=\[\]{};':"\|,.<>\\\/\?|\n|\w+|[^\w\s]""")
        # just like normal WordPunctTokenizer but adds newline as delimiter and breaks up blocks of special characters
        nltk.RegexpTokenizer.__init__(self, r"""\n|\w+|[^\w\s]""")


# noinspection PyTypeChecker
class CharCounter(DfModifier):
    ASCII_CHARS = set(
        """!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~""")

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        totals = 0

        df[constants.ColNames.HAS_LEADING_ZERO] = \
            df.apply(lambda row: int(row[constants.ColNames.TOKEN_RAW].startswith('0')), axis=1)

        df.loc[:, constants.ColNames.UNICODE_NORM_CHANGED_COUNT] = \
            df.apply(lambda row: sum(c != utils.unicode_norm(c) for c in row[constants.ColNames.TOKEN_RAW]), axis=1)

        df.loc[:, constants.ColNames.NONASCII_COUNT] = \
            df.apply(lambda row: sum(c not in self.ASCII_CHARS for c in row[constants.ColNames.TOKEN_RAW]), axis=1)

        df.loc[:, constants.ColNames.UPPER_COUNT] = \
            df.apply(lambda row: sum(c.isupper() for c in row[constants.ColNames.TOKEN_RAW]), axis=1)
        totals += df[constants.ColNames.UPPER_COUNT].copy()

        df.loc[:, constants.ColNames.NUMBER_COUNT] = \
            df.apply(lambda row: sum(c.isdigit() for c in row[constants.ColNames.TOKEN_RAW]), axis=1)
        totals += df[constants.ColNames.NUMBER_COUNT].copy()

        df.loc[:, constants.ColNames.TOTAL_COUNT] = \
            df.apply(lambda row: len(row[constants.ColNames.TOKEN_RAW]), axis=1)
        df.loc[:, constants.ColNames.OTHER_COUNT] = \
            df.apply(lambda row: row[constants.ColNames.TOTAL_COUNT], axis=1) - totals

        return df


# class DetailedOtherCharCounter(DfModifier):
#     COUNT_TEMPLATE = f'{constants.ColumnNames.CHAR_COUNT_PREFIX}{{}}'
#     CHARS_TO_CHECK = r"""~!@#$%^&*()_+{}[]\|;:"',.?<>/"""
#
#     def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
#         for c in self.CHARS_TO_CHECK:
#             attrib_name = self.COUNT_TEMPLATE.format(c)
#             # attrib_name = html.escape(attrib_name)
#             # attrib_name = attrib_name.replace('/', r'backslash')
#
#             df[attrib_name] = df.apply(lambda row: sum(t == c for t in row[constants.ColumnNames.TOKEN_RAW]), axis=1)
#
#         return df


# class TokenPostProcessor(DfModifier):
#     """unicode norm, lower case, and numbers --> <num>"""
#
#     NUMBER_REGEX = r'\d+'
#
#     def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
#
#         def proc(token):
#             m = re.match(self.NUMBER_REGEX, token)
#             if m:
#                 return constants.SpecialTokens.NUMBER
#             else:
#                 return utils.unicode_norm(token).lower()
#
#         df[constants.ColNames.TOKEN] = df[constants.ColNames.TOKEN_RAW].apply(
#             proc
#         )
#         return df


@ray.remote
class Vocabulizer(DfModifier):
    """Convert tokens to vocab ids."""

    def __init__(self):
        self._all_words = set()
        self._word_to_count = collections.defaultdict(int)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            for token in df[constants.ColNames.TOKEN]:
                self._inc_count(token)
        except KeyError as e:
            print(df.head())
            print(df.columns)
            raise e

        return df

    def _inc_count(self, token):
        if isinstance(token, float):
            print('vocab._inc_count found a float: ', token)
            raise ValueError()
        self._all_words.add(token)
        self._word_to_count[token] += 1

    def get_vocab_size(self):
        return len(self._word_to_id)

    def get_word_to_count(self):
        return self._word_to_count

    def get_all_words(self):
        try:
            return sorted(self._all_words)
        except BaseException as e:
            print(self._all_words)
            raise e


@ray.remote
class RareWordEliminator(DfModifier):
    def __init__(self, word_to_count: Dict[str, int], min_count: int = 2):
        """Call after the Vocabulizer has been run."""
        self.min_count = min_count
        self._word_to_count = None
        self._word_to_id = None
        self._counts = None

        self._counts = word_to_count

        self._word_to_count = {k: v for k, v in sorted(self._counts.items()) if v >= self.min_count}
        self._word_to_id = {
            k: index
            for index, k, in enumerate(self._word_to_count.keys())
        }

        self.UNKNOWN_ID = len(self._word_to_id)
        self._word_to_id[constants.SpecialTokens.UNKNOWN] = self.UNKNOWN_ID
        self._word_to_count[constants.SpecialTokens.UNKNOWN] = sum(
            [v for v in self._counts.values() if v >= self.min_count])

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df[constants.ColNames.TOKEN_ID] = df[constants.ColNames.TOKEN].apply(
            lambda token: self._word_to_id[token] if token in self._word_to_id else self.UNKNOWN_ID,
        )

        df[constants.ColNames.WORD_WAS_ELIMINATED] = \
            (df[constants.ColNames.TOKEN_ID] == self.UNKNOWN_ID).astype(np.int)

        return df

    def get_word_to_count(self):
        return self._word_to_count

    def get_word_to_id(self):
        return self._word_to_id


class Tokenizer(DfModifier):
    def __init__(self, tokenizer=None):
        if tokenizer is None:
            tokenizer = MyWordPunctTokenizer()
        self.tokenizer = tokenizer

    def _tokenize(self, s: str) -> List[str]:
        out = self.tokenizer.tokenize(s)
        return out

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
        bbox_names = constants.ColNames.BBOX_NAMES

        tokenized_rows = []
        for index, row in df.iterrows():
            original_text = row[constants.ColNames.TEXT]
            if isinstance(original_text, float) and np.isnan(original_text):
                continue
            tokens = self._tokenize(row[constants.ColNames.TEXT])
            bbox = tuple(n for n in row[bbox_names])
            bboxes = self._break_up_bounding_box(bbox, tokens)
            for token, bbox in zip(tokens, bboxes):
                row_copy = row.copy()
                row_copy[constants.ColNames.TOKEN_RAW] = token
                row_copy[bbox_names] = bbox
                row_copy[constants.ColNames.RAW_OCR_INDEX] = index
                tokenized_rows.append(row_copy)

        out_df = pd.DataFrame(tokenized_rows)
        return out_df


class MyBertTokenizer(DfModifier):
    VERSION = '0.1'
    FILL_NA_VALUE = -100

    def __init__(self, model_name='bert-base-uncased'):
        self.tok = self.get_tokenizer(model_name)

    @staticmethod
    def get_tokenizer(model_name='bert-base-uncased'):
        return BertTokenizerFast.from_pretrained(model_name)

    def get_name_str(self):
        return f'{self.__class__.__name__}-{self.VERSION}'

    @staticmethod
    def _shave(seq):
        ## ignore [cls] and [sep]
        return seq[1:-1]

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # some tokens are empty or have nan values. not sure why.
        df[constants.ColNames.RAW_OCR_INDEX] = df.index
        non_nan_rows = df[constants.ColNames.TEXT].isna().apply(lambda x: not x)
        df = df[non_nan_rows].copy()

        keep_rows = df[constants.ColNames.TEXT].apply(lambda t: re.match(r'\s+', t) is None and t != '')
        df = df[keep_rows].copy()

        org_text_values = list(df[constants.ColNames.TEXT])

        join_col = constants.ColNames.ORIGINAL_TEXT_INDEX
        new_df_ind = [int(i) for i in range(len(df))]
        df[join_col] = new_df_ind
        df.index = new_df_ind.copy()

        try:
            out = self.tok(
                org_text_values,
                is_split_into_words=True,
                return_offsets_mapping=True,
            )
        except BaseException as e:
            print(org_text_values)
            print(type(org_text_values))
            for ind, e in enumerate(org_text_values):
                if not isinstance(e, str):
                    print(f'element at position {ind}, is {e} which has type {type(e)}')
            raise e

        # offset_mapping: [(0, 1), (1, 4), (4, 5), (6, 9), (9, 10), (10, 13), (13, 14), (0, 2), (2, 4), (5, 6), (0, 4)]
        offset_mappings = out['offset_mapping']
        token_ids = out['input_ids']

        tokens = self.tok.convert_ids_to_tokens(token_ids)

        # [cls] and [sep]
        tok_cls = tokens[0]
        tok_sep = tokens[-1]
        org_text_values = [tok_cls] + org_text_values + [tok_sep]

        is_new_token = [om[0] == 0 for om in offset_mappings]
        org_text_inds = np.cumsum(is_new_token) - 1

        raw_tokens = [
            org_text_values[org_text_ind][om[0]:om[1]]
            for org_text_ind, om in zip(org_text_inds, offset_mappings)
        ]
        raw_tokens[0] = tok_cls
        raw_tokens[-1] = tok_sep

        # for the join, we don't want anything to match with [cls]
        org_text_inds -= 1

        tokens_df = pd.DataFrame(
            data=utils.invert_list_of_lists([org_text_inds, raw_tokens, tokens, token_ids]),
            columns=[
                constants.ColNames.ORIGINAL_TEXT_INDEX,
                constants.ColNames.TOKEN_RAW,
                constants.ColNames.TOKEN,
                constants.ColNames.TOKEN_ID,
            ],
        )

        tokens_df[constants.ColNames.TOKENIZER] = self.get_name_str()

        lsuffix = '_left'
        rsuffix = '_right'

        tokens_df = tokens_df.join(df, how='outer', on=join_col, lsuffix=lsuffix, rsuffix=rsuffix, sort=True)
        tokens_df.drop(f'{join_col}_left', axis=1, inplace=True)
        tokens_df = tokens_df.rename(
            columns={f'{join_col}_right': join_col},
        )

        tokens_df.fillna(self.FILL_NA_VALUE, inplace=True)

        return tokens_df


if __name__ == '__main__':
    t = MyBertTokenizer()
    df = pd.DataFrame([
        {'left': 2000, 'right': 2200, 'top': 100, 'bottom': 150, 'text': '(345) 234-1234'},
        {'left': 1000, 'right': 1200, 'top': 100, 'bottom': 150, 'text': 'ASDF 4'},
        {'left': 2000, 'right': 2200, 'top': 100, 'bottom': 150, 'text': 'blah'},
    ])

    pd.set_option('display.max_columns', 300)

    out = t(df)
    print(out)
