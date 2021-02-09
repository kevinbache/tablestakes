import abc
from typing import List
import re
import sys

import nltk
import numpy as np
import pandas as pd

import ray

from tablestakes import utils, constants

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

        df.loc[:, constants.ColNames.LOWER_COUNT] = \
            df.apply(lambda row: sum(c.islower() for c in row[constants.ColNames.TOKEN_RAW]), axis=1)
        totals += df[constants.ColNames.LOWER_COUNT].copy()

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


class MyBertTokenizer(DfModifier):
    VERSION = '0.2'

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

        # throw out rows with bad data
        keep_rows = df[constants.ColNames.TEXT].apply(lambda t: re.match(r'\s+', t) is None and t != '')
        df = df[keep_rows].copy()

        org_text_values = list(df[constants.ColNames.TEXT])

        join_col = constants.ColNames.ORIGINAL_TEXT_INDEX
        new_df_ind = [int(i) for i in range(len(df))]
        df[join_col] = new_df_ind
        df.index = new_df_ind.copy()

        # doesn't work
        # import warnings
        # with warnings.catch_warnings():
        #     warnings.filterwarnings(
        #         "ignore",
        #         message=r".*indices sequence length is longer.*",
        #         append=True,
        #     )
        #     out = self.tok(
        #         org_text_values,
        #         is_split_into_words=True,
        #         return_offsets_mapping=True,
        #         # max_length=sys.maxsize,
        #         truncation=False,
        #         padding=False,
        #     )

        try:
            out = self.tok(
                org_text_values,
                is_split_into_words=True,
                return_offsets_mapping=True,
                # max_length=sys.maxsize,
                truncation=False,
                padding=False,
            )
        except BaseException as e:
            print("MyBertTokenizer exception!!")
            print(org_text_values)
            print(type(org_text_values))
            for ind, e in enumerate (org_text_values):
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

        tokens_df.fillna(self.tok.mask_token_id, inplace=True)

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
