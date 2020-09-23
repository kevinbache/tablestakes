import pandas as pd

from tablestakes import df_modifiers


# def make_xymeta(joined_df: pd.DataFrame):
#     # Tokenizer change the number of rows of the DF if there are any rows with multi-word text
#     joined_df = df_modifiers.DfModifierStack([
#         df_modifiers.Tokenizer(),
#         df_modifiers.Vocabulizer(),
#     ])(joined_df)
#
#     x_maker = df_modifiers.XMaker()
#     x_df = df_modifiers.DfModifierStack([
#         x_maker,
#         df_modifiers.CharCounter(),
#         df_modifiers.DetailedOtherCharCounter(),
#     ])(joined_df)
#
#     y_maker = df_modifiers.YMaker(do_include_field_id_cols=False)
#     y_df = df_modifiers.DfModifierStack([
#         y_maker,
#     ])(joined_df)
#
#     meta_df = df_modifiers.DfModifierStack([
#         df_modifiers.MetaMaker(x_cols=x_maker.original_col_names, y_cols=y_maker.original_col_names),
#     ])(joined_df)
#
#     return x_df, y_df, meta_df
