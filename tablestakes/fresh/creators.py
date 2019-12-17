import abc
from functools import partial
from typing import Callable, Union, Iterable, Optional, Dict, List, Any

import faker
import numpy as np

from tablestakes.fresh import html_css, kv, utils, chunks


# from tablestakes.fresh.kv import kv.ProbDict, kv.Kloc, kv.KVHtml


class Creator(abc.ABC):
    """Randomly generates values when called."""

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class ChoiceCreator(Creator):
    def __init__(self, choices: Union[list, kv.ProbDict]):
        if isinstance(choices, dict):
            choices = choices.keys()
            probs = choices.values()
            probs /= sum(probs)
        else:
            probs = None
        self.choices = choices
        self.probs = probs

    def __call__(self, *args, **kwargs):
        return np.random.choice(self.choices, p=self.probs)


class ConstantCreator(Creator):
    def __init__(self, constant: Any):
        self.constant = constant

    def __call__(self, *args, **kwargs):
        return self.constant


class Combiner(abc.ABC):
    @abc.abstractmethod
    def __call__(self, values: Iterable, *args, **kwargs):
        pass


class StrCombiner(abc.ABC):
    def __init__(self, join_str: str = ''):
        self.join_str = join_str

    def __call__(self, values: Iterable, *args, **kwargs):
        return self.join_str.join([str(v) for v in values])


class ParallelCreators(Creator):
    """Runs multiple creators independently and combines their results."""
    def __init__(
            self,
            creators: Iterable[Creator],
            combiner: Combiner = StrCombiner(''),
    ):
        self.creators = creators
        self.combiner = combiner

    def __call__(self, *args, **kwargs):
        return self.combiner([c() for c in self.creators])


class LambdaCreator(Creator):
    def __init__(self, fn: Callable):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn()


# class NestedCreators(Creator):
#     def __init__(
#             self,
#             creators: Union[List[Creator], Dict[Creator, float]],
#             initial_value=None,
#     ):
#         if isinstance(creators, dict):
#             self.creators = creators.keys()
#             self.probs = creators.values()
#         else:
#             self.creators = creators
#             self.probs = [1.0] * len(self.creators)
#
#         self.initial_value = initial_value
#
#     def __call__(self, *args, **kwargs):
#         value = self.initial_value
#         for c, p in zip(self.creators, self.probs):
#             if np.random.uniform() < p:
#                 value = c(value)
#         return value


class TagWrapperCreator(Creator):
    def __init__(
            self,
            tag_name: str,
            classes: html_css.HtmlClassesType = None,
            attributes: Optional[utils.StrDict] = None,
            probability_of_running = 1.0,
    ):
        self.tag_name = tag_name
        self.classes = classes
        self.attributes = attributes
        self.probability_of_running = probability_of_running

    def __call__(self, contents: html_css.DirtyHtmlChunk, *args, **kwargs):
        if np.random.uniform() < self.probability_of_running:
            return str(html_css.HtmlTag(
                tag_name=self.tag_name,
                contents=contents,
                classes=self.classes,
                attributes=self.attributes,
            ))
        else:
            return html_css.clean_dirty_html_chunk(contents)


class BoldTagWrapper(TagWrapperCreator):
    def __init__(self, probability_of_running=1.0):
        super().__init__('b', probability_of_running=probability_of_running)


class AbstractFakerCreator(Creator, abc.ABC):
    def __init__(self, seed: int = 42):
        self.faker = faker.Faker()
        self.faker.seed(seed)


class AddressCreator(AbstractFakerCreator):
    def __call__(self, *args, **kwargs):
        address = self.faker.address().replace("\n", "<br>\n")
        return f'{self.faker.company()}<br>\n{address}'


class DateCreator(AbstractFakerCreator):
    patterns = [
        '%Y-%m-%d',      # 2019-12-17
        '%Y.%m.%d',      # 2019.12.17
        '%m/%d/%Y',      # 12/17/2019
        '%m/%d/%y',      # 12/17/19
        '%b %d, %Y'      # Dec 17, 2019
        '%B %d, %Y'      # December 17, 2019
        '%A, %B %d, %Y'  # Monday, December 17, 2019
    ]

    def __init__(self, pattern: Optional[str]='%m/%d/%Y', seed=42):
        super().__init__(seed)
        if pattern is None:
            pattern = np.random.choice(self.patterns)
        self.pattern = pattern

    def __call__(self, *args, **kwargs):
        return self.faker.date(pattern=self.pattern)


class IntCreator(AbstractFakerCreator):
    def __init__(self, min=0, max=int(1e10), zero_pad_to_width: Optional[int]=None, seed=42):
        super().__init__(seed)
        self.min = min
        self.max = max
        self.zero_pad_to_width = zero_pad_to_width

    def __call__(self, *args, **kwargs):
        num = np.random.randint(self.min, self.max)
        if self.zero_pad_to_width is None:
            return f'{num:d}'
        else:
            return f'{num:0{self.zero_pad_to_width}d}'


class CssCreator(Creator):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> html_css.Css:
        pass


class KvCssCreator(Creator):
    def __init__(self, kv_loc: kv.KLoc, extra_css_creator: Optional[CssCreator] = None):
        # self.kv_name = kv_name
        self.kv_loc = kv_loc
        self.extra_css_creator = extra_css_creator

    def __call__(self, kv_name: Optional[str] = None, *args, **kwargs) -> html_css.Css:
        extra_css = html_css.Css()
        if self.extra_css_creator is not None:
            extra_css += self.extra_css_creator()
        return self.kv_loc.get_css(kv_name).add_style(extra_css)


class KeyValueCreator(Creator):
    """Just a holder for a kv_name and associated creators"""
    def __init__(
            self,
            name: str,
            key_contents_creator: Creator,
            value_contents_creator: Creator,
            style_creator: Optional[KvCssCreator] = None,
    ):
        self.name = name
        self.key_contents_gen = key_contents_creator
        self.value_contents_gen = value_contents_creator
        self.style_generator = style_creator

        self._html = None
        self._css = None

    def __call__(self, *args, **kwargs):
        return self.get_html(), self.get_css()

    def get_html(self) -> kv.KVHtml:
        if self._html is None:
            self._html = kv.KVHtml.from_strs(
                k_contents=self.key_contents_gen(),
                v_contents=self.value_contents_gen(),
                kv_name=self.name,
            )
        return self._html

    def get_css(self):
        if self.style_generator is None:
            return html_css.Css()

        if self._css is None:
            self._css = self.style_generator(self.name)
        return self._css


# class ContentsModifier(Callable):
#     @abc.abstractmethod
#     def __call__(self, contents: html_css.DirtyHtmlChunk, *args, **kwargs) -> html_css.DirtyHtmlChunk:
#         pass
#
#
# class ColonAdderModifier(ContentsModifier):
#     def __call__(self, contents: html_css.DirtyHtmlChunk, *args, **kwargs) -> str:
#         return f'{html_css.clean_dirty_html_chunk(contents)}:'
#
#
# class BoldWrapperModifier(ContentsModifier):
#     def __call__(self, contents: html_css.DirtyHtmlChunk, *args, **kwargs) -> str:
#         return f'<b>{html_css.clean_dirty_html_chunk(contents)}</b>'


if __name__ == '__main__':
    # # make a style generator that uses the KLoc CSS and adds extra styling to it
    #
    # # make a kv creator loop over KVConfigs
    #
    # # make a css grid
    # # pack css grid with KVs
    #
    # # kv_css = html_css.Css([
    # #     html_css.CssChunk(html_css.HtmlClass('asdf'), {'style': 'grid'}),
    # #     html_css.CssChunk(html_css.HtmlClass('2qwer'), {'style2': 'grid2'}),
    # # ])
    # # for chunk in kv_css:
    # #     print(chunk)
    #
    # colon_adder = html_css.CssChunk(
    #     html_css.CssSelector('.key.U:after'),
    #     {'content': ':'}
    # ),
    #
    # d = {
    #     'border-bottom-style': 'solid',
    #     'font-weight': 'bold',
    #     '': '',
    # }
    #
    # position_probabilities = {
    #     kv.KLoc.UL: 0.3,
    #     kv.KLoc.U:  1.0,
    #     kv.KLoc.UR: 0.01,
    #     kv.KLoc.R:  0.1,
    #     kv.KLoc.BR: 0.01,
    #     kv.KLoc.B:  0.2,
    #     kv.KLoc.BL: 0.01,
    #     kv.KLoc.L:  1.0,
    # }
    #
    # kvc = KeyValueCreator(
    #     name='receiving_address',
    #     key_contents_creator=ChoiceCreator(['Receiving', 'Receiving Address', 'Address To', 'To']),
    #     value_contents_creator=AddressCreator(),
    #     # TODO: don't add creator here, add CSS.
    #     #       CSS is created once by global random gen and then static?
    #     style_creator=KvCssCreator(kv.KLoc.L),
    # )
    # html, css = kvc()
    #
    # print('html:')
    # print(html)
    #
    # print('css:')
    # print(css)
    #
    # kvh = kv.KVHtml.from_strs('address_to', 'Address To:', '1232 Apache Ave </br>Santa Fe, NM 87505')
    #
    # # "https://css-tricks.com/snippets/css/complete-guide-grid/#prop-grid-auto-flow"
    #
    # kvc = KeyValueCreator(
    #     name='receiving_address',
    #     key_contents_creator=ChoiceCreator(['Receiving', 'Receiving Address', 'Address To', 'To']),
    #     value_contents_creator=AddressCreator(),
    #     # TODO: don't add creator here, add CSS.
    #     #       CSS is created once by global random gen and then static?
    #     # style_creator=KvCssCreator(kv.KLoc.L),
    # )
    # html, css = kvc()
    np.random.seed(42)

    my_date_creator = DateCreator()
    kv_creators = [
        KeyValueCreator(
            name='to_address',
            key_contents_creator=ChoiceCreator(['Receiving', 'Receiving Address', 'Sent To', 'To']),
            value_contents_creator=AddressCreator(),
        ),
        KeyValueCreator(
            name='sale_address',
            key_contents_creator=ChoiceCreator(['Sale Address', 'Sold To']),
            value_contents_creator=AddressCreator(),
        ),
        KeyValueCreator(
            name='from_address',
            key_contents_creator=ChoiceCreator(['Shipping', 'Shipping Address', 'From', 'Address From']),
            value_contents_creator=AddressCreator(),
        ),
        KeyValueCreator(
            name='date_sent',
            key_contents_creator=ChoiceCreator(['Sent', 'Date Sent', 'Statement Date']),
            value_contents_creator=my_date_creator,
        ),
        KeyValueCreator(
            name='date_received',
            key_contents_creator=ChoiceCreator(['Received', 'Date Received']),
            value_contents_creator=my_date_creator,
        ),
        KeyValueCreator(
            name='invoice_number',
            key_contents_creator=ChoiceCreator(['Invoice', 'Invoice number', 'Account']),
            value_contents_creator=IntCreator(),
        ),
    ]

    grid = html_css.Grid(classes=['maingrid'], num_rows=4, num_cols=4)
    for kvc in kv_creators:
        grid.add_both(*kvc())

    doc = html_css.Document()
    doc.add_styled_html(grid)

    kvcssc = KvCssCreator(kv.KLoc.U)
    doc.add_style(kvcssc())

    extra_css = html_css.Css([
        chunks.Keys.add_colon,
        chunks.Keys.bold,
        chunks.Body.font_sans_serif,
    ])
    doc.add_style(extra_css)

    print(str(doc))
    html_css.open_html_str(str(doc))

