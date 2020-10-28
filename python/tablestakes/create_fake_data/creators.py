import abc
from typing import Callable, Union, Iterable, Optional, Any, Tuple

import faker
from faker.providers import internet, phone_number

import numpy as np

from tablestakes import utils
from tablestakes.create_fake_data import kv, html_css as hc


class Creator(abc.ABC):
    """Randomly generates values when called."""

    def __call__(self, *args, **kwargs):
        return self._post_process(self._call_inner(*args, **kwargs))

    @abc.abstractmethod
    def _call_inner(self, *args, **kwargs):
        pass

    def _post_process(self, output):
        return output


class CssCreator(Creator):
    def __call__(self, *args, **kwargs) -> hc.Css:
        return self._call_inner(*args, **kwargs)

    @abc.abstractmethod
    def _call_inner(self, *args, **kwargs) -> hc.Css:
        pass


class ChoiceCreator(Creator):
    def __init__(self, choices: Union[list, kv.ProbDict]):
        if isinstance(choices, dict):
            probs = np.array([v for v in choices.values()])
            probs /= np.sum(probs)
            probs = list(probs)
            choices = [c for c in choices.keys()]
        else:
            probs = None
        self.choices = choices
        self.probs = probs

    def _call_inner(self, *args, **kwargs):
        try:
            c = np.random.choice(self.choices, p=self.probs)
        except:
            print(self.choices)
            print(self.probs)
            print('error')
        return c


class CssProbCreator(ChoiceCreator):
    def __init__(self, css: Union[hc.Css, hc.CssChunk], prob: float = 0.5):
        if isinstance(css, hc.CssChunk):
            css = hc.Css([css])
        choices = {
            css:            prob,
            hc.Css():   1 - prob,
        }
        super().__init__(choices)


class ConstantCreator(Creator):
    def __init__(self, constant: Any):
        self.constant = constant

    def _call_inner(self, *args, **kwargs):
        return self.constant


class Combiner(abc.ABC):
    @abc.abstractmethod
    def call_inner(self, values: Iterable, *args, **kwargs):
        pass


class StrCombiner(Combiner):
    def __init__(self, join_str: str = ''):
        self.join_str = join_str

    def call_inner(self, values: Iterable, *args, **kwargs):
        return self.join_str.join([str(v) for v in values])


class CssCombiner(Combiner):
    def call_inner(self, values: Iterable[hc.Css], *args, **kwargs):
        css = hc.Css()
        for c in values:
            css.add_style(c)
        return css


class ParallelCreators(Creator):
    """Runs multiple creators independently and combines their results."""
    def __init__(
            self,
            creators: Iterable[Creator],
            combiner: Combiner = StrCombiner(''),
    ):
        self.creators = creators
        self.combiner = combiner

    def _call_inner(self, *args, **kwargs):
        return self.combiner([c() for c in self.creators])


class LambdaCreator(Creator):
    def __init__(self, fn: Callable):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn()


class TagWrapperCreator(Creator):
    def __init__(
            self,
            tag_name: str,
            classes: hc.HtmlClassesType = None,
            attributes: Optional[utils.StrDict] = None,
            probability_of_running = 1.0,
    ):
        self.tag_name = tag_name
        self.classes = classes
        self.attributes = attributes
        self.probability_of_running = probability_of_running

    def _call_inner(self, contents: hc.DirtyHtmlChunk, *args, **kwargs):
        if np.random.uniform() < self.probability_of_running:
            return str(hc.HtmlTag(
                tag_name=self.tag_name,
                contents=contents,
                classes=self.classes,
                attributes=self.attributes,
            ))
        else:
            return hc.clean_dirty_html_chunk(contents)


class BoldTagWrapper(TagWrapperCreator):
    def __init__(self, probability_of_running=1.0):
        super().__init__('b', probability_of_running=probability_of_running)


class AbstractFakerCreator(Creator, abc.ABC):
    def __init__(self, seed: Optional[int] = None):
        self.faker = faker.Faker()
        self.faker.add_provider(internet)
        self.faker.add_provider(phone_number)
        if seed is not None:
            self.faker.seed(seed)


class AddressCreator(AbstractFakerCreator):
    def _call_inner(self, *args, **kwargs):
        address = self.faker.address().replace("\n", "<br>\n")
        return f'{self.faker.company()}<br>\n{address}'


class NameCreator(AbstractFakerCreator):
    def _call_inner(self, *args, **kwargs):
        return self.faker.name()


class DateCreator(AbstractFakerCreator):
    patterns = [
        '%Y-%m-%d',      # 2019-12-17
        '%Y.%m.%d',      # 2019.12.17
        '%m/%d/%Y',      # 12/17/2019
        '%m/%d/%y',      # 12/17/19
        '%b %d, %Y'      # Dec 17, 2019
        '%B %d, %Y'      # December 17, 2019
        # '%A, %B %d, %Y'  # Monday, December 17, 2019
    ]

    def __init__(self, pattern: Optional[str] = '%m/%d/%Y', seed=None):
        super().__init__(seed)
        if pattern is None:
            pattern = np.random.choice(self.patterns)
        self.pattern = pattern

    def _call_inner(self, *args, **kwargs):
        return self.faker.date(pattern=self.pattern)


class IntCreator(AbstractFakerCreator):
    def __init__(self, min=0, max=int(1e10), zero_pad_to_width: Optional[int]=None, seed=None):
        super().__init__(seed)
        self.min = min
        self.max = max
        self.zero_pad_to_width = zero_pad_to_width

    def _call_inner(self, *args, **kwargs):
        num = np.random.randint(self.min, self.max)
        if self.zero_pad_to_width is None:
            return f'{num:d}'
        else:
            return f'{num:0{self.zero_pad_to_width}d}'


class UrlCreator(AbstractFakerCreator):
    def _call_inner(self, *args, **kwargs):
        return self.faker.url()


class EmailCreator(AbstractFakerCreator):
    def _call_inner(self, *args, **kwargs):
        return self.faker.email()


class DollarsCreator(AbstractFakerCreator):
    def __init__(self, min=0, max=1e5, do_include_cents=True, do_include_dollar=True, seed=None):
        super().__init__(seed)
        self.min = min
        self.max = max
        self.do_include_cents = do_include_cents
        self.do_include_dollar = do_include_dollar

    def _call_inner(self, *args, **kwargs):
        range = self.max - self.min
        num = np.random.random() * range + self.min
        dollar_str = '$' if self.do_include_dollar else ''
        f_str = '.2f' if self.do_include_cents else '.0f'
        return f'{dollar_str}{num:{f_str}}'


class PhoneCreator(AbstractFakerCreator):
    DEFAULT_FORMATS = (
        '### ### ####',
        '(###) ### ####',
        '###-###-####',
        '(###) ###-####',
        '(###)###-####',
        '###.###.####',
        '(###) ###.####',
        '(###)###.####',
    )

    def __init__(self, formats=DEFAULT_FORMATS, seed=None):
        super().__init__(seed)
        self.format = self.faker.random_element(formats)

    def _phone_number(self):
        return self.faker.numerify(self.format)

    def _call_inner(self, *args, **kwargs):
        return self._phone_number()


class RandomStrCreator(AbstractFakerCreator):
    CHARS = 'qwertyuiopasdfghjkklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890'

    def __init__(self, min_chars_per_word=2, max_chars_per_word=10, min_words=1, max_words=10, seed=None):
        super().__init__(seed)
        self.min_chars_per_word = min_chars_per_word
        self.max_chars_per_word = max_chars_per_word
        self.min_words = min_words
        self.max_words = max_words

    def _call_inner(self, *args, **kwargs):
        num_words = np.random.randint(self.min_words, self.max_words, size=1)[0]
        words = [self._generate_word() for _ in range(num_words)]
        return ' '.join(words)

    def _generate_word(self):
        num_chars = np.random.randint(self.min_chars_per_word, self.max_chars_per_word)
        return ''.join(np.random.choice([c for c in self.CHARS], num_chars))


class KvCssCreator(Creator):
    def __init__(self, kv_loc: kv.KLoc, extra_css_creator: Optional[CssCreator] = None):
        self.kv_loc = kv_loc
        self.extra_css_creator = extra_css_creator

    def _call_inner(self, kv_name: Optional[str] = None, *args, **kwargs) -> hc.Css:
        extra_css = hc.Css()
        if self.extra_css_creator is not None:
            extra_css += self.extra_css_creator()
        return self.kv_loc.get_css(kv_name).add_style(extra_css)


class KvCreator(Creator):
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

    def _call_inner(self, *args, **kwargs) -> Tuple[kv.KvHtml, hc.Css]:
        return self.get_html(), self.get_css()

    def get_html(self) -> kv.KvHtml:
        if self._html is None:
            self._html = kv.KvHtml.from_strs(
                k_contents=self.key_contents_gen(),
                v_contents=self.value_contents_gen(),
                kv_name=self.name,
            )
        return self._html

    def get_css(self) -> hc.Css:
        if self.style_generator is None:
            return hc.Css()

        if self._css is None:
            self._css = self.style_generator(self.name)
        return self._css


if __name__ == '__main__':
    # TODO: make styles
    #   key box over value box (white and black key variants)
    #   k/v in same box, key small top left or top middle or left
    #   small key under underline value (left, center, right).  needs baseline font size
    #   key font variantes:
    #       bold, italics, bold / italics, smallcaps, all caps
    #   font size variants
    #       8-24 for key
    #       +/- 4 for value
    #   spacing variants
    #       horizontal left / center
    #       vertical left / center

    position_probabilities = {
        kv.KLoc.AL: 0.3,
        kv.KLoc.A:  1.0,
        kv.KLoc.AR: 0.01,
        kv.KLoc.R:  0.1,
        kv.KLoc.BR: 0.01,
        kv.KLoc.B:  0.2,
        kv.KLoc.BL: 0.01,
        kv.KLoc.L:  1.0,
    }
    np.random.seed(42)

    my_date_creator = DateCreator()
    kv_creators = [
        KvCreator(
            name='to_address',
            key_contents_creator=ChoiceCreator(['Receiving', 'Receiving Address', 'Sent To', 'To']),
            value_contents_creator=AddressCreator(),
        ),
        KvCreator(
            name='sale_address',
            key_contents_creator=ChoiceCreator(['Sale Address', 'Sold To']),
            value_contents_creator=AddressCreator(),
        ),
        KvCreator(
            name='from_address',
            key_contents_creator=ChoiceCreator(['Shipping', 'Shipping Address', 'From', 'Address From']),
            value_contents_creator=AddressCreator(),
        ),
        KvCreator(
            name='date_sent',
            key_contents_creator=ChoiceCreator(['Sent', 'Date Sent', 'Statement Date']),
            value_contents_creator=my_date_creator,
        ),
        KvCreator(
            name='date_received',
            key_contents_creator=ChoiceCreator(['Received', 'Date Received']),
            value_contents_creator=my_date_creator,
        ),
        KvCreator(
            name='invoice_number',
            key_contents_creator=ChoiceCreator(['Invoice', 'Invoice number', 'Account']),
            value_contents_creator=IntCreator(),
        ),
    ]

    grid = hc.Grid(classes=['maingrid'], num_rows=4, num_cols=4)
    for kvc in kv_creators:
        grid.add_both(*kvc())

    # assemble document
    doc = hc.Document()
    doc.add_styled_html(grid)

    kvcssc = KvCssCreator(kv.KLoc.A)
    doc.add_style(kvcssc())

    print(str(doc))
    hc.open_html_str(str(doc))

