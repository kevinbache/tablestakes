import abc
from typing import Callable, Union, Iterable

import faker
import numpy as np

from tablestakes.fresh import html_css, kv
# from tablestakes.fresh.kv import kv.ProbDict, kv.Kloc, kv.KVHtml


class Creator(abc.ABC):
    """Randomly generates values when called."""

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class LambdaCreator(Creator):
    def __init__(self, fn: Callable):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn()


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


class Combiner(abc.ABC):
    @abc.abstractmethod
    def __call__(self, values: Iterable, *args, **kwargs):
        pass


class StrCombiner(abc.ABC):
    def __init__(self, join_str: str = ''):
        self.join_str = join_str

    def __call__(self, values: Iterable, *args, **kwargs):
        return self.join_str.join([str(v) for v in values])


class MultiCreator(Creator):
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


class AbstractFakerCreator(Creator, abc.ABC):
    def __init__(self, seed: int = 42):
        self.faker = faker.Faker()
        self.faker.seed(seed)


class AddressCreator(AbstractFakerCreator):
    def __call__(self, *args, **kwargs):
        address = self.faker.address().replace("\n", "<br>\n")
        return f'{self.faker.company()}<br>\n{address}'


class CssCreator(Creator):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> html_css.Css:
        pass


class KvCssCreator(Creator):
    def __init__(self, kv_name: str, kv_loc: kv.Kloc, extra_css_creator: CssCreator):
        self.kv_name = kv_name
        self.kv_loc = kv_loc
        self.extra_css_creator = extra_css_creator

    def __call__(self, *args, **kwargs) -> html_css.Css:
        return self.kv_loc.get_css(self.kv_name) + self.extra_css_creator()


class KeyValueCreator(Creator):
    """Just a holder for a kv_name and associated creators"""
    def __init__(
            self,
            name: str,
            key_contents_creator: Creator,
            value_contents_creator: Creator,
            style_creator: CssCreator,
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
            self._html = kv.KVHtml.from_strs(self.name, self.key_contents_gen(), self.value_contents_gen())
        return self._html

    def get_css(self):
        if self._css is None:
            self._css = self.style_generator()
        return self._css