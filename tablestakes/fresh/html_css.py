import abc
from typing import List, Union, Optional

import yattag

from tablestakes.fresh import utils


class CssSelector:
    """Raw CssSelector.  Can be a complex string."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(self.to_selector_str())

    def to_selector_str(self): return self.name


class AbstractHtmlClasses(CssSelector, abc.ABC):
    def __init__(self, classes: List[str]):
        super().__init__('')
        del self.name

        self.classes = classes

    def __repr__(self):
        return self.classes

    @abc.abstractmethod
    def get_join_str(self): pass

    def to_selector_str(self): return self.get_join_str().join([f'.{c}' for c in self.classes])

    def get_class_list(self):
        return self.classes


class HtmlClassesAll(AbstractHtmlClasses):
    def get_join_str(self): return ''


class HtmlClass(HtmlClassesAll):
    def __init__(self, name: str):
        super().__init__([name])


class HtmlClassesNested(AbstractHtmlClasses):
    def get_join_str(self): return ' '


class HtmlId(CssSelector):
    def to_selector_str(self): return f'#{self.name}'


HtmlClassesType = Optional[Union[List[HtmlClass], AbstractHtmlClasses]]


class CssChunk:
    """Chunk of CSS with a single selector."""

    def __init__(self, selector: CssSelector, values: utils.StrDict):
        self.selector = selector
        self.values = values

    def __str__(self):
        dict_str = utils.dict_to_str(self.values, do_norm_key_width=False, line_end=';')
        return f'''{self.selector.to_selector_str()} {{
{dict_str}
}}'''

    def __add__(self, other: 'CssChunk'):
        if other.selector != self.selector:
            raise ValueError("You can only add two chunks whose selectors are equal.  "
                             "Use the CSS class instead too aggregate multiple CSSChunks "
                             "which have different selectors.")


class Css:
    """A group of CSS Chunks; i.e.: multiple selectors and values."""
    def __init__(self, chunks: Optional[List[CssChunk]] = None):
        self._selector_to_chunk = {}

        if chunks:
            for chunk in chunks:
                self._selector_to_chunk[chunk.selector] = chunk

    def _add_chunk(self, chunk: CssChunk):
        if chunk.selector in self._selector_to_chunk:
            self._selector_to_chunk[chunk.selector] += chunk
        else:
            self._selector_to_chunk[chunk.selector] = chunk
        return self

    def __add__(self, other: Union[CssChunk, 'Css']):
        if isinstance(other, CssChunk):
            self._add_chunk(other)
        elif isinstance(other, Css):
            for chunk in other:
                self._add_chunk(chunk)
        return self

    def __iter__(self):
        for chunk in self._selector_to_chunk.values():
            yield chunk

    def __repr__(self):
        """str(chunk) includes the selector"""
        return '\n\n'.join([str(chunk) for chunk in self])


HtmlChunk = List[Union[str, 'HtmlTag']]
# an HtmlChunk which can also be a raw str or tag rather than a list of them.
DirtyHtmlChunk = Union[HtmlChunk, str, 'HtmlTag']


def _clean_dirty_html_chunk(chunk: DirtyHtmlChunk):
    if isinstance(chunk, (str, HtmlTag)):
        chunk = [chunk]
    if not isinstance(chunk, list):
        raise ValueError(f'chunk is type: {type(chunk)}')
    return chunk


def _html_chunk_to_str(chunk: HtmlChunk):
    return ''.join([str(t) for t in chunk])


class HtmlTag:
    def __init__(
            self,
            tag_name: str,
            contents: DirtyHtmlChunk,
            classes: HtmlClassesType,
            attributes: Optional[utils.StrDict] = None,
    ):
        self.tag_name = tag_name
        self.contents = _clean_dirty_html_chunk(contents)
        if classes is None:
            self.classes = []
        elif isinstance(classes, AbstractHtmlClasses):
            self.classes = classes.get_class_list()
        elif isinstance(classes, list):
            self.classes = []
            for c in classes:
                if isinstance(c, HtmlClass):
                    self.classes.append(c.name)
                elif isinstance(c, str):
                    self.classes.append(c)
                else:
                    raise ValueError(f'Got unknown classes type: {type(c)}.  Classes: {classes}')
        self.attributes = attributes or {}

    def get_class_str(self):
        return ' '.join([str(c) for c in self.classes])

    def __str__(self):
        doc = yattag.Doc()

        with doc.tag(self.tag_name, klass=self.get_class_str(), **self.attributes):
            doc.asis(_html_chunk_to_str(self.contents))

        return yattag.indent(doc.getvalue())


class Div(HtmlTag):
    def __init__(self, contents: DirtyHtmlChunk, classes: HtmlClassesType, attributes: Optional[utils.StrDict] = None):
        super().__init__('div', contents, classes, attributes)

class Body(HtmlTag):
    def __init__(self, contents: DirtyHtmlChunk, classes: HtmlClassesType, attributes: Optional[utils.StrDict] = None):
        super().__init__('body', contents, classes, attributes)


class Document:
    def __init__(self, contents: DirtyHtmlChunk, style: Css):
        self.body = Body(contents, classes=None)

    def __str__(self):
        doc = yattag.Doc()
        doc.asis('<!DOCTYPE html>')

        with doc.tag('html'):
            pass

        return yattag.indent(doc.getvalue())


    '''
<!DOCTYPE html>
<html>
<body>

<h1>My First Heading</h1>

<p>My first paragraph.</p>

</body>
</html>
    '''
