import re
import tempfile
import webbrowser

import abc
import enum

from typing import List, Union, Optional

import pdfkit
import yattag

from tablestakes import utils


#################
# CSS Selectors #
#################
class CssSelector:
    """A raw CssSelector.  Can be a complex string."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.to_selector_str()

    def __eq__(self, other):
        return self.to_selector_str() == other.to_selector_str()

    def __hash__(self):
        return hash(self.to_selector_str())

    def to_selector_str(self):
        return self.name


class AbstractHtmlClasses(CssSelector, abc.ABC):
    """Abstract CssSelector for one or more classes, nested or all."""
    def __init__(self, classes: List[str]):
        super().__init__('')
        del self.name

        self.classes = classes

    @abc.abstractmethod
    def get_join_str(self): pass

    def to_selector_str(self): return self.get_join_str().join([f'.{c}' for c in self.classes])

    def get_class_list(self):
        return self.classes


class HtmlClassesAll(AbstractHtmlClasses):
    """Select all elements with all of these classes"""
    def get_join_str(self): return ''


class HtmlClass(HtmlClassesAll):
    """Select all elements with this class"""
    def __init__(self, name: str):
        super().__init__([name])


class HtmlClassesNested(AbstractHtmlClasses):
    """Select an element with class nested inside element with class (possibly more than 2 layers deep)"""
    def get_join_str(self): return ' '


class HtmlId(CssSelector):
    """Select the element with the given id."""
    def to_selector_str(self): return f'#{self.name}'


HtmlClassesType = Optional[Union[List[HtmlClass], AbstractHtmlClasses, List[str]]]

#####################
# End CSS Selectors #
#####################


####################
# Start CSS Things #
####################
class CssChunk:
    """Chunk of CSS with a single selector."""

    def __init__(self, selector: Union[CssSelector, str], values: utils.StrDict):
        if isinstance(selector, str):
            selector = CssSelector(selector)
        self.selector = selector
        self.values = values

    def __str__(self):
        dict_str = utils.dict_to_str(self.values, do_norm_key_width=False, line_end=';')
        return f'''{self.selector.to_selector_str()} {{
{dict_str}
}}'''

    def add_style(self, other: 'CssChunk'):
        if other.selector != self.selector:
            raise ValueError("You can only add two chunks whose selectors are equal.  "
                             "Use the CSS class instead too aggregate multiple CSSChunks "
                             "which have different selectors.")
        self.values.update(other.values)


class Css:
    """A group of CSS Chunks; i.e.: multiple selectors and values."""
    def __init__(self, chunks: Optional[List[CssChunk]] = None):
        self._selector_to_chunk = {}

        if chunks:
            for chunk in chunks:
                self._add_chunk(chunk)

    def _add_chunk(self, chunk: CssChunk):
        new_selector_str = chunk.selector.to_selector_str()
        if new_selector_str in self._selector_to_chunk:
            self._selector_to_chunk[new_selector_str].add_style(chunk)
        else:
            self._selector_to_chunk[new_selector_str] = chunk
        return self

    def add_style(self, css: Union[CssChunk, 'Css']):
        if isinstance(css, CssChunk):
            self._add_chunk(css)
        elif isinstance(css, Css):
            for chunk in css:
                self._add_chunk(chunk)
        else:
            raise ValueError(f'Got unknown css: {css} of type: {type(css)}.  Was expecting Css or CssChunk.')

        return self

    def __iter__(self):
        for chunk in self._selector_to_chunk.values():
            yield chunk

    def __repr__(self):
        """str(chunk) includes the selector"""
        return '\n\n'.join([str(chunk) for chunk in self])

##################
# End CSS Things #
##################


#####################
# Start HTML Things #
#####################
HtmlChunk = List[Union[str, 'HtmlTag']]
# an HtmlChunk which can also be a raw str or tag rather than a list of them.
DirtyHtmlChunk = Union[HtmlChunk, str, 'HtmlTag']


def check_dirty_html_chunk_type(chunk: DirtyHtmlChunk):
    if isinstance(chunk, list):
        for e in chunk:
            if not isinstance(e, (str, HtmlTag)):
                ValueError(f'Found found element, {e} in DirtyHtmlChunk of type {type(e)} but if you pass a list, '
                           f'all elements should be strs or HtmlTags.')
    elif not isinstance(chunk, (str, HtmlTag)):
        ValueError(f'Passed chunk, {chunk} is of type {type(chunk)} but it should be either a str, HtmlTag, '
                   f'or list mixing those two.')


def clean_dirty_html_chunk(chunk: DirtyHtmlChunk):
    if isinstance(chunk, (str, HtmlTag)):
        chunk = [chunk]
    if not isinstance(chunk, list):
        raise ValueError(f'chunk is type: {type(chunk)}')
    return chunk


def _html_chunk_to_str(chunk: HtmlChunk, join_str='\n'):
    return join_str.join([str(t) for t in chunk])


# class SelectableHtml(abc.ABC):
#     @abc.abstractmethod
#     def get_selector(self, sector_type: enum.Enum):
#         pass
#     def get_class_based_selector():
#         pass
#     def get_id_based_selector():
#         pass


def does_contain_an_html_tag(s: str) -> bool:
    return re.search(r'(<.*>)', s) is not None


# TODO: convert to SelectableHtml?
class HtmlTag:
    def __init__(
            self,
            tag_name: str,
            contents: DirtyHtmlChunk,
            classes: HtmlClassesType = None,
            attributes: Optional[utils.StrDict] = None,
    ):
        self.tag_name = tag_name
        self.contents = clean_dirty_html_chunk(contents)
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
        elif isinstance(classes, str):
            self.classes = [HtmlClass(classes)]
        else:
            raise ValueError(f'Got unknown classes variable, {classes}, of type: {type(classes)}.')
        self.attributes = attributes or {}

    def get_class_list(self):
        return self.classes

    def get_class_str(self):
        return ' '.join([str(c) for c in self.get_class_list()])

    def add_contents(self, other: DirtyHtmlChunk):
        check_dirty_html_chunk_type(other)
        self.contents.extend(clean_dirty_html_chunk(other))

    def __str__(self):
        doc = yattag.Doc()

        class_str = self.get_class_str()
        if class_str:
            tag = doc.tag(self.tag_name, klass=class_str, **self.attributes)
        else:
            tag = doc.tag(self.tag_name, **self.attributes)

        with tag:
            doc.asis(_html_chunk_to_str(self.contents))

        return yattag.indent(doc.getvalue())


class Div(HtmlTag):
    def __init__(
            self,
            contents: DirtyHtmlChunk,
            classes: HtmlClassesType = None,
            attributes: Optional[utils.StrDict] = None,
    ):
        super().__init__('div', contents, classes, attributes)


class Span(HtmlTag):
    def __init__(
            self,
            contents: DirtyHtmlChunk,
            classes: HtmlClassesType = None,
            attributes: Optional[utils.StrDict] = None,
    ):
        super().__init__('span', contents, classes, attributes)


class Body(HtmlTag):
    def __init__(
            self,
            contents: DirtyHtmlChunk,
            classes: HtmlClassesType = None,
            attributes: Optional[utils.StrDict] = None,
    ):
        super().__init__('body', contents, classes, attributes)


class Style(HtmlTag):
    def __init__(
            self,
            contents: DirtyHtmlChunk,
            classes: HtmlClassesType = None,
            attributes: Optional[utils.StrDict] = None,
    ):
        super().__init__('style', contents, classes, attributes)


class Head(HtmlTag):
    def __init__(
            self,
            contents: DirtyHtmlChunk,
            classes: HtmlClassesType = None,
            attributes: Optional[utils.StrDict] = None,
    ):
        super().__init__('head', contents, classes, attributes)


class Html(HtmlTag):
    def __init__(
            self,
            contents: DirtyHtmlChunk,
            classes: HtmlClassesType = None,
            attributes: Optional[utils.StrDict] = None,
    ):
        super().__init__('html', contents, classes, attributes)


class Table(HtmlTag):
    def __init__(
            self,
            contents: DirtyHtmlChunk,
            classes: HtmlClassesType = None,
            attributes: Optional[utils.StrDict] = None,
    ):
        super().__init__('table', contents, classes, attributes)


class Tr(HtmlTag):
    def __init__(
            self,
            contents: DirtyHtmlChunk,
            classes: HtmlClassesType = None,
            attributes: Optional[utils.StrDict] = None,
    ):
        super().__init__('tr', contents, classes, attributes)


class Td(HtmlTag):
    def __init__(
            self,
            contents: DirtyHtmlChunk,
            classes: HtmlClassesType = None,
            attributes: Optional[utils.StrDict] = None,
    ):
        super().__init__('td', contents, classes, attributes)

###################
# End HTML Things #
###################


######################
# Start Combo Things #
######################
# TODO: subclass HtmlTag?
class StyledHtmlTag:
    # todo: this should really be something like: HtmlTag, CssChunk (selector is taken from tag)
    def __init__(self, html_tag: HtmlTag, css: Css):
        self.html_tag = html_tag
        self.css = css

    # @classmethod
    # def div_from_contents_and_cssdict(
    #         cls,
    #         class_names: HtmlClassesType,
    #         html_contents: DirtyHtmlChunk,
    #         css_values: utils.StrDict,
    #         attributes: Optional[utils.StrDict] = None,
    # ):

    def get_classes_list(self):
        return self.html_tag.get_class_list()

    def add_contents(self, html_chunk: DirtyHtmlChunk):
        check_dirty_html_chunk_type(html_chunk)
        self.html_tag.add_contents(html_chunk)

    def add_style(self, css: Union[Css, CssChunk]):
        self.css.add_style(css)

    def add_both(self, html_chunk: DirtyHtmlChunk, css: Union[Css, CssChunk]):
        self.add_contents(html_chunk)
        self.add_style(css)

    def get_html(self) -> HtmlTag:
        return self.html_tag

    def get_css(self) -> Css:
        return self.css


class Grid(StyledHtmlTag):
    def __init__(
            self,
            classes: HtmlClassesType,
            num_rows: int,
            num_cols: int,
            auto_flow: str = 'row',
            extra_css_values: Optional[utils.StrDict] = None,
    ):
        html = Div(contents='', classes=classes)

        values = {
            'display': 'grid',
            'grid-template-columns': ' '.join(['auto'] * num_cols),
            'grid-template-rows': ' '.join(['auto'] * num_rows),
            'grid-auto-flow': auto_flow,
            'grid-gap': '15px 15px'
        }
        if extra_css_values:
            values.update(extra_css_values)
        css = Css([
            CssChunk(
                selector=HtmlClassesAll(classes),
                values=values,
            ),
        ])

        super().__init__(html_tag=html, css=css)


class PageSize(enum.Enum):
    A4 = 'A4',
    LETTER = 'Letter',

    def __init__(self, size_str: str):
        self.size_str = size_str


class BaseCss(enum.Enum):
    FIREFOX = 'default_firefox_css.css'
    CHROME = 'default_chrome_css.css'

    def __init__(self, filename: str):
        self.filename = filename

    def __str__(self):
        return utils.read_txt(self.filename)


# TODO: subclass StyledHtmlTag?  StyledHtmlContents?
class Document:
    def __init__(self, contents: Optional[DirtyHtmlChunk] = None, css: Optional[Css] = None):
        if contents is None:
            contents = []
        self.contents = clean_dirty_html_chunk(contents)

        if css is None:
            css = Css()
        self.css = css

    def add_contents(self, html_chunk: DirtyHtmlChunk):
        check_dirty_html_chunk_type(html_chunk)
        self.contents.extend(clean_dirty_html_chunk(html_chunk))

    def add_style(self, css: Union[Css, CssChunk]):
        self.css.add_style(css)

    def add_styled_html(self, styled_html: StyledHtmlTag):
        self.add_contents(styled_html.get_html())
        self.add_style(styled_html.get_css())

    def __str__(self):
        html = Html([
            Head(['<meta charset="utf-8"/>', Style(f'\n{str(self.css)}\n')]),
            Body(_html_chunk_to_str(self.contents)),
        ])

        return str(html)

    def open_in_browser(self):
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
            url = 'file://' + f.name
            f.write(str(self))
            webbrowser.open(url)
            f.close()

    def save_pdf(self, output_fullfile: str, page_size=PageSize.LETTER, margin='1in') -> Optional[str]:
        options = {
            'page-size': page_size.size_str,
        }
        for side in ('top', 'bottom', 'left', 'right'):
            options[f'margin-{side}'] = margin
        return pdfkit.from_string(
            input=str(self),
            output_path=output_fullfile,
            options=options,
        )

    def save_html(self, output_fullfile: str):
        utils.save_txt(output_fullfile, str(self))



'''
<!DOCTYPE html>
<html>
    <style>
    </style>
    <body>
        <h1>My First Heading</h1>
        <p>My first paragraph.</p>
    </body>
</html>
'''
####################
# End Combo Things #
####################


def open_html_str(html_str: str, do_print_too=True):
    """open a string in a browser"""
    import tempfile
    import webbrowser

    if do_print_too:
        print(html_str)

    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        url = 'file://' + f.name
        f.write(html_str)
    webbrowser.open(url)


class SelectorType(enum.Enum):
    """Represents regions within a KV Group"""
    GROUP = (1, None, )
    TRS_IN_GROUP = (2, None, )
    TDS_IN_GROUP = (3, None, )

    # DIVs holding Ks and Vs, present even in tables
    KV_CONTAINER = (4, 'kv_container', )
    KEY = (5, 'kv_key',)
    VALUE = (6, 'kv_value',)
    KEY_OUTER = (7, 'kv_key_container',)
    VALUE_OUTER = (8, 'kv_value_container',)

    # can be a TR for ATable or a TD for LTable
    TABLE_KEY_HOLDER = (9, 'table_key_holder', )
    TABLE_VALUE_HOLDER = (10, 'table_value_holder', )

    def __init__(self, id: int, html_class_name: str):
        # need ids so equality tests don't fail with the three None classes
        self.id = id
        self.html_class_name = html_class_name
