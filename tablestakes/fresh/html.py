from typing import *

import yattag

from tablestakes.fresh import utils


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


class CssSelector:
    def __init__(self, name: str):
        self.name = name

    def to_selector(self): return self.name

    def __repr__(self):
        return self.name


class HtmlClass(CssSelector):
    def to_selector(self): return f'.{self.name}'


class HtmlId(CssSelector):
    def to_selector(self): return f'#{self.name}'


HtmlClasses = Union[List[HtmlClass], HtmlClass]
CssSelector = Union[HtmlClass, HtmlId, str]


class HtmlTag:
    def __init__(
            self,
            tag_name: str,
            contents: DirtyHtmlChunk,
            classes: HtmlClasses,
            attributes: Optional[utils.StrDict] = None,
    ):
        self.tag_name = tag_name
        self.contents = _clean_dirty_html_chunk(contents)
        self.classes = utils.to_list(classes)
        self.attributes = attributes or {}

    def get_class_str(self):
        return ' '.join([str(c) for c in self.classes])

    def __str__(self):
        doc = yattag.Doc()

        with doc.tag(self.tag_name, klass=self.get_class_str(), **self.attributes):
            doc.asis(_html_chunk_to_str(self.contents))

        return yattag.indent(doc.getvalue())


class Div(HtmlTag):
    def __init__(self, contents: DirtyHtmlChunk, classes: HtmlClasses, attributes: Optional[utils.StrDict] = None):
        super().__init__('div', contents, classes, attributes)


class KVHtml(Div):
    def __init__(
            self,
            k_tag: HtmlTag,
            v_tag: HtmlTag,
            container_classes: HtmlClasses,
            container_attributes: Optional[utils.StrDict] = None,
    ):
        super().__init__([k_tag, v_tag], container_classes, container_attributes)
        self._k_tag = k_tag
        self._v_tag = v_tag

    @staticmethod
    def _get_container_class(kv_name: str):
        return HtmlClass(f'{kv_name}_container')

    @staticmethod
    def _get_key_class(kv_name: str):
        return HtmlClass(f'{kv_name}_key')

    @staticmethod
    def _get_value_class(kv_name: str):
        return HtmlClass(f'{kv_name}_value')

    @classmethod
    def from_strs(
            cls,
            kv_name: str,
            k_contents: DirtyHtmlChunk,
            v_contents: DirtyHtmlChunk,
    ) -> 'KVHtml':
        return cls(
            container_classes=[cls._get_container_class(kv_name)],
            k_tag=Div(k_contents, classes=[cls._get_key_class(kv_name)]),
            v_tag=Div(v_contents, classes=[cls._get_value_class(kv_name)]),
        )

    def get_container_classes(self):
        return self.classes

    def get_key_classes(self):
        return self._k_tag.classes

    def get_value_classes(self):
        return self._v_tag.classes


if __name__ == '__main__':
    kvh = KVHtml.from_strs('address_to', 'Address To:', '1232 Apache Ave </br>Santa Fe, NM 87505')
    print(kvh)
    print(kvh.get_container_classes())
    print(kvh.get_key_classes())
    print(kvh.get_value_classes())
