from collections import defaultdict
from typing import *

import yattag


class HtmlClass:
    def __init__(self, name: Text):
        self.name = name

    def __str__(self):
        return self.name


ClassesType = Optional[Union[List[HtmlClass], Text, List[Text]]]
TextTag = Union[Text, 'HtmlTag']
TextTagList = Union[List[TextTag], TextTag]
TextDict = Dict[Text, Text]


class HtmlTagClass:
    def __init__(
            self,
            name: Text,
            classes: ClassesType = None,
            contents: Optional[TextTagList] = None,
            attributes: Optional[TextDict] = None,
    ):
        if isinstance(classes, str):
            classes = [HtmlClass(classes)]
        elif isinstance(classes, list):
            out = []
            for c in classes:
                if isinstance(c, str):
                    out.append(HtmlClass(c))
                elif isinstance(c, HtmlClass):
                    out.append(c)
                else:
                    raise ValueError(f'Unknown class entry: {c}')
        else:
            raise ValueError(f'Unknown classes: {classes}')

        self.name = name
        self.classes = classes
        self.contents = self.add_contents(contents)
        self.attributes = attributes or {}

        self.styles = StyleTag()

    def __str__(self):
        doc = yattag.Doc()

        class_str = ' '.join(str(c) for c in self.classes)

        with doc.tag(self.name, klass=class_str, **self.attributes):
            doc.asis(''.join([str(c) for c in self.contents]))

        return yattag.indent(doc.getvalue())

    def add_class(self, class_name: HtmlClass):
        self.classes.append(class_name)

    @classmethod
    def _massage_contents(cls, contents: Optional[TextTagList]):
        if not contents:
            contents = []

        if isinstance(contents, str) or isinstance(contents, cls):
            contents = [contents]
        elif not isinstance(contents, list):
            raise ValueError(f"Got unknown contents type: {type(contents)}.  Contents: {contents}.")

        return contents

    def add_contents(self, contents: Optional[TextTagList]):
        contents = self._massage_contents(contents)
        self.contents.extend(contents)

    def add_style(self, style: Dict[Text, TextDict]):
        self.styles.update(style)


class HtmlTag(HtmlTagClass):
    def __init__(
            self,
            classes: ClassesType = None,
            contents: Optional[TextTagList] = None,
            attributes: Optional[TextDict] = None,
    ):
        super().__init__('html', classes, contents, attributes)


class BodyTag(HtmlTagClass):
    def __init__(
            self,
            classes: ClassesType = None,
            contents: Optional[TextTagList] = None,
            attributes: Optional[TextDict] = None,
    ):
        super().__init__('body', classes, contents, attributes)


class HeadTag(HtmlTagClass):
    def __init__(
            self,
            classes: ClassesType = None,
            contents: Optional[TextTagList] = None,
            attributes: Optional[TextDict] = None,
    ):
        super().__init__('head', classes, contents, attributes)


class StyleTag(HtmlTagClass):
    def __init__(
            self,
            classes: ClassesType = None,
            contents: Optional[TextTagList] = None,
            attributes: Optional[TextDict] = None,
    ):
        super().__init__('style', classes, contents, attributes)


class DivTag(HtmlTagClass):
    def __init__(
            self,
            classes: ClassesType = None,
            contents: Optional[TextTagList] = None,
            attributes: Optional[TextDict] = None,
    ):
        super().__init__('div', classes, contents, attributes)

# Represents a CSS doc.  Maps CSS selector to dict of CSS {k: v}s
SelectorDict = Dict[Text, TextDict]


class CssStyles:
    def __init__(self, d: Optional[SelectorDict] = None):
        self.d = defaultdict(lambda: {})
        self.update(d)

    def update(self, other: 'CssStyles'):
        for selector, css_dict in other.d.items():
            self.d[selector].update(css_dict)

    # def
    #     for selector, css_dict in other.d.items():
    #         self.d[selector].update(css_dict)



class HtmlDoc:
    def __init__(self, ):
        self.body = []
        self.styles = {}

    def add_body(self, contents: TextTagList):
        pass

    def add_styles(self, selector_to_kv_tags: Dict[Text, TextDict]):
        pass

    def styles_str(self) -> str:
        out = '\n\n'.join(self._tag_dict_2_str(d))
        return out

    @classmethod
    def _selector_and_tag_dict_2_str(cls, selector: Text, d: TextDict):
        return f'{selector} {cls._tag_dict_2_str(d)}'

    @classmethod
    def _tag_dict_2_str(cls, d: TextDict):
        x = ';\n  '.join([f'{str(k)}: {str(v)}' for k, v in d.items()])
        return f'{{\n  {x};\n}}'

    def __str__(self):
        HtmlTag(contents=[
            HeadTag(contents=[
                StyleTag(contents=self.styles),
            ]),
            BodyTag(contents=self.contents),
        ])

        doc = yattag.Doc()

        class_str = ' '.join(str(c) for c in self.classes)

        with doc.tag(self.name, klass=class_str, **self.attributes):
            doc.asis(''.join([str(c) for c in self.contents]))

        return yattag.indent(doc.getvalue())


if __name__ == '__main__':
    print(HtmlDoc._tag_dict_2_str({'k': 'v', 'k2': 'v2'}))


# if __name__ == '__main__':
#     key_div = DivTag(classes='key', contents=['Key'])
#     value_div = DivTag(classes='value', contents=['Value 1<br>Value 2'])
#     kv_div = DivTag(classes='container_class', contents=[key_div, value_div])
#
#     '''
#     i want to be able to say key_div.get_styles()
#     i want to construct styles block from all the tags in the html.
#     that'd be something like:
#
#     styles = Styles()
#     for tag in tree_traversal():
#         styles.add(tag.get_identifier(), tag.get_styles())
#
#     for each k:v tag.get_identifier() has got to give me 'container key'
#         but the key div doesn't know that it exists in the container
#
#     how'm i going to construct the style sheet anyway?
#     maybe it'll just make most sense to do it globally and separately.
#
#     let's just make this concrete.
#     it'll be something like
#
#     # construct html
#     # choose kv template
#     # generate random variations on kv template
#
#     html generation is something like:
#         doc = HtmlDoc()
#         doc.add_body(content_tags)
#         doc.add_styles(kv_template.get_styles())
#         doc.add_styles({'global selector': {'css_k': 'css_v'}})
#
#         all_html = str(doc)
#
#     the
#
#     '''
#     print(f'key_div: {key_div}')
#     print(f'value_div: {value_div}')
#     print(f'kv_div: {kv_div}')
#
#     # kv = KeyValue(
#     #     container_class='container_class',
#     #     key_contents='Key',
#     #     value_contents='Value<br>Value',
#     # )
