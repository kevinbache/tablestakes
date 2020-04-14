import abc
import html

import copy
from functools import partial
import re
from typing import Any, Callable, List, Union, Optional

import pandas as pd
from lxml.cssselect import CSSSelector
from lxml import etree

from tablestakes import html_css as hc, utils


class EtreeModifier(abc.ABC):
    def __call__(self, root_or_doc: Union[etree._Element, hc.Document]) -> Optional[hc.Document]:
        """If called on a document, wrap yourself in a Stack which does etree wrapping / unwrapping.

        This lets you write
            doc = etree_modifiers.WordColorDocCssAdder(doc)(doc)
        which looks funny but works.
        """
        if isinstance(root_or_doc, hc.Document):
            stack = EtreeModifierStack([self])
            return stack(root_or_doc)
        else:
            self._call_inner(root_or_doc)

    @abc.abstractmethod
    def _call_inner(self, root: etree._Element):
        pass

    @staticmethod
    def _modify_nodes_inplace(root: etree._Element, css_selector_str: str, fn: Callable):
        sel = CSSSelector(css_selector_str, translator='html')
        for w in sel(root):
            fn(w)

    @staticmethod
    def _add_kv(w: etree._Element, k: Any, v: str):
        w.attrib[k] = v


class EtreeModifierStack:
    def __init__(self, modifiers: List[EtreeModifier], do_use_timers=True):
        self.modifiers = modifiers
        self.do_use_timers = do_use_timers

    def __call__(self, doc: hc.Document) -> hc.Document:
        doc_root = doc.to_etree()
        for modifier in self.modifiers:
            if self.do_use_timers:
                with utils.Timer(f'Doc modifier {modifier.__class__.__name__}'):
                    modifier(doc_root)
            else:
                modifier(doc_root)
        doc.replace_contents_with_etree(doc_root)
        return doc


class WordWrapper(EtreeModifier):
    """Wraps each word in the bare textual content of the document in a <w> tag, whitespace in <wsp> tag."""
    re_whitespace = re.compile(r'(\s+)')
    WORD_TAG = 'w'
    WHITESPACE_TAG = 'wsp'

    PARENT_CLASS_ATTRIB_NAME = 'parent_class'

    # leave as 'id' so that id selectors work for the css selector that's used to set colors on each word box
    WORD_ID_ATTRIB_NAME = 'id'

    def __init__(self, starting_word_id=0):
        self._used_word_ids = {}
        self._starting_word_id = starting_word_id

    @classmethod
    def _str_2_word_nodes(cls, str_to_wrap: str):
        # re.spitting on whitespaces keeps the whitespace regions as entries in the resulting list
        word_strs = re.split(cls.re_whitespace, str_to_wrap.strip())

        # don't worry about word_ids, we'll set those in a future iteration through the tree
        word_nodes = []
        for word_ind, word_str in enumerate(word_strs):
            tag = cls.WHITESPACE_TAG if re.match(cls.re_whitespace, word_str) else cls.WORD_TAG
            word_node = etree.Element(tag)
            word_node.text = word_str
            word_nodes.append(word_node)

        return word_nodes

    @classmethod
    def _handle_text(cls, node: etree._Element, do_handle_tail_instead=False):
        if do_handle_tail_instead:
            if not node.tail or not node.tail.strip():
                return
            text = node.tail
            node.tail = ''
            insert_node = node.getparent()
            insert_start = insert_node.index(node) + 1
        else:
            if not node.text or not node.text.strip():
                return
            text = node.text.strip()
            node.text = ''
            insert_node = node
            insert_start = 0

        word_nodes = cls._str_2_word_nodes(text)

        # the child nodes all get the classes of the parents.  that's used later in postproc
        for word_node in word_nodes:
            word_node.attrib[cls.PARENT_CLASS_ATTRIB_NAME] = insert_node.attrib.get('class', '')

        # set the newly created word nodes as children of the parent node.
        # for text they go below the current node, at the beginning.
        # for tail, they get inserted into the current node's parent after the current node.
        for word_ind, word_node in enumerate(word_nodes):
            insert_node.insert(word_ind + insert_start, word_node)

        return

    def _call_inner(self, root: etree._Element):
        # do it as a BFS rather than using etree._Element.iter().
        # using iter, you add nodes to the tree as you go and they get double visited.
        # with the BFS, you've already got the parent nodes in your queue when you visit the child
        # and you won't ever double visit
        to_visit = [root]
        while to_visit:
            node = to_visit.pop(0)
            to_visit.extend(list(node))
            self._handle_text(node)
            self._handle_text(node, do_handle_tail_instead=True)

        docwide_word_id = self._starting_word_id

        for node in root.iter():
            if node.tag == self.WORD_TAG:
                # TODO: factor out word_id definition
                word_id = f'word_{docwide_word_id:0>6d}'
                node.attrib[self.WORD_ID_ATTRIB_NAME] = word_id
                self._used_word_ids[word_id] = node
                docwide_word_id += 1

    def get_used_id_to_word_dict(self):
        return self._used_word_ids


class SetBooleanAttribOnWordsModifier(EtreeModifier):
    def __init__(self, true_word_css_selector: str, key_name: str):
        self.true_word_css_selector = true_word_css_selector
        self.key_name = key_name

    def _call_inner(self, root: etree._Element):
        self._modify_nodes_inplace(root, WordWrapper.WORD_TAG, partial(self._add_kv, k=self.key_name, v="0"))
        # then modify some nodes to true
        self._modify_nodes_inplace(root, self.true_word_css_selector, partial(self._add_kv, k=self.key_name, v="1"))


class SetIsKeyOnWordsModifier(SetBooleanAttribOnWordsModifier):
    KEY_NAME = 'isKey'
    TRUE_WORD_CSS_SELECTOR = f'.{hc.SelectorType.KEY.html_class_name} w'

    def __init__(self):
        super().__init__(
            true_word_css_selector=self.TRUE_WORD_CSS_SELECTOR,
            key_name=self.KEY_NAME,
        )


class SetIsValueOnWordsModifier(SetBooleanAttribOnWordsModifier):
    KEY_NAME = 'isValue'
    TRUE_WORD_CSS_SELECTOR = f'.{hc.SelectorType.VALUE.html_class_name} {WordWrapper.WORD_TAG}'

    def __init__(self):
        super().__init__(
            true_word_css_selector=self.TRUE_WORD_CSS_SELECTOR,
            key_name=self.KEY_NAME,
        )


class ConvertParentClassNamesToWordAttribsModifier(EtreeModifier):
    """Convert each word's parent_class attribute into tags.

    The WordWrapper gives each word a <w> tag with the attribute parent_class=... whatever the parent
    element's classes are.

    This Modifier converts those classes into many-hot attribute labels.

    This is used to mark specific kv names on each word while ignoring selector type classes.

    So if the document has kvs named "from_address", "to_address", and "date_received", this will convert:
        <w parent_class="kv_key from_address">From</w>
            to
        <w kv=from_address="1", kv=to_address="0", kv=date_received="0">From</w>
    """
    PARENT_CLASSES_TO_IGNORE = [st.html_class_name for st in hc.SelectorType if st.html_class_name]

    # prefix assigned to parent class names when they're set on each word.
    # currenttly, this class is being used to propagate the id of the kv that this word came from so name it for that.
    TAG_PREFIX = 'kv_is_'

    def __init__(self):
        self._seen_parent_classes = []

    def _call_inner(self, root: etree._Element):
        # first make a list of all parent classes in the document
        self._modify_nodes_inplace(
            root=root,
            css_selector_str=WordWrapper.WORD_TAG,
            fn=self._note_parent_classes
        )

        # then add a many-hot vector for this word's parent classes
        self._modify_nodes_inplace(
            root=root,
            css_selector_str=WordWrapper.WORD_TAG,
            fn=self._convert_parent_classes_to_tags
        )

    @classmethod
    def _get_parent_classes(cls, word: etree._Element):
        parent_classes_this_node = word.attrib.get(WordWrapper.PARENT_CLASS_ATTRIB_NAME, '')
        return [c for c in parent_classes_this_node.split() if c not in cls.PARENT_CLASSES_TO_IGNORE]

    def _note_parent_classes(self, word: etree._Element):
        for parent_class in self._get_parent_classes(word):
            if parent_class not in self._seen_parent_classes:
                self._seen_parent_classes.append(parent_class)

    @classmethod
    def _set_word_attrib(cls, word: etree._Element, key: str, value: str):
        word.attrib[f'{cls.TAG_PREFIX}{key}'] = value

    def _convert_parent_classes_to_tags(self, word: etree._Element):
        for seen_parent_class in self._seen_parent_classes:
            self._set_word_attrib(word, seen_parent_class, '0')

        for current_parent_class in self._get_parent_classes(word):
            self._set_word_attrib(word, current_parent_class, '1')

        del word.attrib[WordWrapper.PARENT_CLASS_ATTRIB_NAME]


class CopyWordTextToAttribModifier(EtreeModifier):
    TEXT_ATTRIB_NAME = 'text'

    def _call_inner(self, root: etree._Element):
        self._modify_nodes_inplace(
            root=root,
            css_selector_str=WordWrapper.WORD_TAG,
            fn=self._copy_text_to_attrib,
        )

    @classmethod
    def _copy_text_to_attrib(cls, word: etree._Element):
        word.attrib[cls.TEXT_ATTRIB_NAME] = word.text


class SaveWordAttribsToDataFrame(EtreeModifier):
    def __init__(self):
        self._attrib_dicts = []

    def _call_inner(self, root: etree._Element):
        self._attrib_dicts = []
        self._modify_nodes_inplace(
            root=root,
            css_selector_str=WordWrapper.WORD_TAG,
            fn=self._save_node_attribs
        )

    def _save_node_attribs(self, word: etree._Element):
        self._attrib_dicts.append(copy.copy(word.attrib))

    def get_df(self):
        df = pd.DataFrame(self._attrib_dicts)
        df.apply(pd.to_numeric, errors='ignore')
        return df


class WordColorizer(EtreeModifier):
    R_ATTRIB_NAME = 'r'
    G_ATTRIB_NAME = 'g'
    B_ATTRIB_NAME = 'b'

    RGB = [R_ATTRIB_NAME, G_ATTRIB_NAME, B_ATTRIB_NAME]

    def __init__(self):
        self._word_count = 0
        self.colors = []

    def _call_inner(self, root: etree._Element):
        self._word_count = 0
        self._modify_nodes_inplace(
            root=root,
            css_selector_str=WordWrapper.WORD_TAG,
            fn=self._count_words,
        )

        if not self._word_count:
            return

        self.color_array = utils.generate_unique_color_matrix(num_colors=self._word_count)
        self._current_color_ind = 0
        self._modify_nodes_inplace(
            root=root,
            css_selector_str=WordWrapper.WORD_TAG,
            fn=self._add_colors_to_nodes,
        )

    def _count_words(self, word: etree._Element):
        self._word_count += 1

    def _add_colors_to_nodes(self, word: etree._Element):
        color = self.color_array[self._current_color_ind]

        word.attrib[self.R_ATTRIB_NAME] = str(color[0])
        word.attrib[self.G_ATTRIB_NAME] = str(color[1])
        word.attrib[self.B_ATTRIB_NAME] = str(color[2])

        self._current_color_ind += 1


class WordColorDocCssAdder(EtreeModifier):
    def __init__(self, doc: hc.Document):
        self.doc = doc

    def _call_inner(self, root: etree._Element):
        self._modify_nodes_inplace(
            root=root,
            css_selector_str=WordWrapper.WORD_TAG,
            fn=self._add_css_to_doc,
        )

    @staticmethod
    def get_color_for_word(word: etree._Element):
        """This assumes you've already set the color attributes on this word with ."""
        r = word.attrib[WordColorizer.R_ATTRIB_NAME]
        g = word.attrib[WordColorizer.G_ATTRIB_NAME]
        b = word.attrib[WordColorizer.B_ATTRIB_NAME]
        return f'rgb({r}, {g}, {b})'

    def _add_css_to_doc(self, word: etree._Element):
        self.doc.add_style(
            self._get_css_chunk(
                word_id=word.attrib[WordWrapper.WORD_ID_ATTRIB_NAME],
                color_str=self.get_color_for_word(word)
            )
        )

    @staticmethod
    def _get_css_chunk(word_id: str, color_str: str):
        return hc.CssChunk(
            selector=f'#{word_id}',
            values={
                'background-color': color_str,
                'color': color_str,
            },
        )
