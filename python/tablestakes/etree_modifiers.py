import abc
import copy
from functools import partial
import re
from typing import Any, Callable, List

import pandas as pd
from lxml.cssselect import CSSSelector
from lxml import etree

from tablestakes import html_css as hc


class EtreeModifier(abc.ABC):
    @abc.abstractmethod
    def __call__(self, root: etree._Element):
        pass

    @staticmethod
    def _modify_nodes_inplace(root: etree._Element, css_selector_str: str, fn: Callable):
        sel = CSSSelector(css_selector_str, translator='html')
        for w in sel(root):
            fn(w)

    @staticmethod
    def _add_kv(w: etree._Element, k: Any, v: str):
        w.attrib[k] = v


class WordWrapper(EtreeModifier):
    """Wraps each word in the bare textual content of the document in a <w> tag, whitespace in <wsp> tag."""
    re_whitespace = re.compile(r'(\s+)')
    WORD_TAG = 'w'
    WHITESPACE_TAG = 'wsp'

    PARENT_CLASS_ATTRIB_NAME = 'parent_class'

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

    # def wrap_words_in_str(self, root: str, starting_word_id=0):
    #     # TODO: this will autowrap to <html><body>CONTENTS</html></body> but you might not want the <html><body>
    #     root = etree.fromstring(text=root, parser=etree.HTMLParser())
    #     self.wrap_words_on_tree_inplace(root, starting_word_id)
    #     return etree.tostring(root, encoding='unicode')

    def __call__(self, root: etree._Element):
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
                node.attrib['id'] = word_id
                self._used_word_ids[word_id] = node
                docwide_word_id += 1

    def get_used_id_to_word_dict(self):
        return self._used_word_ids


class SetBooleanAttribOnWordsModifier(EtreeModifier):
    def __init__(self, true_word_css_selector: str, key_name: str):
        self.true_word_css_selector = true_word_css_selector
        self.key_name = key_name

    def __call__(self, root: etree._Element):
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
    TRUE_WORD_CSS_SELECTOR = f'.{hc.SelectorType.VALUE.html_class_name} w'

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
        <w from_address="1", to_address="0", date_received="0">From</w>
    """
    PARENT_CLASSES_TO_IGNORE = [st.html_class_name for st in hc.SelectorType if st.html_class_name]

    def __init__(self):
        self._seen_parent_classes = []

    def __call__(self, root: etree._Element):
        # first make a list of all parent classes in the document
        self._modify_nodes_inplace(
            root=root,
            css_selector_str='w',
            fn=self._note_parent_classes
        )

        # then add a many-hot vector for this word's parent classes
        self._modify_nodes_inplace(
            root=root,
            css_selector_str='w',
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

    def _convert_parent_classes_to_tags(self, word: etree._Element):
        for seen_parent_class in self._seen_parent_classes:
            word.attrib[seen_parent_class] = '0'

        for current_parent_class in self._get_parent_classes(word):
            word.attrib[current_parent_class] = '1'

        del word.attrib[WordWrapper.PARENT_CLASS_ATTRIB_NAME]


class CopyWordTextToAttribModifier(EtreeModifier):
    TEXT_ATTRIB_NAME = 'word_text'

    def __call__(self, root: etree._Element):
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

    def __call__(self, root: etree._Element):
        self._modify_nodes_inplace(
            root=root,
            css_selector_str='w',
            fn=self._save_node_attribs
        )

    def _save_node_attribs(self, word: etree._Element):
        self._attrib_dicts.append(copy.copy(word.attrib))

    def get_df(self):
        return pd.DataFrame(self._attrib_dicts)


class EtreeModifierStack(EtreeModifier):
    def __init__(self, modifiers: List[EtreeModifier]):
        self.modifiers = modifiers

    def __call__(self, doc: hc.Document) -> hc.Document:
        doc_root = doc.to_etree()
        for modifier in self.modifiers:
            modifier(doc_root)
        doc.replace_contents_with_etree(doc_root)
        return doc


if __name__ == '__main__':
    from tablestakes import utils
    h = '''
    <div class='container'>
        blah_1
        <a>
            <aaa>
            </aaa>
            blah_a
            <aa>
                blah_aa1 blah_aa2
            </aa>
        </a>
        <b>
            blah_b1 blah_b2 <br>
            blah_b3  blah_b4
        </b>
        blah_2 blah_3 
    </div>
    '''
    root = etree.fromstring(text=h, parser=etree.HTMLParser()).find('.//div')
    utils.hprint('before:')
    print(utils.root_2_pretty_str(root))

    WordWrapper.wrap_words_in_str(root=root)
    utils.hprint('after:')
    print(utils.root_2_pretty_str(root))

