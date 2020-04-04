import re

from typing import Any, Callable

from lxml.cssselect import CSSSelector
from lxml import etree

from functools import partial


def _add_kv(w: etree._Element, k: Any, v: str):
    w.attrib[k] = v


def _modify_nodes_inplace(root: etree._Element, css_selector_str: str, fn: Callable):
    # root = etree.fromstring(text=doc_contents, parser=etree.HTMLParser())
    sel = CSSSelector(css_selector_str, translator='html')
    print('modify_nodes starting')
    for w in sel(root):
        fn(w)
        print('  ', w, w.attrib, w.text)
    print('modify_nodes done \n')


def set_words_boolean(root: etree._Element, true_word_css_selector: str, key_name: str):
    # note: you can't use the :not() css selector with compound predicates.  so instead:
    # first set all nodes to false
    _modify_nodes_inplace(root, f'w', partial(_add_kv, k=key_name, v="0"))
    # then modify some nodes to true
    _modify_nodes_inplace(root, true_word_css_selector, partial(_add_kv, k=key_name, v="1"))


class WordWrapper:
    re_whitespace = re.compile(r'(\s+)')
    WORD_TAG = 'w'

    def __init__(self):
        self._used_word_ids = {}

    @classmethod
    def str_2_word_nodes(cls, str_to_wrap: str):
        # re.spitting on whitespaces keeps the whitespace regions as entries in the resulting list
        word_strs = re.split(cls.re_whitespace, str_to_wrap.strip())

        word_nodes = []
        for word_ind, word_str in enumerate(word_strs):
            if re.match(cls.re_whitespace, word_str):
                # this word_str is whitespace.  it's going to be the tail of the previous node
                if not len(word_nodes):
                    raise ValueError(f"What you doing starting this string with whitespace? Didn't you call strip()?"
                                     f"words: {word_strs}.")
                word_nodes[-1].tail = word_str
            else:
                # don't worry about word_ids, we'll set those in a future iteration through the tree
                word_node = etree.Element(cls.WORD_TAG)
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

        word_nodes = cls.str_2_word_nodes(text)

        # set the newly created word nodes as children of the parent node.
        # for text they go below the current node, at the beginning.
        # for tail, they get inserted into the current node's parent after the current node.
        for word_ind, word_node in enumerate(word_nodes):
            insert_node.insert(word_ind + insert_start, word_node)

        return

    def wrap_words_in_str(self, root: str, starting_word_id=0):
        # TODO: this will autowrap to <html><body>CONTENTS</html></body> but you might not want the <html><body>
        root = etree.fromstring(text=root, parser=etree.HTMLParser())
        self.wrap_words_on_tree_inplace(root, starting_word_id)
        return etree.tostring(root, encoding='unicode')

    def wrap_words_on_tree_inplace(self, root: etree._Element, starting_word_id=0):
        # do it as a BFS.  that way you've already got the parent nodes in your queue and you won't ever double visit
        to_visit = [root]
        while to_visit:
            node = to_visit.pop(0)
            to_visit.extend(list(node))
            self._handle_text(node)
            self._handle_text(node, do_handle_tail_instead=True)

        docwide_word_id = starting_word_id

        for node in root.iter():
            if node.tag == self.WORD_TAG:
                # TODO: factor out word_id definition
                word_id = f'word_{docwide_word_id:0>6d}'
                node.attrib['id'] = word_id
                self._used_word_ids[word_id] = node
                docwide_word_id += 1

    def get_used_id_to_word(self):
        return self._used_word_ids


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

