import re

from lxml import etree

from tablestakes import utils


class WordWrapper:
    re_whitespace = re.compile(r'(\s+)')
    WORD_TAG = 'w'

    @classmethod
    def text_2_word_nodes(cls, text_to_wrap: str):
        # re.spitting on whitespaces keeps the whitespace regions as entries in the resulting list
        word_strs = re.split(cls.re_whitespace, text_to_wrap.strip())

        word_nodes = []
        for word_ind, word_str in enumerate(word_strs):
            if re.match(cls.re_whitespace, word_str):
                # this word_str is whitespace.  it's going to be the tail of the previous node
                if not len(word_nodes):
                    raise ValueError(f"What you doing starting this string with whitespace? Didn't you call strip()?"
                                     f"words: {word_strs}, e: {etree.tostring(node)}")
                word_nodes[-1].tail = word_str
            else:
                # don't worry about word_ids, we'll set those in a future iteration through the tree
                word_node = etree.Element(cls.WORD_TAG)
                word_node.text = word_str
                word_nodes.append(word_node)

        return word_nodes

    @classmethod
    def handle_text(cls, node: etree._Element, do_handle_tail_instead=False):
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

        word_nodes = cls.text_2_word_nodes(text)

        # set the newly created word nodes as children of the parent node.
        # for text they go below the current node, at the beginning.
        # for tail, they get inserted into the current node's parent after the current node.
        for word_ind, word_node in enumerate(word_nodes):
            insert_node.insert(word_ind + insert_start, word_node)

        return

    @classmethod
    def wrap_words(cls, root: etree._Element, starting_wordid=0):
        # do it as a BFS.  that way you've already got the parent nodes in your queue and you won't ever double visit
        to_visit = [root]
        while to_visit:
            node = to_visit.pop(0)
            to_visit.extend(list(node))
            cls.handle_text(node)
            cls.handle_text(node, do_handle_tail_instead=True)

        docwide_word_id = starting_wordid
        for node in root.iter():
            if node.tag == cls.WORD_TAG:
                node.attrib['word_id'] = f'{docwide_word_id:0>6d}'
                docwide_word_id += 1


if __name__ == '__main__':
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

    WordWrapper.wrap_words(root)
    utils.hprint('after:')
    print(utils.root_2_pretty_str(root))

