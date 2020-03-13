import io
import re

from lxml import etree, html

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
print(h)
re_whitespace = re.compile(r'(\s+)')
WORD_TAG = 'w'


root = etree.fromstring(text=h, parser=etree.HTMLParser())

docwide_word_id = 0


def text_2_word_nodes(text_to_wrap: str):
    # re.spitting on whitespaces keeps the whitespace regions as entries in the resulting list
    word_strs = re.split(re_whitespace, text_to_wrap.strip())

    word_nodes = []
    for word_ind, word_str in enumerate(word_strs):
        if re.match(re_whitespace, word_str):
            # this word_str is whitespace.  it's going to be the tail of the previous node
            if not len(word_nodes):
                raise ValueError(f"What you doing starting this string with whitespace? Didn't you call strip()?"
                                 f"words: {word_strs}, e: {etree.tostring(node)}")
            word_nodes[-1].tail = word_str
        else:
            # don't worry about word_ids, we'll set those in a future iteration through the tree
            word_node = etree.Element(WORD_TAG)
            word_node.text = word_str
            word_nodes.append(word_node)

    return word_nodes


def handle_text(node: etree.Element, do_handle_tail_instead=False):
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

    word_nodes = text_2_word_nodes(text)

    # set the newly created word nodes as children of the parent node.
    # for text they go below the current node, at the beginning.
    # for tail, they get inserted into the current node's parent after the current node.
    for word_ind, word_node in enumerate(word_nodes):
        insert_node.insert(word_ind + insert_start, word_node)

    return


for node in root.iter():
    if node.tag == 'b':
        print('found it')
    if node.tail == 'blah_2':
        print('found it')
    if node.text == 'blah_2':
        print('found it')
    handle_text(node)
    handle_text(node, do_handle_tail_instead=True)

docwide_word_id = 0
for node in root.iter():
    if node.tag == WORD_TAG:
        node.attrib['word_id'] = f'{docwide_word_id:0>6d}'
        docwide_word_id += 1

out_str = etree.tostring(root, method='html', pretty_print=True, encoding='unicode')
document_root = html.fromstring(out_str)
print(etree.tostring(document_root, method='html', encoding='unicode', pretty_print=True))

# blah_2 is getting double wrapped

# import xml.dom.minidom
# dom = xml.dom.minidom.parse(io.StringIO(out_str)) # or xml.dom.minidom.parseString(xml_string)
# pretty_xml_as_string = dom.toprettyxml()
# print(pretty_xml_as_string)

# print(etree.tostring(etree.parse(io.StringIO(out_str), parser=etree.HTMLParser()).find('//div'), pretty_print=True, encoding='unicode'))
