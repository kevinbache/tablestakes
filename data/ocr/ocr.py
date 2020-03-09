"""Document representation with conversion from Google OCR format."""
import abc
import enum
import itertools
from typing import List, Optional

from tablestakes.fresh import utils


class BBox:
    def __init__(self, xmin: int, xmax: int, ymin: int, ymax: int):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def __repr__(self):
        return f'BBox({self.simple_repr()})'

    def simple_repr(self):
        return f'x=[{self.xmin}, {self.xmax}], y=[{self.ymin}, {self.ymax}]'

    @classmethod
    def from_dict(cls, d: dict):
        verts = d['vertices']
        xmin = verts[0]['x']
        xmax = verts[1]['x']
        ymin = verts[0]['y']
        ymax = verts[3]['y']
        return cls(xmin, xmax, ymin, ymax)


class Bounded:
    def __init__(self, bbox: BBox):
        self.bbox = bbox


DEBUG_MODE = False
def print_extra_properties(symbol_dict: dict, word_text: str):
    """debug function for looking at extra properties"""
    if 'property' in symbol_dict:
        import copy
        p = copy.copy(symbol_dict['property'])
        if 'detectedLanguages' in p:
            del p['detectedLanguages']
        if p:
            print(f'in word with text {word_text}, symbol {symbol_dict["text"]} got extra properties')
            utils.print_dict(p)


"""
in word with text Sasco, symbol o got extra properties
  detectedBreak: {'type': 'SPACE'}
in word with text 2400188ar, symbol r got extra properties
  detectedBreak: {'type': 'EOL_SURE_SPACE'}
in word with text 8.93, symbol 3 got extra properties
  detectedBreak: {'type': 'EOL_SURE_SPACE'}
in word with text 8.93, symbol 3 got extra properties
  detectedBreak: {'type': 'SPACE'}
"""


class Word(Bounded):
    class WordType(enum.Enum):
        TEXT = 1
        LINEBREAK = 2

    def __init__(self, text: str, bbox: BBox, word_type=WordType.TEXT):
        super().__init__(bbox)
        # if text == "D16352004":
        #     print("found it")
        self.text = text
        self.word_type = word_type

    def __repr__(self):
        return f'Word("{self.text}", {self.bbox.simple_repr()})'

    @staticmethod
    def _get_word_text(w: dict):
        text = ''.join([s['text'] for s in w['symbols']])
        if DEBUG_MODE:
            for s in w['symbols']:
                print_extra_properties(s, text)
        return text

    @classmethod
    def from_dict(cls, w: dict):
        text = cls._get_word_text(w)
        return cls(
            text=text,
            bbox=BBox.from_dict(w['boundingBox']),
        )


class HasWordsMixin(abc.ABC):
    @abc.abstractmethod
    def get_words(self) -> List[Word]:
        pass

    @staticmethod
    def _flatten_words(elements: List["HasWordsMixin"]):
        return list(itertools.chain.from_iterable([e.get_words() for e in elements]))


class Paragraph(Bounded, HasWordsMixin):
    def __init__(self, words: List[Word], bbox: BBox):
        super().__init__(bbox)
        self.words = words

    def __repr__(self):
        words = ' '.join([w.text for w in self.words if w.word_type == Word.WordType.TEXT])
        return f'Paragraph("{words}", {self.bbox.simple_repr()})'

    @staticmethod
    def _maybe_create_break_word(w: dict) -> Optional[Word]:
        s = w['symbols'][-1]
        if 'property' in s:
            p = s['property']
            if 'detectedBreak' in p:
                db = p['detectedBreak']
                if db['type'] == 'EOL_SURE_SPACE':
                    bbox = BBox.from_dict(w['boundingBox'])
                    bbox.xmin = bbox.xmax
                    word = Word(text='\n', bbox=bbox, word_type=Word.WordType.LINEBREAK)
                    return word
        return None

    @classmethod
    def from_dict(cls, d: dict):
        words = []
        for w in d['words']:
            word = Word.from_dict(w)
            words.append(word)
            break_word = cls._maybe_create_break_word(w)
            if break_word is not None:
                words.append(break_word)
        return cls(
            words=words,
            bbox=BBox.from_dict(d['boundingBox']),
        )

    def get_words(self) -> List[Word]:
        return self.words


class Block(Bounded, HasWordsMixin):
    def __init__(self, paragraphs: List[Paragraph], bbox: BBox, block_type: str):
        super().__init__(bbox),
        self.paragraphs = paragraphs
        self.block_type = block_type

    def __repr__(self):
        paragraphs = '\n  '.join([str(p) for p in self.paragraphs])
        return f'Block(\n  {paragraphs} \n  {self.bbox.simple_repr()}\n)'

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            paragraphs=[Paragraph.from_dict(p) for p in d['paragraphs']],
            bbox=BBox.from_dict(d['boundingBox']),
            block_type=d['blockType'],
        )

    def get_words(self) -> List[Word]:
        return self._flatten_words(self.paragraphs)


class Page(Bounded, HasWordsMixin):
    def __init__(self, blocks: List[Block], bbox: BBox):
        super().__init__(bbox),
        self.blocks = blocks

    def __repr__(self):
        blocks = '\n  '.join([str(b).replace('\n', '\n  ') for b in self.blocks])
        return f'Page(\n  {blocks} \n  {self.bbox.simple_repr()}\n)'

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            blocks=[Block.from_dict(b) for b in d['blocks']],
            bbox=BBox(xmin=0, xmax=d['width'], ymin=0, ymax=d['height']),
        )

    def get_words(self) -> List[Word]:
        return self._flatten_words(self.blocks)


class Document(HasWordsMixin):
    def __init__(self, pages: List[Page]):
        self.pages = pages

    def __repr__(self):
        pages = '\n  '.join([str(p).replace('\n', '\n  ') for p in self.pages])
        return f'Document(\n  {pages}\n)'

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            pages=[Page.from_dict(p) for p in d['fullTextAnnotation']['pages']],
        )

    def get_words(self) -> List[Word]:
        return self._flatten_words(self.pages)


if __name__ == '__main__':
    d = utils.read_json('sample_invoice_ocrd.json')

    # print("================ Paragraph ================")
    # pd = d['fullTextAnnotation']['pages'][0]['blocks'][0]['paragraphs'][0]
    # p = Paragraph.from_dict(pd)
    # print(p)
    #
    # print("================ Block ================")
    # bd = d['fullTextAnnotation']['pages'][0]['blocks'][0]
    # b = Block.from_dict(bd)
    # print(b)
    # print()

    print("================ Page ================")
    paged = d['fullTextAnnotation']['pages'][0]
    page = Page.from_dict(paged)
    print(page)
    print()

    # print("================ Document ================")
    # doc = Document.from_dict(d)
    # print(doc)
    # print()
