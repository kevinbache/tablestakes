"""Document, Page, Word etc. objects."""
import copy

import abc
import enum
import itertools
from typing import List, Optional

import numpy as np

from tablestakes import utils


class BBox:
    def __init__(self, left: float, right: float, top: float, bottom: float):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def __repr__(self):
        return f'BBox({self.simple_repr()})'

    def simple_repr(self):
        return f'x=[{self.left:0.0f}, {self.right:0.0f}], y=[{self.top:0.0f}, {self.bottom:0.0f}]'

    @classmethod
    def from_dict(cls, d: dict):
        verts = d['vertices']
        xmin = verts[0]['x']
        xmax = verts[1]['x']
        ymin = verts[0]['y']
        ymax = verts[3]['y']
        return cls(xmin, xmax, ymin, ymax)

    def to_array(self):
        return np.array([self.left, self.right, self.top, self.bottom])

    @classmethod
    def from_array(cls, a: np.array):
        assert a.shape == (4,)
        return cls(*a)


class Bounded:
    def __init__(self, bbox: BBox):
        self.bbox = bbox


DEBUG_MODE = False


def print_extra_properties(symbol_dict: dict, word_text: str):
    """debug function for looking at extra properties"""
    if 'property' in symbol_dict:
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

    def __init__(self, text: str, bbox: BBox, word_type=WordType.TEXT, confidence=-1):
        super().__init__(bbox)
        self.text = text
        self.word_type = word_type
        self.confidence = confidence

    def __repr__(self):
        return f'Word("{self.text}", {self.bbox.simple_repr()})'


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

    def get_words(self) -> List[Word]:
        return self.words


class Block(Bounded, HasWordsMixin):
    def __init__(self, paragraphs: List[Paragraph], bbox: BBox, block_type: str):
        super().__init__(bbox),
        self.paragraphs = paragraphs
        if block_type != 'TEXT':
            raise ValueError(f"you've never seen a non-TEXT block-type before, but you just got: {block_type}")
        self.block_type = block_type

    def __repr__(self):
        paragraphs = '\n  '.join([str(p) for p in self.paragraphs])
        return f'Block(\n  {paragraphs} \n  {self.bbox.simple_repr()}\n)'

    def get_words(self) -> List[Word]:
        return self._flatten_words(self.paragraphs)


class Page(Bounded, HasWordsMixin):
    def __init__(self, blocks: List[Block], bbox: BBox):
        super().__init__(bbox),
        self.blocks = blocks

    def __repr__(self):
        blocks = '\n  '.join([str(b).replace('\n', '\n  ') for b in self.blocks])
        return f'Page(\n  {blocks} \n  {self.bbox.simple_repr()}\n)'

    def get_words(self) -> List[Word]:
        return self._flatten_words(self.blocks)


class Document(HasWordsMixin):
    def __init__(self, pages: List[Page]):
        self.pages = pages

    def __repr__(self):
        pages = '\n  '.join([str(p).replace('\n', '\n  ') for p in self.pages])
        return f'Document(\n  {pages}\n)'

    def get_words(self) -> List[Word]:
        return self._flatten_words(self.pages)



class GoogleOcrDocumentFactory:
    """Factory for creating Document structure from GoogleOcr json blob loaded as a dict.

    The input dictionary should have a key called "fullTextAnnotation".

    See https://cloud.google.com/vision/docs/ocr for more info.
    """
    @staticmethod
    def _get_word_text(word_dict: dict):
        text = ''.join([s['text'] for s in word_dict['symbols']])
        if DEBUG_MODE:
            for s in word_dict['symbols']:
                print_extra_properties(s, text)
        return text

    @classmethod
    def _word_dict_2_word(cls, w: dict):
        text = cls._get_word_text(w)
        return Word(
            text=text,
            bbox=BBox.from_dict(w['boundingBox']),
        )

    @staticmethod
    def _maybe_create_break_word(w: dict) -> Optional[Word]:
        s = w['symbols'][-1]
        if 'property' in s:
            p = s['property']
            if 'detectedBreak' in p:
                db = p['detectedBreak']
                if db['type'] == 'EOL_SURE_SPACE':
                    bbox = BBox.from_dict(w['boundingBox'])
                    bbox.left = bbox.right
                    word = Word(text='\n', bbox=bbox, word_type=Word.WordType.LINEBREAK)
                    return word
        return None

    @classmethod
    def _paragraph_dict_2_paragraph(cls, d: dict):
        words = []
        for w in d['words']:
            word = cls._word_dict_2_word(w)
            words.append(word)
            break_word = cls._maybe_create_break_word(w)
            if break_word is not None:
                words.append(break_word)
        return Paragraph(
            words=words,
            bbox=BBox.from_dict(d['boundingBox']),
        )

    @classmethod
    def _block_dict_2_block(cls, d: dict):
        return Block(
            paragraphs=[cls._paragraph_dict_2_paragraph(p) for p in d['paragraphs']],
            bbox=BBox.from_dict(d['boundingBox']),
            block_type=d['blockType'],
        )

    @classmethod
    def _page_dict_2_page(cls, d: dict):
        return Page(
            blocks=[cls._block_dict_2_block(b) for b in d['blocks']],
            bbox=BBox(left=0, right=d['width'], top=0, bottom=d['height']),
        )

    @classmethod
    def document_dict_2_document(cls, d: dict):
        return Document(
            pages=[cls._page_dict_2_page(p) for p in d['fullTextAnnotation']['pages']],
        )


if __name__ == '__main__':
    d = utils.read_json('../data/ocr/sample_invoice_google_ocr_output.json')

    # print("================ Paragraph ================")
    # pd = d['fullTextAnnotation']['pages'][0]['blocks'][0]['paragraphs'][0]
    # p = GoogleOcrDocumentFactory._paragraph_dict_2_paragraph(pd)
    # print(p)

    # print("================ Block ================")
    # bd = d['fullTextAnnotation']['pages'][0]['blocks'][0]
    # b = GoogleOcrDocumentFactory._block_dict_2_block(bd)
    # print(b)
    # print()

    # print("================ Page ================")
    # paged = d['fullTextAnnotation']['pages'][0]
    # page = GoogleOcrDocumentFactory._page_dict_2_page(paged)
    # print(page)
    # print()

    # print("================ Document ================")
    # doc = GoogleOcrDocumentFactory.document_dict_2_document(d)
    # print(doc)
    # print()
