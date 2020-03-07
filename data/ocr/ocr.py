from typing import Iterable

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


class Word(Bounded):
    def __init__(self, text: str, bbox: BBox):
        super().__init__(bbox)
        self.text = text

    def __repr__(self):
        return f'Word("{self.text}", {self.bbox.simple_repr()})'

    @staticmethod
    def _get_word_text(w: dict):
        return ''.join([s['text'] for s in w['symbols']])

    @classmethod
    def from_dict(cls, w: dict):
        return cls(
            text=cls._get_word_text(w),
            bbox=BBox.from_dict(w['boundingBox']),
        )


class Paragraph(Bounded):
    def __init__(self, words: Iterable[Word], bbox: BBox):
        super().__init__(bbox)
        self.words = words

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            words=[Word.from_dict(w) for w in d['words']],
            bbox=BBox.from_dict(d['boundingBox']),
        )

    def __repr__(self):
        words = ' '.join([w.text for w in self.words])
        return f'Paragraph("{words}", {self.bbox.simple_repr()})'


class Block(Bounded):
    def __init__(self, paragraphs: Iterable[Paragraph], bbox: BBox, block_type: str):
        super().__init__(bbox),
        self.paragraphs = paragraphs
        self.block_type = block_type

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            paragraphs=[Paragraph.from_dict(p) for p in d['paragraphs']],
            bbox=BBox.from_dict(d['boundingBox']),
            block_type=d['blockType'],
        )

    def __repr__(self):
        paragraphs = '\n  '.join([str(p) for p in self.paragraphs])
        return f'Block(\n  {paragraphs} \n  {self.bbox.simple_repr()}\n)'


class Page(Bounded):
    def __init__(self, blocks: Iterable[Block], bbox: BBox):
        super().__init__(bbox),
        self.blocks = blocks
        # self.block_type = block_type

    def __repr__(self):
        blocks = '\n  '.join([str(b).replace('\n', '\n  ') for b in self.blocks])
        return f'Page(\n  {blocks} \n  {self.bbox.simple_repr()}\n)'

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            blocks=[Block.from_dict(b) for b in d['blocks']],
            bbox=BBox(xmin=0, xmax=d['width'], ymin=0, ymax=d['height']),
        )


class Document:
    def __init__(self, pages: Iterable[Page]):
        self.pages = pages

    def __repr__(self):
        pages = '\n  '.join([str(p).replace('\n', '\n  ') for p in self.pages])
        return f'Document(\n  {pages}\n)'

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            pages=[Page.from_dict(p) for p in d['fullTextAnnotation']['pages']],
        )


if __name__ == '__main__':
    d = utils.read_json('sample_invoice_ocrd.json')

    print("================ Paragraph ================")
    pd = d['fullTextAnnotation']['pages'][0]['blocks'][0]['paragraphs'][0]
    # utils.print_dict(pd)
    p = Paragraph.from_dict(pd)
    print(p)

    print("================ Block ================")
    bd = d['fullTextAnnotation']['pages'][0]['blocks'][0]
    # utils.print_dict(bd)
    b = Block.from_dict(bd)
    print(b)
    print()

    print("================ Page ================")
    paged = d['fullTextAnnotation']['pages'][0]
    # utils.print_dict(paged)
    page = Page.from_dict(paged)
    print(page)
    print()

    print("================ Document ================")
    # utils.print_dict(dd)
    doc = Document.from_dict(d)
    print(doc)
    print()
