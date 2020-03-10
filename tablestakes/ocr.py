import pdf2image
import pytesseract
import abc

from tablestakes import doc


class OcrProvider(abc.ABC):
    def __init__(self, dpi=400):
        self.dpi = dpi

    @classmethod
    def load_pdf_to_images(cls, pdf_filename: str, dpi: int):
        return pdf2image.convert_from_path(pdf_filename, dpi=dpi)

    @abc.abstractmethod
    def ocr(self, pdf_filename: str) -> doc.Document:
        pass


class TesseractOcrProvider(OcrProvider):
    def ocr(self, pdf_filename: str) -> doc.Document:
        pyt_data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DATAFRAME,
        )
        # import pandas as pd
        # df = pd.DataFrame()
        # df.to_pickle()
        pass


class TesseractDocumentFactory:
    pass


if __name__ == '__main__':
    fullfile = 'sample_weasy_output.pdf'
    resolution = 500

    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(fullfile, dpi=resolution)
    image = images[0]

    pyt_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)

    # import utils
    # with utils.Timer('data'):
    #     pyt_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
    #
    # with utils.Timer('boxes'):
    #     pyt_boxes = pytesseract.image_to_boxes(image, output_type=pytesseract.Output.DICT)

    # for image in images:
    #     ocr_data = pytesseract.image_to_boxes(image)
    #     ocr_data =
    #     print(ocr_data)
    #     print(image)

    print('done')


# tesseract box/line painting example
# https://nanonets.com/blog/ocr-with-tesseract/#installingtesseract

