# import pytesseract
#
# from wand.image import Image as WandImage
# from PIL import Image
#
# def load_pdf(fullfile: str, resolution=300):
#     with WandImage(filename=fullfile, resolution=resolution) as img:
#         return img
#
# def load_pdf(fullfile: str, resolution=300):
#     files = []
#     with(WandImage(filename=fullfile, resolution=resolution)) as conn:
#         for index, image in enumerate(conn.sequence):
#
#             image_name = os.path.splitext(pdf_file)[0] + str(index + 1) + '.png'
#             # Image(image).save(filename = image_name)
#             files.append(image_name)


if __name__ == '__main__':
    fullfile = 'sample_weasy_output.pdf'
    resolution = 300

    # with(WandImage(filename=fullfile, resolution=resolution)) as conn:
    #     for index, image in enumerate(conn.sequence):
    #         print(index, image)
    #
    # # pdf = load_pdf('sample_weasy_output.pdf')
    # # print(pdf)

    # ##############
    # # PYMUPDF
    # import fitz
    #
    # doc = fitz.open("sample_weasy_output.pdf")
    #
    #
    # for i in range(len(doc)):
    #     for img in doc.getPageImageList(i):
    #         xref = img[0]
    #         pix = fitz.Pixmap(doc, xref)
    #         if pix.n < 5:  # this is GRAY or RGB
    #             pix.writePNG("p%s-%s.png" % (i, xref))
    #         else:  # CMYK: convert to RGB first
    #             pix1 = fitz.Pixmap(fitz.csRGB, pix)
    #             pix1.writePNG("p%s-%s.png" % (i, xref))
    #             pix1 = None

    from pdf2image import convert_from_path
    import pytesseract
    import time

    images = convert_from_path(fullfile, dpi=299.999)
    image = images[0]


    class Timer:
        def __init__(self, name: str):
            self.name = name

        def __enter__(self):
            self.t = time.time()
            print(f"Starting timer {self.name} at time {self.t}.", end=" ")

        def __exit__(self, *args):
            print(f"Took {time.time() - self.t}")


    with Timer('data'):
        pyt_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)

    with Timer('boxes'):
        pyt_boxes = pytesseract.image_to_boxes(image, output_type=pytesseract.Output.DICT)

    # for image in images:
    #     ocr_data = pytesseract.image_to_boxes(image)
    #     ocr_data =
    #     print(ocr_data)
    #     print(image)

    print('done')


# tesseract box/line painting example
# https://nanonets.com/blog/ocr-with-tesseract/#installingtesseract

