#!/bin/bash

## https://weasyprint.readthedocs.io/en/stable/install.html#macos
##brew install python3 cairo pango gdk-pixbuf libffi
#brew install pango gdk-pixbuf libffi
## https://anaconda.org/conda-forge/cairo
#conda install -c conda-forge cairo weasyprint
### https://pycairo.readthedocs.io/en/latest/
##pip install pycairo WeasyPrint

# http://docs.wand-py.org/en/0.4.1/guide/install.html #--with-liblqr # with-liblqr caused an error and it's just for liquid resize
pip install wand

pip install yattag
pip install faker

#git@github.com:tirthajyoti/pydbgen.git


brew install wkhtmltopdf
pip install pdfkit


# tesseract
# haven't installed opencv
#conda install -c conda-forge opencv
conda install pillow
pip install pytesseract

## imagemagick
## ref https://stackoverflow.com/questions/44624479/how-to-use-imagemagick-with-xquartz
#brew cask install xquartz
#brew tap tlk/imagemagick-x11
#brew install tlk/imagemagick-x11/imagemagick


## ref: https://github.com/pymupdf/PyMuPDF
#brew install mupdf-tools
#pip install PyMuPDF


conda install -c conda-forge poppler
pip install pdf2image

# selenium, firefox edition
conda install -c conda-forge geckodriver