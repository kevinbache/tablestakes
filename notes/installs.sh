#!/bin/bash

# https://weasyprint.readthedocs.io/en/stable/install.html#macos
#brew install python3 cairo pango gdk-pixbuf libffi
brew install pango gdk-pixbuf libffi
# https://anaconda.org/conda-forge/cairo
conda install -c conda-forge cairo
# https://pycairo.readthedocs.io/en/latest/
pip install pycairo WeasyPrint

# http://docs.wand-py.org/en/0.4.1/guide/install.html
brew install imagemagick #--with-liblqr # with-liblqr caused an error and it's just for liquid resize
pip install wand

pip install yattag
pip install faker

#git@github.com:tirthajyoti/pydbgen.git
