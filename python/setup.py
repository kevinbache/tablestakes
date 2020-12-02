import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="tablestakes",
    version='0.0.1',
    url="git@github.com:wonkygiraffe/tablestakes",
    license='MIT',

    author="My Name",
    author_email="wonkygiraffe@outlook.com",

    description="Brown, paper, tied up with string.",
    long_description=read("README.md"),

    packages=find_packages(exclude=('tests',)),

    install_requires=[
        'tqdm',
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'colorama',
        'keras',
        'google-cloud-storage',
        'yattag',
        'faker',
        'lxml',
        'nltk',
        'cssselect',
        'pdf2image',
        'pdfkit',
        'wkhtmltopdf',
        'pytesseract',
        'colorspacious',
        'cloudpickle>=1.6.0',
        'gputil',
        'psutil',
        'transformers',
        's3fs',
        'neptune-client',
        'neptune-contrib',
        'boto3',
        'dataclasses',
        # 'pytorch_memlab',
        'git+git://github.com/stonesjtu/pytorch_memlab',
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
    ],
)
