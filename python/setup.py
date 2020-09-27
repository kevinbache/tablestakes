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

    # When you install this package, this console_scripts option ensures that another executable python script named
    # `tablestakes_docker_entrypoint` is added to your path which will call the function called
    # `main_funk` in the file imported in
    # `tablestakes.tablestakes_docker_entrypoint`
    entry_points={
        'console_scripts': [
            'tablestakes_docker_entrypoint=tablestakes.tablestakes_docker_entrypoint:main_funk',
        ],
    },

    install_requires=[
        'tqdm',
        'numpy',
        'scipy',
        'pandas',
        'keras',
        'google-cloud-storage',
        'yattag',
        'faker',
        'lxml',
        'cssselect',
        'pdf2image',
        'pdfkit',
        'wkhtmltopdf',
        'pytesseract',
        'colorspacious',
        'torch',
        'torchnlp',
        'pytorch-lightning',
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
