#!/usr/bin/bash

# Linux PyTorch build from Source
# https://github.com/pytorch/pytorch#from-source

conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -y -c pytorch magma-cuda102

cd ~
mkdir -p projects
cd projects

git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# python setup.py install
python setup.py bdist_wheel
