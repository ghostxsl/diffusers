#!/bin/bash
pip install -U pip
pip uninstall torchaudio torchtext torchdata torchvision -y
pip install /apps/dat/cv/xsl/weights/torch-1.13.1+cu117-cp310-cp310-linux_x86_64.whl
pip install /apps/dat/cv/xsl/weights/torchvision-0.14.1+cu117-cp310-cp310-linux_x86_64.whl
pip install /apps/dat/cv/xsl/weights/Pillow-10.0.0.tar.gz
pip install /apps/dat/cv/xsl/weights/opencv_python-4.8.0.74-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install /apps/dat/cv/xsl/weights/transformers-4.31.0.tar.gz
pip install /apps/dat/cv/xsl/weights/datasets-2.14.0.tar.gz
pip install /apps/dat/cv/xsl/weights/accelerate-0.21.0.tar.gz
pip install /apps/dat/cv/xsl/weights/pandas-2.0.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install /apps/dat/cv/xsl/weights/pyarrow-12.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install /apps/dat/cv/xsl/weights/scipy-1.11.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
cd /home/diffusers/
pip install -v -e .
