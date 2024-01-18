#!/bin/bash
rm -rf /opt/conda/lib/python3.10/site-packages/distutils-precedence.pth
cp prepare/distutils-precedence.pth /opt/conda/lib/python3.10/site-packages/distutils-precedence.pth
mkdir ~/.pip
cp prepare/pip.conf ~/.pip/pip.conf

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r examples/vip/requirements.txt
pip install -v -e .
python /home/diffusers/prepare/prepare_accelerate.py

echo "accelerate config done"

sleep infinity
