#!/bin/bash
mkdir ~/.pip
echo "[global]" >> ~/.pip/pip.conf
echo "index-url = http://pypi.vip.vip.com/simple/" >> ~/.pip/pip.conf
echo "[install]" >> ~/.pip/pip.conf
echo "trusted-host = pypi.vip.vip.com" >> ~/.pip/pip.conf

conda create --name xsl python=3.8 -y
conda activate xsl
pip install torch==2.0.1 torchvision==0.15.2
cd /home/diffusers
pip install -r examples/vip/requirements.txt
