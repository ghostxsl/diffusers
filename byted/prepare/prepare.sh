#!/bin/bash

sudo apt-get update
sudo apt-get install -y libgl1 libglx0 mesa-utils
pip install -U pip
pip install -r byted/requirements.txt --user
pip install --user deepspeed

pip install -v -e .

python byted/prepare/prepare_accelerate.py
cat ~/.cache/huggingface/accelerate/default_config.yaml

echo "====== accelerate config done ======"
