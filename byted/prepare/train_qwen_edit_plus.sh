#!/bin/bash

sh ./byted/prepare/prepare.sh

export NCCL_HOSTID=${MY_POD_NAME}

nohup accelerate launch byted/training_scripts/train_qwen_edit_plus.py \
--pretrained_model_name_or_path=/mnt/bn/creative-algo/xsl/models/Qwen-Image-Edit-2509 \
--output_dir=/mnt/bn/creative-algo/xsl/xsl-all-pretrain-1128 \
--dataset_file=/mnt/bn/creative-algo/xsl/data/atoms_dataset/train_v4.json,/mnt/bn/creative-algo/xsl/data/atoms_dataset/train_v6.json,/mnt/bn/creative-algo/xsl/data/fei_0901_dataset/train.json,/mnt/bn/creative-algo/xsl/data/gemini_dataset/train.json,/mnt/bn/creative-algo/xsl/data/gemini_dataset/train_3_4.json,/mnt/bn/creative-algo/xsl/data/gpt_dataset/train.json \
--train_batch_size=1 --num_train_epochs=10 --gradient_checkpointing --checkpointing_steps=10000 \
--learning_rate=1e-5 --lr_warmup_steps=10000 --dataloader_num_workers=8 \
--allow_tf32 --mixed_precision=bf16 > log 2>&1 &

sleep infinity
