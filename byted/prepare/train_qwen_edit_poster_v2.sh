#!/bin/bash

sh ./byted/prepare/prepare.sh

export NCCL_HOSTID=${MY_POD_NAME}

# python -m accelerate.commands.launch
# --lora_model_path=/mnt/bn/creative-algo/xsl-lora/checkpoint-9000/pytorch_lora_weights.safetensors \

nohup accelerate launch byted/training_scripts/train_qwen_poster_lora_latents.py \
--pretrained_model_name_or_path=/mnt/bn/creative-algo/xsl/models/Qwen-Image-Edit \
--output_dir=/mnt/bn/creative-algo/xsl/xsl-poster-0923-new \
--dataset_file=/mnt/bn/creative-algo/xsl/data/gpt_dataset/train.json,/mnt/bn/creative-algo/xsl/data/fei_0901_dataset/train.json,/mnt/bn/creative-algo/xsl/data/gemini_dataset/train.json \
--train_batch_size=1 --num_train_epochs=20 --gradient_checkpointing --checkpointing_steps=10000 \
--learning_rate=3e-5 --lr_warmup_steps=1000 --dataloader_num_workers=8 \
--allow_tf32 --mixed_precision=bf16 --rank=128 --use_time_shift > log 2>&1 &

sleep infinity
