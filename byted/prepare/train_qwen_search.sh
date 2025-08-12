#!/bin/bash

sh ./byted/prepare/prepare.sh

export NCCL_HOSTID=${MY_POD_NAME}

nohup accelerate launch byted/training_scripts/train_qwen_poster_lora_latents.py \
--pretrained_model_name_or_path=/mnt/bn/creative-algo/xsl/models/Qwen-Image-Edit-2509 \
--transformer_model_path=/mnt/bn/creative-algo/xsl/xsl-subject-1017/checkpoint-30000 \
--output_dir=/mnt/bn/creative-algo/xsl/xsl-search-1023 \
--dataset_file=/mnt/bn/creative-algo/xsl/data/search_dataset/train.json \
--cond_size=1024x1024 \
--train_batch_size=1 --num_train_epochs=20 --gradient_checkpointing --checkpointing_steps=10000 \
--learning_rate=1e-4 --lr_warmup_steps=1000 --dataloader_num_workers=8 \
--lr_scheduler=piecewise_constant --lr_step_rules=1:80000,0.1:160000,0.01 \
--allow_tf32 --mixed_precision=bf16 --rank=128 --use_time_shift > log 2>&1 &

sleep infinity
