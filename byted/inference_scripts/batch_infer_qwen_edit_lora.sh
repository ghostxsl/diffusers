#!/bin/bash

# 检查参数数量是否正确
if [ $# -ne 2 ]; then
    echo "使用方法：sh $0 <NPROC_PER_NODE> <lora_model_path>"
    echo "示例：sh $0 8 /path/to/file"
    exit 1
fi

# 从命令行参数获取变量值
NPROC_PER_NODE=$1
lora_model_path=$2
echo $NPROC_PER_NODE
echo $lora_model_path

# 验证lora_model_path是否为空
if [ -z "$lora_model_path" ]; then
    echo "错误：第二个参数（lora_model_path）不能为空！"
    exit 1
fi

# 循环启动进程
i=0
while [ $i -lt ${NPROC_PER_NODE} ]
do
  CUDA_VISIBLE_DEVICES=${i} nohup python byted/inference_scripts/infer_qwen_edit_v3.py \
  --lora_model_path="${lora_model_path}" \
  --num_ranks=${NPROC_PER_NODE} --rank=${i} > log_${i} 2>&1 &
  echo "已启动进程，CUDA_VISIBLE_DEVICES=${i}，日志文件：log_${i}"
  i=$((i + 1))
done

echo "所有进程启动完成，共启动 ${NPROC_PER_NODE} 个进程"
