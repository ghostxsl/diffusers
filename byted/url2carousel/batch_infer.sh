#!/bin/bash

# 检查参数数量是否正确
if [ $# -ne 4 ]; then
    echo "使用方法：sh $0 <NPROC_PER_NODE> <python_script> <input_file> <output_file> "
    echo "示例：sh $0 8 /path/to/file /path/to/file /path/to/file"
    exit 1
fi

# 从命令行参数获取变量值
NPROC_PER_NODE=$1
python_script=$2
input_file=$3
output_file=$4
echo ${NPROC_PER_NODE}
echo ${python_script}
echo ${input_file}
echo ${output_file}

# 循环启动进程
i=0
while [ $i -lt ${NPROC_PER_NODE} ]
do
  nohup python ${python_script} --input_file=${input_file} --output_file=${output_file}_${i}.json \
  --num_ranks=${NPROC_PER_NODE} --rank=${i} > log_${i} 2>&1 &
  echo "已启动进程 ${i}: nohup python ${python_script}  --input_file=${input_file} --output_file=${output_file}_${i}.json --num_ranks=${NPROC_PER_NODE} --rank=${i} > log_${i} 2>&1 &"
  i=$((i + 1))
done

echo "所有进程启动完成, 共启动 ${NPROC_PER_NODE} 个进程"
