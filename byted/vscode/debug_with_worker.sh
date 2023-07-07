set -ex

CPU=10 # 设置 cpu
MEMORY=30 # 设置内存 单位未 g
GPU=1 # 设置 gpu 数量
GPU_TYPE='Tesla-V100-SXM2-32GB' # 设置 gpu 类型

MASTER_IP=$(hostname -i | awk '{print $1}')
MASTER_PORT=9000
if [ -n "$PORT0" ]; then
    MASTER_PORT=$PORT0
    MASTER_IP=$MY_HOST_IPV6
fi

cat <<EOF > /workspace/vscode/debug_ipv6.sh
#!/bin/bash
set -ex
ssh -fN -L 5678:localhost:5678 -p ${MASTER_PORT} root@${MASTER_IP}
python3 -m debugpy --connect localhost:5678 ${@:1}
EOF

chmod +x /workspace/vscode/debug_ipv6.sh

# chmod +x .vscode/debug_with_worker.sh

# mlx worker launch --cpu=${CPU} --gpu=${GPU} --memory=${MEMORY} --type=${GPU_TYPE} -- /workspace/vscode/debug_ipv6.sh ${file}

# 可以通过 CPU GPU MEMORY GPU_TYPE 等变量设置启动的资源大小
# 也可以直接修改 launch 指令，启动更复杂的 worker，比如下面的例子，启动了一个 arnold 集群的 worker，使用 A100-SXM-80GB 卡，进行 debug
# mlx worker launch --resourcetype=arnold --usergroup=cloudnative-debug --type=A100-SXM-80GB --gpu=1 --cluster=cloudnative-lq -- /workspace/vscode/debug_ipv6.sh ${file}


# 如何已经有 worker, 如何进行连接调试
# 要调试的 workerid 可以用 mlx worker list 获取
mlx worker login 740344 -- /workspace/vscode/debug_ipv6.sh ${file}
