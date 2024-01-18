import os
import yaml

root_dir = os.path.abspath(os.path.join(__file__, *(['..'] * 1)))
with open(os.path.join(root_dir, "default_config.yaml"), encoding="utf-8") as f:
    config = yaml.safe_load(f.read())

# 节点数量
num_nodes = int(os.environ.get("NNODES", 1))
# GPU数量
num_gpus = int(os.environ.get("NPROC_PER_NODE", 8))
# 主节点IP地址
main_process_ip = os.environ.get("MASTER_ADDR", None)
# 节点序号
rank = int(os.environ.get("NODE_RANK", 0))
# DATA_OUTPUT_DIR
out_dir = os.environ.get("DATA_OUTPUT_DIR", 0)

config["machine_rank"] = rank
config["main_process_ip"] = main_process_ip
config["num_machines"] = num_nodes
config["num_processes"] = num_nodes * num_gpus

acc_dir = "/home/jovyan/.cache/huggingface/accelerate"
os.makedirs(acc_dir, exist_ok=True)
with open(os.path.join(acc_dir, "default_config.yaml"), "w", encoding="utf-8") as f:
    f.write(yaml.dump(config, allow_unicode=True))
