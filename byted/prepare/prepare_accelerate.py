# Copyright (c) wilson.xu. All rights reserved.
import os
import yaml
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare accelerate script.")
    parser.add_argument(
        "--config_file",
        default="deepspeed_config.yaml",
        type=str)
    parser.add_argument(
        "--cache_dir",
        default="/home/tiger",
        type=str)
    parser.add_argument(
        "--mixed_precision",
        default="bf16",
        type=str)

    parser.add_argument(
        "--zero_stage",
        default=1,
        type=int)
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int)
    parser.add_argument(
        "--gradient_clipping",
        default=1.0,
        type=float)

    args = parser.parse_args()

    cache_dir = os.path.join(args.cache_dir, ".cache/huggingface/accelerate")
    os.makedirs(cache_dir, exist_ok=True)
    args.cache_dir = cache_dir

    return args


if __name__ == "__main__":
    args = parse_args()

    # 0. Open config.yaml to modify
    root_dir = os.path.abspath(os.path.join(__file__, *(['..'] * 1)))
    with open(os.path.join(root_dir, args.config_file), encoding="utf-8") as f:
        config = yaml.safe_load(f.read())

    # 主节点IP地址
    main_process_ip = os.environ.get("ARNOLD_WORKER_0_HOST", None)
    # 主节点Port端口号
    main_process_port = int(os.environ.get("ARNOLD_WORKER_0_PORT", None))
    # 节点数量
    num_nodes = int(os.environ.get("ARNOLD_WORKER_NUM", 1))
    # 单节点上的GPU数量
    num_gpus = int(os.environ.get("ARNOLD_WORKER_GPU", 1))
    # 节点序号
    rank = int(os.environ.get("ARNOLD_ID", 0))

    config["main_process_ip"] = main_process_ip
    config["main_process_port"] = main_process_port
    config["num_machines"] = num_nodes
    config["num_processes"] = num_nodes * num_gpus
    config["machine_rank"] = rank
    config["mixed_precision"] = args.mixed_precision

    if "deepspeed" in args.config_file:
        config["deepspeed_config"]["zero_stage"] = args.zero_stage
        config["deepspeed_config"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        config["deepspeed_config"]["gradient_clipping"] = args.gradient_clipping

    with open(os.path.join(args.cache_dir, "default_config.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml.dump(config, allow_unicode=True))
