# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Batch run script.")
    parser.add_argument(
        "--script",
        default="vip_tools/human_filter.py",
        type=str,
        help="path to run script")

    parser.add_argument(
        "--gpus",
        default=None,
        type=str,
        help="List of GPUs.")
    parser.add_argument(
        "--nproc_per_node",
        default=8,
        type=int,
        help="Number of GPUs.")
    parser.add_argument(
        "--node_rank",
        default=0,
        type=int,
        help="Worker's order number.")
    parser.add_argument(
        "--nnodes",
        default=1,
        type=int,
        help="Number of workers.")

    parser.add_argument(
        "--img_dir",
        default=None,
        type=str,
        help="Directory to image.")
    parser.add_argument(
        "--json_file",
        default=None,
        type=str,
        help="File to json.")
    parser.add_argument(
        "--out_dir",
        default="output",
        type=str,
        help="Directory to save.")
    parser.add_argument(
        "--weight_dir",
        default="weights",
        type=str,
        help="Directory to weights.")

    args = parser.parse_args()
    if args.gpus is not None:
        args.gpus = args.gpus.split(",")
        args.nproc_per_node = len(args.gpus)

    args.total_ranks = args.nproc_per_node * args.nnodes

    return args


if __name__ == '__main__':
    args = parse_args()

    for i in range(args.nproc_per_node):
        ind = i if args.gpus is None else args.gpus[i]
        run_code = f"CUDA_VISIBLE_DEVICES={ind} nohup python {args.script} "
        if args.img_dir is not None:
            run_code += f"--img_dir={args.img_dir} "
        elif args.json_file is not None:
            run_code += f"--json_file={args.json_file} "
        else:
            raise Exception("error input `img_dir` or `json_file`")

        run_code += f"--out_dir={args.out_dir} --weight_dir={args.weight_dir} "
        run_code += f"--rank={args.node_rank * args.nproc_per_node + i} --num_ranks={args.total_ranks} "
        run_code += f"> log{ind}.txt 2>&1 &"

        os.system(run_code)
        print(run_code)
