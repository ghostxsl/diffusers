# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import splitext, join
from tqdm import tqdm
import argparse
from diffusers.data.utils import *
from diffusers.data.vos_client import VOSClient
from diffusers.utils.vip_utils import load_image


def parse_args():
    parser = argparse.ArgumentParser(description="llm-vos upload script.")
    parser.add_argument(
        "--data_file",
        default=None,
        type=str,
        help=" ")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="Directory to image.")
    parser.add_argument(
        "--save_dir",
        default=None,
        type=str,
        help="Directory to save.")
    parser.add_argument(
        "--data_type",
        default="img",
        choices=["img", "pkl", "other"],
        type=str,
        help="Upload data type.")

    parser.add_argument(
        "--rank",
        default=None,
        type=int)
    parser.add_argument(
        "--num_ranks",
        default=None,
        type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.data_file is not None:
        data_list = load_file(args.data_file)
    else:
        data_list = sorted(os.listdir(args.data_dir))

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        total_num = len(data_list)
        stride = int(total_num / args.num_ranks)

        start_idx = stride * args.rank
        end_idx = stride * (args.rank + 1) if args.rank + 1 < args.num_ranks else len(data_list)
        data_list = data_list[start_idx: end_idx]

    vos_client = VOSClient()

    damaged = []
    for name in tqdm(data_list):
        try:
            if args.data_type == "img":
                img = load_image(join(args.data_dir, name))
                vos_client.upload_vos_pil(
                    img,
                    join(args.save_dir, splitext(name)[0] + '.jpg'),
                    format='jpeg',
                    quality=90,
                )
            elif args.data_type == "pkl":
                pkl_obj = pkl_load(join(args.data_dir, name))
                vos_client.upload_vos_pkl(pkl_obj, join(args.save_dir, name))
            else:
                with open(join(args.data_dir, name), "rb") as f:
                    file_bytes = f.read()
                vos_client.upload_vos_bytes(file_bytes, join(args.save_dir, name))
        except:
            damaged.append(name)

    print('Done!')
    print(damaged)
