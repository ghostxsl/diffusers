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
        choices=["img", "pickle", "other"],
        type=str,
        help="Upload data type.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    data_list = load_file(args.data_file)
    vos_client = VOSClient()

    damaged = []
    for name in tqdm(data_list):
        try:
            if args.data_type == "img":
                img = load_image(join(args.data_dir, name))
                vos_client.upload_vos_pil(
                    img,
                    join(args.save_dir, splitext(name)[0] + '.jpg'),
                    format='jpeg'
                )
            elif args.data_type == "pickle":
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
