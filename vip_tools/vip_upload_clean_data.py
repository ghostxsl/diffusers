# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import splitext, join, basename
from tqdm import tqdm
import argparse
from diffusers.data.utils import *
from diffusers.data.vos_client import VOSClient
from diffusers.utils.vip_utils import load_image


def parse_args():
    parser = argparse.ArgumentParser(description="llm-vos upload script.")
    parser.add_argument(
        "--img_dir",
        default=None,
        type=str,
        help="Directory to image.")
    parser.add_argument(
        "--pose_dir",
        default=None,
        type=str,
        help="Directory to pose.")

    parser.add_argument(
        "--name2hash",
        default=None,
        type=str,
        help=" ")
    parser.add_argument(
        "--group2hash",
        default=None,
        type=str,
        help=" ")

    parser.add_argument(
        "--save_img",
        default=None,
        type=str,
        help="Directory to save.")
    parser.add_argument(
        "--save_pose",
        default=None,
        type=str,
        help="Directory to save.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    img_list = sorted(os.listdir(args.img_dir))
    pose_list = sorted(os.listdir(args.pose_dir))
    print(len(img_list), len(pose_list))

    group2hash = load_file(args.group2hash)
    name2hash = load_file(args.name2hash)

    hash2name = {}
    for name in tqdm(pose_list):
        if splitext(name)[0] in name2hash:
            hash_ = name2hash[splitext(name)[0]]
            hash2name[hash_] = splitext(name)[0]
    print(len(pose_list), len(hash2name))

    vos_client = VOSClient()

    damaged = []
    out_dict = {}
    for k, v in tqdm(group2hash.items()):
        temp = []
        for hash_ in v:
            if hash_ in hash2name:
                temp.append(hash_)
                try:
                    name = hash2name[hash_]
                    img = load_image(join(args.img_dir, name))
                    vos_client.upload_vos_pil(
                        img,
                        join(args.save_img, hash_ + '.jpg'),
                        format='jpeg',
                        quality=90,
                    )

                    pose = pkl_load(join(args.pose_dir, name + '.pose'))
                    vos_client.upload_vos_pkl(pose, join(args.save_pose, hash_ + '.pose'))
                except:
                    damaged.append(hash_)
        if temp:
            out_dict[k] = list(set(temp))

    pkl_save(out_dict, f"{splitext(args.group2hash)[0]}_filtered.group2hash")
    print('Done!')
    print(damaged)
