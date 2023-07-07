# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, splitext, basename, exists
import argparse
from tqdm import tqdm
from diffusers.data.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--img_dir",
        default=None,
        type=str,
        help="Directory to image")
    parser.add_argument(
        "--group_file",
        default=None,
        type=str,
        help="Path to group infos")
    parser.add_argument(
        "--out_dir",
        default="output",
        type=str,
        help="Directory to output.")
    parser.add_argument(
        "--save_name",
        default=None,
        type=str,
        help="File name to save.")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    return args


if __name__ == '__main__':
    args = parse_args()
    img_list = sorted(os.listdir(args.img_dir))

    # 保存每张模特图像文件的md5
    name2hash = {}
    for name in tqdm(img_list):
        name2hash[name] = get_file_md5(join(args.img_dir, name))
    pkl_save(name2hash, join(args.out_dir, args.save_name + '.name2hash'))

    # 得到模特图的group信息、hash值
    group_infos = load_file(args.group_file)
    group_hash = {}
    for k, v in tqdm(group_infos.items()):
        temp = []
        for url in v:
            name = basename(url)
            if exists(join(args.img_dir, name)):
                temp.append(name2hash[name])
        if temp:
            group_hash[k] = list(set(temp))

    print(len(group_infos), len(group_hash))
    print(sum([len(v) for v in group_infos.values()]), sum([len(v) for v in group_hash.values()]))

    # 合并hash值相同的group
    hash2group = {}
    for k, v in tqdm(group_hash.items()):
        for hash_ in v:
            if hash_ not in hash2group:
                hash2group[hash_] = [k]
            else:
                hash2group[hash_].append(k)
    rep_group = {k: v for k, v in hash2group.items() if len(v) > 1}
    print(f"repeat: {len(rep_group)}")

    unique_dict = {}
    done_list = set()
    for k, v in tqdm(group_hash.items()):
        if k in done_list:
            continue

        temp = []
        for name in v:
            temp.append(len(hash2group[name]))

        if sum(temp) == max(temp) * len(temp) and max(temp) == 1:
            done_list.add(k)
            if k not in unique_dict:
                unique_dict[k] = v
            else:
                unique_dict[k] += v

        elif sum(temp) == max(temp) * len(temp):
            for d_k in hash2group[name]:
                done_list.add(d_k)

            new_k = "_".join(sorted(hash2group[name]))
            new_k = get_str_md5(new_k)
            if new_k not in unique_dict:
                unique_dict[new_k] = v
            else:
                unique_dict[new_k] += v
        elif sum(temp) < max(temp) * len(temp):
            if k not in unique_dict:
                unique_dict[k] = v
            else:
                unique_dict[k] += v
        else:
            raise Exception('Error: f3!')

    pkl_save(unique_dict, join(args.out_dir, args.save_name + '.group2hash'))
    print(sum([len(v) for v in unique_dict.values()]))
    print('Done!')
