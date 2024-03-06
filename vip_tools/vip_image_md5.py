# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, splitext, split, basename, exists
from tqdm import tqdm
from diffusers.data.utils import *


if __name__ == '__main__':
    img_dir = "/apps/dat/cvtmp/xsl/upper_0311_split2"
    group_file = "/home/llms/upper_group_split2.pkl"
    save_root = "/home/llms/upper"
    out_name2hash_file = join(save_root, "upper2.name2hash")
    output_file = join(save_root, "upper2_group_hash.pkl")

    img_list = sorted(os.listdir(img_dir))

    # 保存每张有模特图像的md5
    name2hash = {}
    for name in tqdm(img_list):
        name2hash[name] = get_file_md5(join(img_dir, name))
    pkl_save(name2hash, out_name2hash_file)

    # 得到有模特的group信息
    group_infos = load_file(group_file)
    new_group_infos = {}
    for k, v in tqdm(group_infos.items()):
        temp = []
        for url in v:
            name = basename(url)
            if exists(join(img_dir, name)):
                temp.append(name)
        if temp:
            new_group_infos[k] = list(set(temp))

    group_hash = {}
    for k, v in tqdm(new_group_infos.items()):
        group_hash[k] = list(set([name2hash[name] for name in v]))
    print(len(group_infos), len(group_hash))
    print(sum([len(v) for v in group_infos.values()]), sum([len(v) for v in group_hash.values()]))

    # 合并hash值相同的group
    hash2group = {}
    for k, v in tqdm(group_hash.items()):
        for name in v:
            if name not in hash2group:
                hash2group[name] = [k]
            else:
                hash2group[name].append(k)
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

    pkl_save(unique_dict, output_file)
    print(sum([len(v) for v in unique_dict.values()]))
    print('Done!')
