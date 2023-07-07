# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, splitext, split, basename
import copy
import argparse
from tqdm import tqdm
from diffusers.data.utils import *
import numpy as np
import random
import requests
from PIL import Image
from io import BytesIO


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--data_file",
        default=None,
        type=str,
        help=" ")
    parser.add_argument(
        "--out_dir",
        default="output",
        type=str,
        help="Directory to output.")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(join(args.out_dir, "cache"), exist_ok=True)

    return args


def download_img(url):
    content = requests.get(url, timeout=5).content
    image = Image.open(BytesIO(content))
    return image


def f0_txt_to_json(input_file, cache_file):
    # 0. 清洗原始txt数据库中的行信息
    print("start f0 ...")
    with open(input_file, "r") as f:
        txt_list = f.readlines()
    out_list = []
    temp_line = []
    for line in tqdm(txt_list):
        if len(temp_line) >= 13:
            out_list.append(temp_line)
            temp_line = []

        line = line.split('\t')
        if len(line) >= 13:
            out_list.append(line)
        else:
            temp_line += line
    json_save(out_list, cache_file)
    print("end f0 ...")


def f1_get_group_infos(input_file, cache_file):
    # 1. 提取相关信息, 产出group信息
    # spu_id, group_sn, brand_store_sn, image_index, image_url
    print("start f1 ...")
    infos = load_file(input_file)
    out_dict = {}
    for line in tqdm(infos):
        if int(line[-5]) >= 5:
            continue

        key = "_".join([line[0], line[1], line[6]])
        if key not in out_dict:
            out_dict[key] = set([line[-4]])
        else:
            out_dict[key].add(line[-4])
    pkl_save(out_dict, cache_file)
    print("end f1 ...")


def f2_get_name2group(input_file, cache_file):
    # 2. 根据图片url, 反向映射group_name
    print("start f2 ...")
    group_infos = load_file(input_file)
    out_dict = {}
    for k, v in tqdm(group_infos.items()):
        for name in v:
            if name not in out_dict:
                out_dict[name] = [k]
            else:
                out_dict[name].append(k)
    pkl_save(out_dict, cache_file)
    print("end f2 ...")


def f3_merge_duplicate_group(group_file, name2group_file, cache_file):
    # 3. 根据group和name2group, 合并重复组
    print("start f3 ...")
    group_infos = load_file(group_file)
    name2group = load_file(name2group_file)

    unique_dict = {}
    done_list = set()
    for k, v in tqdm(group_infos.items()):
        if k in done_list:
            continue

        temp = []
        for name in v:
            temp.append(len(name2group[name]))

        if sum(temp) == max(temp) * len(temp):
            for d_k in name2group[name]:
                done_list.add(d_k)

            new_k = "_".join(sorted(name2group[name]))
            new_k = get_str_md5(new_k)
            if new_k not in unique_dict:
                unique_dict[new_k] = v
            else:
                unique_dict[new_k] += v
        elif sum(temp) < max(temp) * len(temp):
            done_list.add(k)
            new_k = get_str_md5(k)
            if new_k not in unique_dict:
                unique_dict[new_k] = v
            else:
                unique_dict[new_k] += v
        else:
            raise Exception('Error: f3!')

    pkl_save(unique_dict, cache_file)
    print("end f3 ...")


def f4_merge_duplicate_image_name(input_file, output_file, urls_file):
    # 4. 根据unique_group, 去除重复图片名
    print("start f4 ...")
    group_infos = load_file(input_file)

    name2url = {}
    url2group = {}
    for k, v in tqdm(group_infos.items()):
        for url in v:
            if url not in url2group:
                url2group[url] = [k]
            else:
                url2group[url].append(k)

            name = basename(url)
            if name not in name2url:
                name2url[name] = [url]
            else:
                name2url[name].append(url)

    redundance = [v for v in name2url.values() if len(v) > 1]
    drop_list = []
    for line in tqdm(redundance):
        temp_size = []
        save_ind = None
        for i, url in enumerate(line):
            img = download_img(url)
            w, h = img.size
            temp_size.append([w, h])
            if w == h and w == 1200:
                save_ind = i
                break

        if save_ind is not None:
            drop_list += [a for i, a in enumerate(line) if i != save_ind]
        else:
            temp_size = np.array(temp_size)
            max_ind = np.argmax(temp_size[:, 0] * temp_size[:, 1])
            drop_list += [a for i, a in enumerate(line) if i != max_ind]

    for url in drop_list:
        groups = url2group[url]
        for group in groups:
            if group not in group_infos:
                continue
            v = copy.deepcopy(group_infos[group])
            if url in v:
                v.remove(url)

            if len(v) == 0:
                del group_infos[group]
            else:
                group_infos[group] = v

    pkl_save(group_infos, output_file)

    url_list = []
    for k, v in tqdm(group_infos.items()):
        for url in v:
            url_list.append(url)
    pkl_save(url_list, urls_file)
    print("end f4 ...")


def main(args):
    cache_dir = join(args.out_dir, "cache")
    input_txt_file = args.data_file

    if splitext(input_txt_file)[1] == '.txt':
        # 0. 清洗原始txt数据库中的行信息
        clean_json = join(cache_dir, splitext(basename(input_txt_file))[0] + ".json")
        f0_txt_to_json(input_txt_file, clean_json)
    elif splitext(input_txt_file)[1] == '.json':
        clean_json = input_txt_file
    else:
        raise Exception("Error: f0")

    # 1. 提取相关信息, 产出group信息
    group_pkl = join(cache_dir, splitext(basename(input_txt_file))[0] + ".group")
    f1_get_group_infos(clean_json, group_pkl)

    # 2. 根据图片url, 反向映射group_name
    name2group_pkl = join(cache_dir, splitext(basename(input_txt_file))[0] + ".name2group")
    f2_get_name2group(group_pkl, name2group_pkl)

    # # 3. 根据group和name2group, 合并重复组
    unique_group_pkl = join(cache_dir, splitext(basename(input_txt_file))[0] + ".unique_group")
    f3_merge_duplicate_group(group_pkl, name2group_pkl, unique_group_pkl)

    # 4. 根据unique_group, 去除重复图片名
    output_group_file = join(args.out_dir, splitext(basename(input_txt_file))[0] + ".group2url")
    urls_file = join(args.out_dir, splitext(basename(input_txt_file))[0] + ".urls")
    f4_merge_duplicate_image_name(unique_group_pkl, output_group_file, urls_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
