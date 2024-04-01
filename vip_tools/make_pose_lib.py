# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, exists, splitext, basename
import argparse
from tqdm import tqdm
import numpy as np
import csv

from diffusers.data.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--pose_dir",
        default=None,
        type=str,
        help="Directory to pose.")
    parser.add_argument(
        "--src_pose_file",
        default=None,
        type=str,
        help="Path to src pose file.")
    parser.add_argument(
        "--out_dir",
        default="output",
        type=str,
        help="Directory to save.")
    parser.add_argument(
        "--gender_file",
        default=None,
        type=str,
        help="Path to gender file.")

    parser.add_argument(
        "--resolution",
        type=str,
        default="768x576",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if len(args.resolution.split("x")) == 1:
        args.resolution = [int(args.resolution), ] * 2
    elif len(args.resolution.split("x")) == 2:
        args.resolution = [int(r) for r in args.resolution.split("x")]
    else:
        raise Exception(f"Error `resolution` type({type(args.resolution)}): {args.resolution}.")

    return args


def normalize_pose(pose, crop_size=(768, 576)):
    ch, cw = crop_size
    w, h = pose['width'], pose['height']
    bboxes = pose['body']['bboxes']
    if len(bboxes) == 0:
        return None, None

    # 取得分最高的bbox, 变换为原始坐标
    bbox = bboxes[0] * np.array([w, h, w, h])
    # 取得分最高的keypoints, 变换为原始坐标
    body_kpts = pose['body']['keypoints'][0] * np.array([w, h, 1.])
    # 平移关键点
    body_kpts[..., :2] -= bbox[:2]

    # 按比例resize
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ratio_h, ratio_w = ch / bh, cw / bw

    out_ratio = ratio_h
    pad_shift = [0, 0] # x, y
    if ratio_h < ratio_w:
        # 按高 resize
        ow = int(bh / ch * cw)
        expand_w = ow - bw
        body_kpts[..., 0] += int(expand_w / 2)
        body_kpts[..., :2] *= ratio_h

        pad_shift[0] = int(expand_w / 2)
    elif ratio_h > ratio_w:
        # 按宽 resize
        oh = int(bw / cw * ch)
        expand_h = oh - bh
        body_kpts[..., 1] += int(expand_h / 2)
        body_kpts[..., :2] *= ratio_w

        out_ratio = ratio_w
        pad_shift[1] = int(expand_h / 2)
    else:
        body_kpts[..., :2] *= ratio_h

    return body_kpts[None], (out_ratio, pad_shift, bbox)


def process_gender_file(file_path):
    if file_path is None:
        return {}

    out_dict = {}
    if file_path.endswith(".csv"):
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                k = splitext(basename(line[0]))[0]
                out_dict[k] = line[1]
    else:
        out_dict = load_file(file_path)

    return out_dict


def main(args):
    gender_map = {
        "男性": "man",
        "女性": "woman",
        "男童": "boy",
        "女童": "girl",
        "man": "man",
        "woman": "woman",
        "boy": "boy",
        "girl": "girl",
    }

    pose_list = sorted(os.listdir(args.pose_dir))
    gender_dict = process_gender_file(args.gender_file)

    print("Make and normalize pose lib...")
    out_pose = {"dts": []}
    for name in tqdm(pose_list):
        try:
            pose = load_file(join(args.pose_dir, name))
            body_kpts, _ = normalize_pose(pose, args.resolution)
            if body_kpts is None:
                continue

            out_pose["dts"].append(body_kpts)
            if splitext(name)[0] in gender_dict:
                gender = gender_dict[splitext(name)[0]]
                gender = gender_map[gender] if gender in gender_map else None
                if gender is not None:
                    if gender not in out_pose:
                        out_pose[gender] = [body_kpts]
                    else:
                        out_pose[gender].append(body_kpts)
        except:
            continue

    out_pose = {k: np.concatenate(v) for k, v in out_pose.items()}

    if args.src_pose_file is not None:
        src_pose = load_file(args.src_pose_file)
        for k, v in src_pose.items():
            if k not in out_pose:
                out_pose[k] = v
            else:
                out_pose[k] = np.concatenate([v, out_pose[k]])

    pkl_save(out_pose, join(args.out_dir, 'mote_pose.lib'))
    print(f"Save file to {join(args.out_dir, 'mote_pose.lib')}")
    [print(f"{k}: {len(v)}") for k, v in out_pose.items()]


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
