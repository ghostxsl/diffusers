# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, exists, splitext, basename, split
import argparse
from tqdm import tqdm
import numpy as np
import shutil

from diffusers.data.utils import *


gender_map = {
    "男性": "man",
    "女性": "woman",
    "男童": "boy",
    "女童": "girl",
    "man": "man",
    "woman": "woman",
    "boy": "boy",
    "girl": "girl",
    "male": "man",
    "female": "woman",
}


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--pose_dir",
        default=None,
        type=str,
        help="Directory to pose.")
    parser.add_argument(
        "--out_file",
        default="output.lib",
        type=str,
        help="File path to save.")

    parser.add_argument(
        "--resolution",
        type=str,
        default="768x576",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )
    parser.add_argument(
        "--old_pose_file",
        default=None,
        type=str,
        help="Path to src pose file.")

    args = parser.parse_args()
    out_dir, _ = split(args.out_file)
    if out_dir and not exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if len(args.resolution.split("x")) == 1:
        args.resolution = [int(args.resolution), ] * 2
    elif len(args.resolution.split("x")) == 2:
        args.resolution = [int(r) for r in args.resolution.split("x")]
    else:
        raise Exception(f"Error `resolution` type({type(args.resolution)}): {args.resolution}.")

    return args


def put_item_in_dict(out_dict, keys, item):
    k = keys.pop(0)
    if len(keys) == 0:
        if k in out_dict:
            out_dict[k].append(item)
            return
        else:
            out_dict[k] = [item]
    else:
        if k not in out_dict:
            out_dict[k] = {}
        put_item_in_dict(out_dict[k], keys, item)


def process_img_dict(src_img_dir, out_dir, save_dict, out_dict=None):
    out_dict = {} if out_dict is None else out_dict
    for root, dir_, files in os.walk(src_img_dir):
        for name in files:
            img_md5 = get_file_md5(join(root, name))
            src = join(root, name)
            temp = root.replace(f"{src_img_dir}/", "")
            temp = temp.split('/')
            prefix = '_'.join(temp)

            dst = join(out_dir, f"{prefix}_{img_md5}.jpg")
            shutil.copy(src, dst)
            print(f"{src} -> {dst}")

            put_item_in_dict(out_dict, temp, f"{prefix}_{img_md5}.jpg")

    pkl_save(out_dict, f"{save_dict}/xhs_pose.dict")


def normalize_pose(pose, crop_size=(768, 576)):
    bboxes = pose['body']['bboxes']
    if len(bboxes) == 0:
        return None, None
    if np.max(pose['body']['keypoints'][0]) < 0.3:
        return None, None
    if np.sum(pose['body']['keypoints'][0, :, -1] > 0.3) < 4:
        return None, None

    # 取得分最高的bbox, 变换为原始坐标
    ch, cw = crop_size
    h, w = pose['height'], pose['width']
    bbox = bboxes[0] * np.array([w, h, w, h])
    # 取得分最高的keypoints, 变换为原始坐标
    body_kpts = pose['body']['keypoints'][0] * np.array([w, h, 1.])
    hand_kpts = pose['hand']['keypoints'][:2] * np.array([w, h, 1.])
    face_kpts = pose['face']['keypoints'][0] * np.array([w, h, 1.])
    # 取关键点的外接矩和预测的bbox的并集
    temp_kpts = body_kpts[body_kpts[:, -1] > 0.3, :2]
    min_xy = np.min(temp_kpts, axis=0)
    max_xy = np.max(temp_kpts, axis=0)
    new_bbox = np.concatenate([
        np.min(np.stack([bbox[:2], min_xy]), axis=0),
        np.max(np.stack([bbox[2:], max_xy]), axis=0)
    ])
    bbox = new_bbox
    # 平移关键点
    body_kpts[..., :2] -= bbox[:2]
    hand_kpts[..., :2] -= bbox[:2]
    face_kpts[..., :2] -= bbox[:2]

    # 按比例resize
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ratio_h, ratio_w = ch / bh, cw / bw

    out_ratio = ratio_h
    pad_shift = [0, 0]  # x, y
    if ratio_h < ratio_w:
        # 按高 resize
        ow = int(bh / ch * cw)
        expand_w = ow - bw
        offset_x = int(expand_w / 2)
        # body
        body_kpts[..., 0] += offset_x
        body_kpts[..., :2] *= ratio_h
        # hand
        hand_kpts[..., 0] += offset_x
        hand_kpts[..., :2] *= ratio_h
        # face
        face_kpts[..., 0] += offset_x
        face_kpts[..., :2] *= ratio_h

        pad_shift[0] = int(expand_w / 2)
    elif ratio_h > ratio_w:
        # 按宽 resize
        oh = int(bw / cw * ch)
        expand_h = oh - bh
        offset_y = int(expand_h / 2)
        # body
        body_kpts[..., 1] += offset_y
        body_kpts[..., :2] *= ratio_w
        # hand
        hand_kpts[..., 1] += offset_y
        hand_kpts[..., :2] *= ratio_w
        # face
        face_kpts[..., 1] += offset_y
        face_kpts[..., :2] *= ratio_w

        out_ratio = ratio_w
        pad_shift[1] = int(expand_h / 2)
    else:
        body_kpts[..., :2] *= ratio_h
        hand_kpts[..., :2] *= ratio_h
        face_kpts[..., :2] *= ratio_h

    return (body_kpts[None], hand_kpts[None], face_kpts[None]), (out_ratio, pad_shift, bbox)


def main(args):
    pose_list = sorted(os.listdir(args.pose_dir))

    print("Make and normalize pose lib...")
    out_pose = {}
    if args.old_pose_file:
        out_pose = load_file(args.old_pose_file)

    for name in tqdm(pose_list):
        try:
            pose = load_file(join(args.pose_dir, name))
            kpts, _ = normalize_pose(pose, args.resolution)
            if kpts is None:
                continue

            temp = name.split('_')[:-1]
            put_item_in_dict(out_pose, temp, kpts)

        except Exception as e:
            print(e)
            continue

    pkl_save(out_pose, args.out_file)
    print(f"Save file to {args.out_file}")

    def log_statistic(key, values, out=[]):
        if isinstance(values, list):
            out.append(len(values))
            print(f"{key}: {len(values)}")
        elif isinstance(values, dict):
            for k in values.keys():
                log_statistic(f"{key}_{k}", values[k], out)
        else:
            raise Exception(f"Error {key} type: {type(values)}.")

    for k, v in out_pose.items():
        temp = []
        log_statistic(k, v, temp)
        print(f"====== {k}: {sum(temp)} ======")


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
