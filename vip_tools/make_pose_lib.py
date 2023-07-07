# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, exists, splitext, basename, split
import argparse
from tqdm import tqdm
import numpy as np

from diffusers.data.utils import *
from diffusers.data.vos_client import VOSClient


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
        "--vos_pkl",
        default=None,
        type=str,
        help="Path to image list on vos.")
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
        "--out_file",
        default="output.lib",
        type=str,
        help="File path to save.")
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
    w, h = pose['width'], pose['height']
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


def process_gender_file(file_path):
    if file_path is None:
        return {}

    data_dict = load_file(file_path)
    out_dict = {splitext(k)[0]: v['gender'] for k, v in data_dict.items()}

    return out_dict


def main(args):
    if args.vos_pkl is not None:
        pose_list = load_file(args.vos_pkl)
        args.vos_client = VOSClient()
    else:
        pose_list = sorted(os.listdir(args.pose_dir))
        if args.gender_file is not None:
            gender_dict = process_gender_file(args.gender_file)

    print("Make and normalize pose lib...")
    out_pose = {
        "body": {"person": []},
        "hand": {"person": []},
        "face": {"person": []},
        "file_name": {"person": []},
    }

    for name in tqdm(pose_list):
        try:
            pose = load_file(join(args.pose_dir, name))
            kpts, _ = normalize_pose(pose, args.resolution)
            if kpts is None:
                continue

            body_kpts, hand_kpts, face_kpts = kpts
            out_pose["body"]["person"].append(body_kpts)
            out_pose["hand"]["person"].append(hand_kpts)
            out_pose["face"]["person"].append(face_kpts)
            out_pose["file_name"]["person"].append(splitext(name)[0] + '.jpg')

            if args.gender_file is not None and splitext(name)[0] in gender_dict:
                gender = gender_dict[splitext(name)[0]]
                gender = gender_map.get(gender, None)
                if gender is not None:
                    if gender not in out_pose["body"]:
                        out_pose["body"][gender] = [body_kpts]
                    else:
                        out_pose["body"][gender].append(body_kpts)

                    if gender not in out_pose["hand"]:
                        out_pose["hand"][gender] = [hand_kpts]
                    else:
                        out_pose["hand"][gender].append(hand_kpts)

                    if gender not in out_pose["face"]:
                        out_pose["face"][gender] = [face_kpts]
                    else:
                        out_pose["face"][gender].append(face_kpts)

                    if gender not in out_pose["file_name"]:
                        out_pose["file_name"][gender] = [splitext(name)[0] + '.jpg']
                    else:
                        out_pose["file_name"][gender].append(splitext(name)[0] + '.jpg')
        except:
            continue

    out_pose["body"] = {k: np.concatenate(v) for k, v in out_pose["body"].items()}
    out_pose["hand"] = {k: np.concatenate(v) for k, v in out_pose["hand"].items()}
    out_pose["face"] = {k: np.concatenate(v) for k, v in out_pose["face"].items()}

    if args.src_pose_file is not None:
        src_pose = load_file(args.src_pose_file)
        for k0 in ["body", "hand", "face"]:
            for k, v in src_pose[k0].items():
                if k not in out_pose[k0]:
                    out_pose[k0][k] = v
                else:
                    out_pose[k0][k] = np.concatenate([v, out_pose[k0][k]])

    pkl_save(out_pose, args.out_file)
    print(f"Save file to {args.out_file}")
    [print(f"body: {k}: {len(v)}") for k, v in out_pose["body"].items()]
    [print(f"hand: {k}: {len(v)}") for k, v in out_pose["hand"].items()]
    [print(f"face: {k}: {len(v)}") for k, v in out_pose["face"].items()]
    [print(f"file_name: {k}: {len(v)}") for k, v in out_pose["file_name"].items()]


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
