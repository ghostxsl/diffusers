# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, split, exists, basename
from tqdm import tqdm
import numpy as np

from diffusers.utils.vip_utils import *
from diffusers.data.utils import *
from diffusers.data.vos_client import VOSClient
from aistudio.utils.loader import ROOT_DIR
from aistudio.extensions.HumanPose import HumanPose
from aistudio.extensions.HumanPose.utils import POSE_CONFIG_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--img_dir",
        default=None,
        type=str,
        help="Path to reference image.")
    parser.add_argument(
        "--out_file",
        default="output.json",
        type=str,
        help="File name to save.")
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        help="Directory to save.")
    parser.add_argument(
        "--vos_pkl",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--bbox_thr",
        default=0.2,
        type=float,
        help="")

    parser.add_argument(
        "--rank",
        default=None,
        type=int)
    parser.add_argument(
        "--num_ranks",
        default=None,
        type=int)
    parser.add_argument(
        "--weight_dir",
        default=None,
        type=str,
        help="Directory to weights.")
    parser.add_argument(
        "--device",
        default='cuda',
        type=str,
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")

    args = parser.parse_args()
    out_dir, _ = split(args.out_file)
    if out_dir and not exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    return args


def read_image(args, name):
    if args.vos_pkl is None:
        img = load_image(join(args.img_dir, name))
    else:
        img = args.vos_client.download_vos_pil(name)
        img = load_image(img)

    return img


def main(args):
    device = args.device
    weight_dir = args.weight_dir or join(ROOT_DIR, "weights")

    pose_infer = HumanPose(
        det_cfg=join(POSE_CONFIG_DIR, "rtmdet_l_8xb32-300e_coco.py"),
        det_pth=join(weight_dir, "extensions/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"),
        bodypose_cfg=join(POSE_CONFIG_DIR, "rtmpose-l_8xb256-420e_body8-256x192.py"),
        bodypose_pth=join(weight_dir, "extensions/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"),
        wholebodypose_cfg=join(POSE_CONFIG_DIR, "dwpose_l_wholebody_384x288.py"),
        wholebodypose_pth=join(weight_dir, "extensions/dw-ll_ucoco_384.pth"),
        device=device,
        bbox_thr=args.bbox_thr,
    )

    if args.vos_pkl is None:
        img_list = sorted(os.listdir(args.img_dir))
    else:
        img_list = load_file(args.vos_pkl)
        args.vos_client = VOSClient()

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        total_num = len(img_list)
        stride = int(total_num / args.num_ranks)

        start_idx = stride * args.rank
        end_idx = stride * (args.rank + 1) if args.rank + 1 < args.num_ranks else len(img_list)
        img_list = img_list[start_idx: end_idx]

    save_list = []
    for name in tqdm(img_list):
        try:
            img = read_image(args, name)
            _, pose = pose_infer(img, body=False, hand=False)
            # 保留图像中有人的
            if len(pose['body']['bboxes']) > 0:
                width, height = pose['width'], pose['height']
                bboxes = pose['body']['bboxes'][0] * np.array([width, height, width, height])
                x1, y1, x2, y2 = bboxes
                bh, bw = y2 - y1, x2 - x1
                kpts = pose['body']['keypoints'][0]
                # 过滤人像占比太小的图像
                if bh < height * 0.1 or bw < width * 0.1 or np.max(kpts[..., -1]) < 0.3:
                    continue

                if args.out_dir is not None:
                    if args.vos_pkl is None:
                        os.makedirs(args.out_dir, exist_ok=True)
                        pkl_save(pose, join(args.out_dir, basename(name)))
                    else:
                        args.vos_client.upload_vos_pkl(
                            pose, join(args.out_dir, name)
                        )
                save_list.append(name)
        except:
            continue

    if args.rank is not None:
        out_dir, out_file = split(args.out_file)
        out_file = splitext(out_file)[0] + f"_{args.rank}.json"
        json_save(save_list, join(out_dir, out_file))
    else:
        json_save(save_list, args.out_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
