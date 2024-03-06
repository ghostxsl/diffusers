# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, split
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch

from diffusers.utils.vip_utils import *
from diffusers.data.utils import *
from aistudio.data.utils import erode_mask
from aistudio.utils.loader import ROOT_DIR
from aistudio.extensions.HumanPose import HumanPose
from aistudio.extensions.HumanPose.utils import POSE_CONFIG_DIR
from aistudio.extensions.ClothingSeg import ClothingSeg


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--img_dir",
        default=None,
        type=str,
        help="Path to reference image.")
    parser.add_argument(
        "--out_dir",
        default="output",
        type=str,
        help="Directory to save.")

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
    parser.add_argument(
        "--dtype",
        default='fp16',
        type=str,
        help="Data type to use (e.g. fp16, fp32, etc.)")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    args.device = torch.device(args.device)
    if args.dtype == 'fp16':
        args.dtype = torch.float16
    else:
        args.dtype = torch.float32

    return args


def drop_clothing(cloth_infer, img):
    clothing_mask = cloth_infer(img)
    clothing_mask = erode_mask(np.array(clothing_mask), erode_factor=21)
    img = np.array(img)
    img = np.where(clothing_mask[..., None] > 0, np.zeros_like(img), img)
    return Image.fromarray(img)


def main(args):
    img_list = sorted(os.listdir(args.img_dir))

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        total_num = len(img_list)
        stride = int(total_num / args.num_ranks)

        start_idx = stride * args.rank
        end_idx = stride * (args.rank + 1) if args.rank + 1 < args.num_ranks else len(img_list)
        img_list = img_list[start_idx: end_idx]

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
        bbox_thr=0.5,
    )
    cloth_infer = ClothingSeg(
        model_path=join(weight_dir, "extensions/cloth_seg_v5.0.pt"),
        infer_size=512,
        device=device,
    )

    out_list = []
    for name in tqdm(img_list):
        try:
            img = load_image(join(args.img_dir, name))

            drop_cloth_img = drop_clothing(cloth_infer, img)
            bboxes, _ = pose_infer.detector(drop_cloth_img)
            if len(bboxes) != 1:
                # 0. 过滤无人和多人
                out_list.append(name)
                continue
            else:
                _, pose = pose_infer(img, human_bboxes=bboxes)
                # 取最大的bbox
                area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                ind = np.argmax(area)
                body_kpts = pose['body']['keypoints'][ind, :, 2]
                # 检测: 肩, 髋
                val_kpts = body_kpts[[2, 5, 8, 11]]
                # 检测: 肘
                elbow_kpts = body_kpts[[3, 6]]
                if (sum(val_kpts > pose_infer.body_kpt_thr) < len(val_kpts)) or (
                        sum(elbow_kpts > pose_infer.body_kpt_thr) == 0):
                    # 1. 过滤单人细节图
                    out_list.append(name)
                    continue

                body_kpts = body_kpts.tolist()
                del body_kpts[1]
                if max(body_kpts) < 0.59:
                    # 2. 过滤无pose但有检测框的图
                    out_list.append(name)
        except:
            out_list.append(name)

    if args.rank is not None:
        json_save(out_list, join(args.out_dir, f"drop_human_{args.rank}.json"))
    else:
        json_save(out_list, join(args.out_dir, "drop_human.json"))


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
