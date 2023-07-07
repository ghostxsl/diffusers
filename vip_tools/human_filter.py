# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, basename
from tqdm import tqdm
import numpy as np

from diffusers.utils.vip_utils import *
from diffusers.data.utils import *
from diffusers.data.vos_client import VOSClient
from aistudio.utils.loader import ROOT_DIR
from aistudio.extensions.HumanPose import HumanPose
from aistudio.extensions.HumanPose.utils import POSE_CONFIG_DIR
from aistudio.extensions.HumanParsing import HumanParsing


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
        "--use_parsing",
        default=False,
        action="store_true",
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
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(join(args.out_dir, "pose"), exist_ok=True)

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

    parsing_thr = 0.003
    if args.use_parsing:
        parsing_infer = HumanParsing(
            model_path=join(weight_dir, "extensions/deeplabv3plus-xception-vocNov14_20-51-38_epoch-89.pth"),
            infer_size=512,
            device=device,
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

    out_list = []
    for name in tqdm(img_list):
        try:
            img = read_image(args, name)

            bboxes, _ = pose_infer.detector(img)
            if len(bboxes) == 0:
                # 0. 过滤无人
                out_list.append(name)
                continue
            else:
                # 取最大的bbox进行pose检测
                area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                ind = np.argmax(area)
                bboxes = bboxes[ind: ind + 1]
                _, pose = pose_infer(img, body=False, hand=False, human_bboxes=bboxes)

                body_kpts = pose['body']['keypoints'][0, :, 2]
                # 检测: 肩, 髋
                val_kpts = body_kpts[[2, 5, 8, 11]]
                # 检测: 肘
                elbow_kpts = body_kpts[[3, 6]]
                if (sum(val_kpts > pose_infer.body_kpt_thr) < len(val_kpts)) or (
                        sum(elbow_kpts > pose_infer.body_kpt_thr) == 0):
                    # 1. 过滤单人细节图
                    out_list.append(name)
                    continue

                if args.use_parsing:
                    label_parsing = parsing_infer(img)
                    img_area = img.width * img.height
                    if np.sum(label_parsing == 2) / img_area > parsing_thr or np.sum(
                        label_parsing == 13) / img_area > parsing_thr or np.sum(
                        label_parsing == 14) / img_area > parsing_thr or np.sum(
                        label_parsing == 15) / img_area > parsing_thr or np.sum(
                        label_parsing == 16) / img_area > parsing_thr or np.sum(
                        label_parsing == 17) / img_area > parsing_thr:
                        pkl_save(pose, join(args.out_dir, "pose", basename(name) + '.pose'))
                    else:
                        out_list.append(name)
                else:
                    pkl_save(pose, join(args.out_dir, "pose", basename(name) + '.pose'))
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
