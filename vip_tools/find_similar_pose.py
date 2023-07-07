# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, exists, splitext, basename
import argparse
import time
import random
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

from diffusers.utils.vip_utils import *
from diffusers.data.utils import *

from aistudio.utils.loader import ROOT_DIR
from aistudio.extensions.HumanPose import HumanPose
from aistudio.extensions.HumanPose.utils import POSE_CONFIG_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--ref_img",
        default=None,
        type=str,
        help="Path to reference image.")
    parser.add_argument(
        "--ref_dir",
        default=None,
        type=str,
        help="Directory to reference image.")

    parser.add_argument(
        "--pose_dir",
        default=None,
        type=str,
        help="Directory to pose image.")
    parser.add_argument(
        "--pose_lib",
        default=None,
        type=str,
        help="Path to pose lib.")
    parser.add_argument(
        "--out_dir",
        default="output",
        type=str,
        help="Directory to save.")

    parser.add_argument(
        "--topk",
        default=1,
        type=int,
        help="")
    parser.add_argument(
        "--topk_thr",
        default=0.9,
        type=float,
        help="")
    parser.add_argument(
        "--topk_stride",
        default=None,
        type=float,
        help="")
    parser.add_argument(
        "--display",
        default=False,
        action="store_true",
        help="")

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

    if len(args.resolution.split("x")) == 1:
        args.resolution = [int(args.resolution), ] * 2
    elif len(args.resolution.split("x")) == 2:
        args.resolution = [int(r) for r in args.resolution.split("x")]
    else:
        raise Exception(f"Error `resolution` type({type(args.resolution)}): {args.resolution}.")

    args.device = torch.device(args.device)
    if args.dtype == 'fp16':
        args.dtype = torch.float16
    else:
        args.dtype = torch.float32

    return args


class FindSimilarPose(object):
    def __init__(self, args):
        self.args = args
        self.topk_thr = args.topk_thr
        self.topk_stride = args.topk_stride
        self.infer_size = args.resolution
        self.weight_dir = args.weight_dir or join(ROOT_DIR, "weights")
        self.device = args.device
        self.dtype = args.dtype
        if args.pose_lib:
            self.pose_lib = load_file(args.pose_lib)
        else:
            self.pose_lib = self.load_pose_lib(args.pose_dir)

        self.pose_infer = HumanPose(
            det_cfg=join(POSE_CONFIG_DIR, "rtmdet_l_8xb32-300e_coco.py"),
            det_pth=join(self.weight_dir, "extensions/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"),
            bodypose_cfg=join(POSE_CONFIG_DIR, "rtmpose-l_8xb256-420e_body8-256x192.py"),
            bodypose_pth=join(self.weight_dir,
                              "extensions/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"),
            wholebodypose_cfg=join(POSE_CONFIG_DIR, "dwpose_l_wholebody_384x288.py"),
            wholebodypose_pth=join(self.weight_dir, "extensions/dw-ll_ucoco_384.pth"),
            device=self.device,
            bbox_thr=0.2,
        )

    def normalize_pose(self, pose, crop_size):
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

    def load_pose_lib(self, pose_dir):
        print("Load and normalize pose lib...")
        out = []
        for name in tqdm(os.listdir(pose_dir)):
            try:
                pose = load_file(join(pose_dir, name))
                body_kpts, _ = self.normalize_pose(pose, self.infer_size)
                if body_kpts is None:
                    continue

                out.append(body_kpts)
            except:
                continue
        return {
            "dts": np.concatenate(out)
        }

    def draw_pose(self, pose, draw_size, kpt_thr=0.3):
        height, width = draw_size
        canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        canvas = draw_bodypose(canvas, pose[..., :2], pose[..., 2] > kpt_thr)
        return Image.fromarray(canvas)

    def topk_sampling(self, oks, topk=1):
        topk_ind = np.argsort(-oks)
        out_ind, out_val = [], []
        if self.topk_stride is not None:
            sroted_oks = oks[topk_ind]
            stride_interval = np.linspace(1, -1, int(2 / self.topk_stride) + 1)

            ii = 0
            for h, l in zip(stride_interval[:-1], stride_interval[1:]):
                if len(out_ind) == topk:
                    break

                temp_ind, temp_val = [], []
                while sroted_oks[ii] >= l and sroted_oks[ii] < h:
                    temp_ind.append(topk_ind[ii])
                    temp_val.append(sroted_oks[ii])
                    ii += 1
                if self.topk_thr >= l and len(temp_ind) > 0:
                    ind = random.choice(temp_ind)
                    out_ind.append(ind)
                    out_val.append(oks[ind])
        else:
            while oks[topk_ind[0]] > self.topk_thr:
                topk_ind = np.delete(topk_ind, 0)
            out_ind = [int(topk_ind[i]) for i in range(topk)]
            out_val = [oks[i] for i in out_ind]

        return out_ind, out_val

    def topk_pose(self, pose, topk=1, pose_label="dts"):
        w, h = pose['width'], pose['height']
        kpts = pose['body']['keypoints'][0:1] * np.array([w, h, 1.])
        out_pose_img = [self.draw_pose(kpts, (h, w))]

        norm_pose, affine_params = self.normalize_pose(pose, self.infer_size)
        n_ratio, pad_shift, bbox = affine_params
        # 计算OKS相似度
        norm_pose_img = [self.draw_pose(norm_pose, self.infer_size)]
        oks = compute_OKS(norm_pose, bbox[None], self.pose_lib[pose_label])[0]

        topk_ind, topk_oks = self.topk_sampling(oks, topk)

        for ind, pose_similarity in zip(topk_ind, topk_oks):
            t_pose = self.pose_lib[pose_label][ind]
            norm_pose_img.append(self.draw_pose(t_pose[None], self.infer_size))

            t_pose[..., :2] /= n_ratio
            t_pose[..., :2] -= np.array(pad_shift)
            t_pose[..., :2] += bbox[:2]
            out_pose_img.append(self.draw_pose(t_pose[None], (h, w)))

        return norm_pose_img, out_pose_img

    def __call__(self, image, topk=1, pose_label="dts", **kwargs):
        # 推理pose
        _, pose = self.pose_infer(image, body=False, hand=False)
        if len(pose['body']['bboxes']) == 0:
            return None, None

        # topk相似pose
        norm_pose_img, out_pose_img = self.topk_pose(pose, topk, pose_label)

        return norm_pose_img, out_pose_img


def get_img_path_list(img_file=None, img_dir=None):
    img_path_list = []
    if img_file is None and img_dir is None:
        raise Exception("Please specify `ref_img` or `ref_dir`.")
    elif img_file is not None and img_dir is None:
        # image
        assert exists(img_file)
        img_path_list.append(img_file)
    elif img_file is None and img_dir is not None:
        # dir
        assert exists(img_dir)
        img_path_list = os.listdir(img_dir)
        img_path_list = [join(img_dir, a) for a in img_path_list if splitext(
            a)[1].lower() in ['.jpg', '.jpeg', '.png']]
    else:
        raise Exception("`ref_img` and `ref_dir` cannot both be assigned.")

    return img_path_list


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

    find_sim_pose = FindSimilarPose(args)
    ref_img_list = get_img_path_list(args.ref_img, args.ref_dir)

    for i, file in enumerate(ref_img_list):
        try:
            t1 = time.time()
            print(f"{i + 1}: {file}")
            name = basename(file)
            ref_img = load_image(file)

            pose_label = name.split('_')[0]
            pose_label = "dts" if pose_label not in gender_map else gender_map[pose_label]
            norm_pose_img, out_pose_img = find_sim_pose(
                ref_img, topk=args.topk, pose_label=pose_label)
            if out_pose_img is None:
                print(f"Error: reference image no bbox: {file}")
                continue

            if args.display:
                norm_pose_img = np.concatenate(norm_pose_img, axis=1)
                out_pose_img = [a.resize(args.resolution[::-1], 1) for a in out_pose_img]
                out_pose_img = np.concatenate(out_pose_img, axis=1)
                out_imgs = np.concatenate([norm_pose_img, out_pose_img])
                Image.fromarray(out_imgs).save(join(args.out_dir, basename(file)))
            else:
                for j, im in enumerate(out_pose_img[1:]):
                    im.save(join(args.out_dir, splitext(basename(file))[0] + f'_{j}.jpg'))
            t2 = time.time()
            print(f"time: {t2 - t1}s")
        except Exception as e:
            print(f"Unknown error: {e}")
            continue


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
