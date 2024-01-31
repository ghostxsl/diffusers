# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, split
import numpy as np
from PIL import Image
import torch

from diffusers.pipelines.vip.vip_animate_image import VIPAnimateImagePipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.models.controlnetxs import ControlNetXSModel
from diffusers.models.referencenet import ReferenceNetModel
from diffusers.utils.vip_utils import *
from diffusers.data import DrawPose

from aistudio.utils.loader import ROOT_DIR
from aistudio.extensions.HumanPose import HumanPose
from aistudio.extensions.HumanPose.utils import POSE_CONFIG_DIR


_draw = DrawPose(prob_hand=1.0, prob_face=0.0)


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
        "--pose_img",
        default=None,
        type=str,
        help="Path to pose image.")
    parser.add_argument(
        "--pose_dir",
        default=None,
        type=str,
        help="Directory to pose image.")

    parser.add_argument(
        "--out_dir",
        default="output",
        type=str,
        help="Directory to save.")

    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/xsl/wilson.xu/weights/film",
    )
    parser.add_argument(
        "--controlnet_model_path",
        type=str,
        default="/xsl/wilson.xu/s1_7k_0129/controlnet",
    )
    parser.add_argument(
        "--referencenet_model_path",
        type=str,
        default="/xsl/wilson.xu/s1_7k_0129/referencenet",
    )
    parser.add_argument(
        "--weight_dir",
        default=None,
        type=str,
        help="Directory to weights.")
    parser.add_argument(
        "--resolution",
        type=str,
        default="1024x768",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )

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
        args.resolution = int(args.resolution)
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
    ref_img_list = get_img_path_list(args.ref_img, args.ref_dir)
    pose_img_list = get_img_path_list(args.pose_img, args.pose_dir)

    device = args.device
    dtype = args.dtype
    controlnet = ControlNetXSModel.from_pretrained(args.controlnet_model_path).to(device, dtype=dtype)
    referencenet = ReferenceNetModel.from_pretrained(args.referencenet_model_path).to(device, dtype=dtype)
    pipe = VIPAnimateImagePipeline.from_pretrained(
        args.base_model_path,
        controlnet=controlnet,
        referencenet=referencenet,
        torch_dtype=dtype).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    weight_dir = args.weight_dir or join(ROOT_DIR, "weights")
    pose_infer = HumanPose(
        det_cfg=join(POSE_CONFIG_DIR, "rtmdet_l_8xb32-300e_coco.py"),
        det_pth=join(weight_dir, "extensions/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"),
        bodypose_cfg=join(POSE_CONFIG_DIR, "rtmpose-l_8xb256-420e_body8-256x192.py"),
        bodypose_pth=join(weight_dir, "extensions/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"),
        wholebodypose_cfg=join(POSE_CONFIG_DIR, "dwpose_l_wholebody_384x288.py"),
        wholebodypose_pth=join(weight_dir, "extensions/dw-ll_ucoco_384.pth"),
        device=device
    )
    pose_list = []
    for file in pose_img_list:
        img = load_image(file)
        res, _ = pose_infer(img)
        pose_list.append(Image.fromarray(res))

    for i, file in enumerate(ref_img_list):
        ref_img = load_image(file)
        print(f"{i + 1}: {file}")
        seed = get_fixed_seed(-1)
        generator = get_torch_generator(seed, device=device)

        os.makedirs(join(args.out_dir, split(file)[1]), exist_ok=True)
        for j, pose in enumerate(pose_list):
            out = pipe(
                prompt="",
                reference_image=ref_img,
                control_image=pose,
                height=args.resolution[0],
                width=args.resolution[1],
                num_inference_steps=25,
                guidance_scale=2.0,
                negative_prompt="",
                num_images_per_prompt=1,
                generator=generator,
            )[0]
            out_img = np.concatenate(
                [ref_img.resize(args.resolution[::-1], 1),
                 pose.resize(args.resolution[::-1], 1),
                 out], axis=1)
            Image.fromarray(out_img).save(join(args.out_dir, split(file)[1], f"{j}.png"))

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
