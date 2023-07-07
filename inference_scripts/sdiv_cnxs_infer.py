# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, basename
import numpy as np
from PIL import Image
import torch

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.controlnetxs import ControlNetXSModel
from diffusers.pipelines.vip.vip_sdiv_controlnetxs import VIPIVControlNetXSPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler, KDPMPP2MDiscreteScheduler
from diffusers.utils.vip_utils import *
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

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
        "--out_dir",
        default="output",
        type=str,
        help="Directory to save.")

    parser.add_argument(
        "--resolution",
        type=str,
        default="768",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )
    parser.add_argument(
        "--unet_model_path",
        type=str,
        default="/apps/dat/cv/xsl/exp_animate/sd_i2i_new/unet",
    )
    parser.add_argument(
        "--controlnet_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--scheduler_path",
        type=str,
        default="/apps/dat/cv/xsl/exp_animate/scheduler",
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/vae_ft_mse",
    )
    parser.add_argument(
        "--image_encoder_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/IP-Adapter/image_encoder",
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
        args.resolution = [int(args.resolution),] * 2
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


def pad_image(img, pad_values=255):
    w, h = img.size
    img = np.array(img)
    if w > h:
        pad_ = w - h
        img = np.pad(
            img,
            ((pad_ // 2, pad_ - pad_ // 2), (0, 0), (0, 0)),
            constant_values=pad_values
        )
    elif h > w:
        pad_ = h - w
        img = np.pad(
            img,
            ((0, 0), (pad_ // 2, pad_ - pad_ // 2), (0, 0)),
            constant_values=pad_values
        )
    return Image.fromarray(img)


def main(args):
    ref_img_list = get_img_path_list(args.ref_img, args.ref_dir)

    device = args.device
    dtype = args.dtype
    vae = AutoencoderKL.from_pretrained(args.vae_model_path).to(device, dtype=dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.image_encoder_model_path).to(device, dtype=dtype)

    unet = UNet2DConditionModel.from_pretrained(args.unet_model_path).to(device, dtype=dtype)
    controlnet = ControlNetXSModel.from_pretrained(args.controlnet_model_path).to(device, dtype=dtype)
    scheduler = KDPMPP2MDiscreteScheduler.from_pretrained(args.scheduler_path)
    pipe = VIPIVControlNetXSPipeline(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=CLIPImageProcessor(),
        image_encoder=image_encoder,
        controlnet=controlnet,
    ).to(device)

    pose_infer = HumanPose(
        det_cfg=join(POSE_CONFIG_DIR, "rtmdet_l_8xb32-300e_coco.py"),
        det_pth=join(args.weight_dir, "extensions/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"),
        bodypose_cfg=join(POSE_CONFIG_DIR, "rtmpose-l_8xb256-420e_body8-256x192.py"),
        bodypose_pth=join(args.weight_dir,
                          "extensions/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"),
        wholebodypose_cfg=join(POSE_CONFIG_DIR, "dwpose_l_wholebody_384x288.py"),
        wholebodypose_pth=join(args.weight_dir, "extensions/dw-ll_ucoco_384.pth"),
        device=device,
        bbox_thr=0.2,
    )

    for i, file in enumerate(ref_img_list):
        ref_img = load_image(file)
        ref_img = pad_image(ref_img)
        print(f"{i + 1}: {file}")
        seed = get_fixed_seed(-1)
        generator = get_torch_generator(seed, device=device)

        pose_img, _ = pose_infer(ref_img)
        pose_img = Image.fromarray(pose_img)

        out = pipe(
            image=ref_img,
            control_image=pose_img,
            height=args.resolution[0],
            width=args.resolution[1],
            num_inference_steps=25,
            guidance_scale=2.0,
            num_images_per_prompt=1,
            generator=generator,
            cond_scale=1.0,
        )[0]
        out_img = np.concatenate(
            [ref_img.resize(args.resolution[::-1], 1),
             pose_img.resize(args.resolution[::-1], 1),
             out], axis=1)
        Image.fromarray(out_img).save(join(args.out_dir, basename(file)))

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
