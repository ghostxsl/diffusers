# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, basename
import numpy as np
import cv2
from PIL import Image
import torch

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.utils.vip_utils import *
from diffusers.data.vos_client import VOSClient
from diffusers.data.utils import *
from diffusers.data.outer_vos_tools import download_pil_image


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--vos_pkl",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--out_dir",
        default="output",
        type=str,
        help="Directory to save.")
    parser.add_argument(
        "--resolution",
        type=str,
        default="512",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )

    parser.add_argument(
        "--rank",
        default=None,
        type=int)
    parser.add_argument(
        "--num_ranks",
        default=None,
        type=int)

    parser.add_argument(
        "--device",
        default='cuda',
        type=str,
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    parser.add_argument(
        "--dtype",
        default='bf16',
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
    elif args.dtype == 'bf16':
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float32

    return args


def pad_image(img, pad_values=(255, 255, 255), size=None, padding_down=False):
    w, h = img.size
    img = np.array(img)
    pad_border = [[0, 0], [0, 0], [0, 0]]
    if size is not None:
        # padding成指定size
        assert isinstance(size, (list, tuple)) and len(size) == 2
        oh, ow = size

        if oh > h:
            pad_ = oh - h
            pad_border[0] = [pad_ // 2, pad_ - pad_ // 2] if padding_down else [pad_, 0]
        if ow > w:
            pad_ = ow - w
            pad_border[1] = [pad_ // 2, pad_ - pad_ // 2]

        if pad_values == 'border':
            img = cv2.copyMakeBorder(
                img,
                pad_border[0][0], pad_border[0][1],
                pad_border[1][0], pad_border[1][1],
                cv2.BORDER_REPLICATE)
        else:
            img = cv2.copyMakeBorder(
                img,
                pad_border[0][0], pad_border[0][1],
                pad_border[1][0], pad_border[1][1],
                cv2.BORDER_CONSTANT, value=pad_values)
    else:
        # padding成1:1方图
        if w > h:
            pad_ = w - h
            if padding_down:
                pad_border = ((pad_ // 2, pad_ - pad_ // 2), (0, 0), (0, 0))
            else:
                pad_border = ((pad_, 0), (0, 0), (0, 0))
        elif h > w:
            pad_ = h - w
            pad_border = ((0, 0), (pad_ // 2, pad_ - pad_ // 2), (0, 0))

        if pad_values == 'border':
            img = cv2.copyMakeBorder(
                img,
                pad_border[0][0], pad_border[0][1],
                pad_border[1][0], pad_border[1][1],
                cv2.BORDER_REPLICATE)
        else:
            img = cv2.copyMakeBorder(
                img,
                pad_border[0][0], pad_border[0][1],
                pad_border[1][0], pad_border[1][1],
                cv2.BORDER_CONSTANT, value=pad_values)

    return Image.fromarray(img), pad_border


def unpad_image(img, pad_border=None):
    if pad_border is not None:
        img = np.array(img)
        if sum(pad_border[0]) != 0:
            y1, y2 = pad_border[0]
            y2 = -y2 if y2 != 0 else img.shape[0]
            img = img[y1: y2]
        if sum(pad_border[1]) != 0:
            x1, x2 = pad_border[1]
            x2 = -x2 if x2 != 0 else img.shape[1]
            img = img[:, x1: x2]
        img = Image.fromarray(img)

    return img


def draw_pose(pose, size=512, kpt_thr=0.3, canvas=None, draw_hand=True):
    w, h = pose['width'], pose['height']
    w = int(size / h * w) if size != h else w
    h = size
    if canvas is None:
        canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    else:
        canvas = np.array(canvas)

    size_ = np.array([w, h])

    kpts = pose['body']['keypoints'][..., :2] * size_
    kpt_valid = pose['body']['keypoints'][..., 2] > kpt_thr
    canvas = draw_bodypose(canvas, kpts, kpt_valid)

    if draw_hand:
        kpts = pose['hand']['keypoints'][..., :2] * size_
        kpt_valid = pose['hand']['keypoints'][..., 2] > kpt_thr
        canvas = draw_handpose(canvas, kpts, kpt_valid)

    return Image.fromarray(canvas)


def read_image(args, img):
    if args.vos_pkl is not None:
        if img.startswith('http'):
            img = img.replace('https://a.vpimg2.com/', 'http://a-appsimg.vip.vip.com/')
            img = download_pil_image(img)
        else:
            img = args.vos_client.download_vos_pil(img)
    return load_image(img)


def main(args):
    device = args.device
    dtype = args.dtype
    args.vos_client = VOSClient()

    img_dict = load_file(args.vos_pkl)

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks

        img_keys = sorted(list(img_dict.keys()))
        img_keys = img_keys[args.rank::args.num_ranks]
        img_dict = {k: img_dict[k] for k in img_keys}

    controlnet = ControlNetModel.from_pretrained(
        "/apps/dat/file/llm/model/controlnet-openpose-sdxl-1.0").to(device, dtype=dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "/apps/dat/file/llm/model/IP-Adapter/models/image_encoder").to(device, dtype=dtype)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "/apps/dat/file/llm/model/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        image_encoder=image_encoder,
        feature_extractor=CLIPImageProcessor(),
        safety_checker=None,
    ).to(device, dtype=dtype)
    pipe.load_ip_adapter("/apps/dat/file/llm/model/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")

    for i, (k, v) in enumerate(img_dict.items()):
        ref_item, gt_item = v[0], v[1]
        ref_img = read_image(args, ref_item['image'])
        gt_img = read_image(args, gt_item['image'])
        gt_pose = args.vos_client.download_vos_pkl(gt_item['pose'])
        pose_img = draw_pose(gt_pose, 512)

        print(f"{i + 1}/{len(img_dict)}: {k}")
        seed = get_fixed_seed(-1)
        generator = get_torch_generator(seed)

        input_img, pad_border = pad_image(ref_img)
        pose_img = pad_image(pose_img, pad_values=(0, 0, 0))[0]
        out = pipe(
            prompt="",
            ip_adapter_image=input_img,
            image=pose_img,
            height=args.resolution[0],
            width=args.resolution[0],  # for deepfashion
            num_inference_steps=25,
            num_images_per_prompt=1,
            guidance_scale=6.0,
            generator=generator,
            return_dict=False,
        )[0][0]
        out = unpad_image(out, pad_border=pad_border)
        out_img = np.concatenate(
            [ref_img.resize(args.resolution[::-1], 1),
             gt_img.resize(args.resolution[::-1], 1), out], axis=1)
        Image.fromarray(out_img).save(join(args.out_dir, k + '.png'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
