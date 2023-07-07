# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, basename
import numpy as np
from PIL import Image
import torch
from skimage.metrics import structural_similarity

from diffusers import AutoencoderKL
from diffusers.models.vip.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.vip.flux_image_variation_img2img import FluxImageVariationPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
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
        "--base_model_path",
        type=str,
        default="/apps/dat/cv/wty/models/flux/FLUX.1-dev",
    )
    parser.add_argument(
        "--transformer_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_variation_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default=None,
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

    # Load vae
    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae").to(device, dtype=dtype)

    # Load transformer and scheduler
    if args.transformer_model_path:
        # from pretrained
        transformer = FluxTransformer2DModel.from_pretrained(args.transformer_model_path).to(device, dtype=dtype)
        transformer.load_state_dict(
            torch.load(args.image_variation_model_path, map_location='cpu'), strict=False)
    else:
        # from dev
        transformer = FluxTransformer2DModel.from_pretrained(
            args.base_model_path, subfolder="transformer").to(device, dtype=dtype)
        transformer._init_image_variation(
            joint_attention_dim=transformer.config.in_channels,
            pooled_projection_dim=None,
            state_dict=torch.load(args.image_variation_model_path, map_location=torch.device("cpu")),
            alter_x_embedder=True,
        )

    # Load lora weights
    if args.lora_model_path is not None:
        transformer.load_lora_adapter(args.lora_model_path)
        transformer.fuse_lora(lora_scale=1.0)
        transformer.unload_lora()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")

    pipe = FluxImageVariationPipeline(
        scheduler=scheduler,
        vae=vae,
        transformer=transformer,
        structural_control=True,
    ).to(device)

    for i, (k, v) in enumerate(img_dict.items()):
        ref_item, gt_item = v[0], v[1]
        ref_img = read_image(args, ref_item['image'])
        gt_img = read_image(args, gt_item['image'])
        gt_img = gt_img.resize(args.resolution[::-1], 1)

        gt_pose = args.vos_client.download_vos_pkl(gt_item['pose'])
        pose_img = draw_pose(gt_pose, args.resolution[0])

        print(f"{i + 1}/{len(img_dict)}: {k}")

        out = None
        max_ssim = 0
        for _ in range(2):
            candidate = pipe(
                image=ref_img,
                control_image=pose_img,
                height=args.resolution[0],
                width=args.resolution[1],
                strength=1.0,
                num_inference_steps=30,
                guidance_scale=2.0,
                num_images_per_prompt=1,
            )[0]
            ssim = structural_similarity(
                np.array(gt_img), np.array(candidate),
                data_range=255,
                channel_axis=2,
                gaussian_weights=True,
                sigma=1.2,
                use_sample_covariance=False,
            )
            if ssim > max_ssim:
                max_ssim = ssim
                out = candidate

        out_img = np.concatenate(
            [ref_img.resize(args.resolution[::-1], 1), gt_img, out], axis=1)
        Image.fromarray(out_img).save(join(args.out_dir, k + '.png'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
