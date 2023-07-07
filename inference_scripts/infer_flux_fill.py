# Copyright (c) wilson.xu. All rights reserved.
import argparse
import random
import os
from os.path import join, splitext, basename
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import Resize, InterpolationMode

from diffusers import AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux_fill import FluxFillPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.vip_utils import *
from diffusers.data.vos_client import VOSClient
from diffusers.data.utils import *
from diffusers.data.outer_vos_tools import download_pil_image

from aistudio.utils.loader import ROOT_DIR
from aistudio.extensions.HumanPose import HumanPose
from aistudio.extensions.HumanPose.utils import POSE_CONFIG_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--vos_pkl",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--out_dir",
        default="output_fill",
        type=str,
        help="Directory to save.")

    parser.add_argument(
        "--resolution",
        type=str,
        default="1024",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )

    # Model Path
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/apps/dat/cv/wty/models/flux/FLUX.1-Fill-dev",
    )
    parser.add_argument(
        "--transformer_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default=None,
    )

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

    args.vos_client = VOSClient()
    args.resize = Resize(args.resolution[0], interpolation=InterpolationMode.LANCZOS, max_size=1440)

    return args


def read_image(args, img):
    if args.vos_pkl is not None:
        if img.startswith('http'):
            img = img.replace('https://a.vpimg2.com/', 'http://a-appsimg.vip.vip.com/')
            img = download_pil_image(img)
        else:
            img = args.vos_client.download_vos_pil(img)
    return load_image(img)


def scale_image(img, scale_factor=16):
    w, h = img.size
    w = int(w // scale_factor * scale_factor)
    h = int(h // scale_factor * scale_factor)
    img = np.array(img)[:h, :w]
    return Image.fromarray(img)


def main(args):
    device = args.device
    dtype = args.dtype

    img_list = load_file(args.vos_pkl)

    weight_dir = join(ROOT_DIR, "weights")
    pose_infer = HumanPose(
        det_cfg=join(POSE_CONFIG_DIR, "rtmdet_l_8xb32-300e_coco.py"),
        det_pth=join(weight_dir, "extensions/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"),
        bodypose_cfg=join(POSE_CONFIG_DIR, "rtmpose-l_8xb256-420e_body8-256x192.py"),
        bodypose_pth=join(weight_dir,
                          "extensions/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"),
        wholebodypose_cfg=join(POSE_CONFIG_DIR, "dwpose_l_wholebody_384x288.py"),
        wholebodypose_pth=join(weight_dir, "extensions/dw-ll_ucoco_384.pth"),
        device=device,
        bbox_thr=0.2,
    )

    # Load vae and image_encoder
    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae").to(device, dtype=dtype)

    # Load transformer and scheduler
    if args.transformer_model_path:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.transformer_model_path).to(device, dtype=dtype)
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.base_model_path, subfolder="transformer").to(device, dtype=dtype)

    if args.lora_model_path is not None:
        transformer.load_state_dict(
            torch.load(join(args.lora_model_path, "image_variation.bin"), map_location='cpu'), strict=False)
        transformer.load_lora_adapter(join(args.lora_model_path, "pytorch_lora_weights.safetensors"))
        transformer.fuse_lora(lora_scale=1.0)
        transformer.unload_lora()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")

    pipe = FluxFillPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        transformer=transformer,
    )

    # empty prompt embedding
    prompt_embeds = torch.zeros([1, 512, 4096]).to(device, dtype=dtype)
    pooled_prompt_embeds = torch.zeros([1, 768]).to(device, dtype=dtype)

    for i, line in enumerate(img_list):
        try:
            url = line[0]
            save_name = splitext(basename(url))[0]
            print(f"{i + 1}/{len(img_list)}: {save_name}")

            src_img = read_image(args, url)
            src_img = args.resize(src_img)
            src_img = scale_image(src_img)

            bbox = np.int32(pose_infer.detector(src_img)[0])[0]
            x1, y1, x2, y2 = bbox
            mask_image = np.zeros([src_img.height, src_img.width], dtype=np.uint8)
            mask_image[y1: y2, x1: x2] = 255
            mask_image = 255 - mask_image

            for j in range(1):
                out_img = pipe(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    image=src_img,
                    mask_image=mask_image,
                    height=src_img.height,
                    width=src_img.width,
                    num_inference_steps=30,
                    guidance_scale=30,
                    num_images_per_prompt=1,
                    generator=torch.Generator("cpu").manual_seed(42),
                    max_sequence_length=512,
                    return_dict=False
                )[0][0]
                out_img = np.concatenate([src_img, out_img], axis=1)
                Image.fromarray(out_img).save(join(args.out_dir, save_name + f'_{j}.jpg'))
        except Exception as e:
            print(e)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
