# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, basename
import numpy as np
from PIL import Image
import torch

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.vip_pipeline_controlnet_inpaint import VIPStableDiffusionControlNetInpaintPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler, KDPMPP2MDiscreteScheduler
from diffusers.utils.vip_utils import *
from transformers import AutoTokenizer, CLIPTextModel, CLIPImageProcessor, CLIPVisionModelWithProjection


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
        "--base_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/stable-diffusion-v1-5",
    )
    parser.add_argument(
        "--ip_adapter_model_path",
        type=str,
        default="/apps/dat/cv/xsl/exp_animate/checkpoint-85000/ip_adapter/ip_adapter_plus.bin",
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
        "--controlnet_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/control_v11p_sd15_openpose",
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

    unet = UNet2DConditionModel.from_pretrained(args.base_model_path, subfolder="unet").to(device, dtype=dtype)
    unet._init_ip_adapter_plus(state_dict=torch.load(args.ip_adapter_model_path, map_location=torch.device("cpu")))
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    # import correct text encoder class
    text_encoder = CLIPTextModel.from_pretrained(args.base_model_path, subfolder="text_encoder").to(device, dtype=dtype)
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_path).to(device, dtype=dtype)
    pipe = VIPStableDiffusionControlNetInpaintPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        controlnet=controlnet,
        feature_extractor=CLIPImageProcessor(),
        image_encoder=image_encoder,
    ).to(device)

    for i, file in enumerate(ref_img_list):
        ref_img = load_image(file)
        ref_img = pad_image(ref_img)
        print(f"{i + 1}/{len(ref_img_list)}: {file}")
        seed = get_fixed_seed(-1)
        generator = get_torch_generator(seed, device=device)
        mask_image = Image.new("RGB", args.resolution[::-1], color=(255, 255, 255))

        out, image_overlay = pipe(
            prompt="",
            negative_prompt="",
            image=ref_img,
            mask_image=mask_image,
            control_image=mask_image,
            ip_adapter_image=ref_img,
            height=args.resolution[0],
            width=args.resolution[1],
            strength=1.0,
            num_inference_steps=25,
            guidance_scale=3.0,
            num_images_per_prompt=1,
            generator=generator,
            controlnet_conditioning_scale=0.0,
            masked_content="noise",
        )
        out = alpha_composite(out, image_overlay)[0]
        out_img = np.concatenate(
            [ref_img.resize(args.resolution[::-1], 1), out], axis=1)
        Image.fromarray(out_img).save(join(args.out_dir, basename(file)))

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
