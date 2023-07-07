# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, basename
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
    CLIPTextModel,
    T5EncoderModel,
)

from diffusers import AutoencoderKL
from diffusers.models.vip.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.vip.flux_fill_xhs_ic_pipeline import FluxFillPipeline
from diffusers.pipelines.flux.pipeline_flux_img2img import FluxImg2ImgPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.vip_utils import *
from diffusers.data.vos_client import VOSClient
from diffusers.data.utils import *
from diffusers.data.outer_vos_tools import download_pil_image
from diffusers.utils.export_utils import export_to_gif


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
        default="1280x960",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )

    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/apps/dat/file/llm/model/FLUX.1-Fill-dev",
    )
    parser.add_argument(
        "--pt_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ty_base_model_path",
        type=str,
        default="/apps/dat/cv/wty/models/flux/FLUX.1-dev",
    )
    parser.add_argument(
        "--ty_lora_model_path",
        type=str,
        default="/apps/dat/cv/xsl/exp_animate/ty_lora_dif.safetensors",
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


def center_crop(img, size=(1280, 960)):
    _, image_height, image_width = F.get_dimensions(img)
    crop_height, crop_width = size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = F.pad(img, padding_ltrb, fill=255)  # PIL uses fill value 0
        _, image_height, image_width = F.get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return F.crop(img, crop_top, crop_left, crop_height, crop_width)


def resize_image(img, size=(1280, 960)):
    w, h = img.size
    oh = size[0]
    ow = int(oh / h * w) if oh != h else w
    img = img.resize((ow, oh), 1)
    img = center_crop(img, size)
    return img


def read_image(args, img):
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
    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae").to(device, dtype=torch.float32)

    # Load the tokenizers and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model_path, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(args.base_model_path, subfolder="tokenizer_2")

    text_encoder = CLIPTextModel.from_pretrained(
        args.base_model_path, subfolder="text_encoder").to(device, dtype=dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        args.base_model_path, subfolder="text_encoder_2").to(device, dtype=dtype)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Load transformer and scheduler
    transformer = FluxTransformer2DModel.from_pretrained(
        args.base_model_path, subfolder="transformer").to(device, dtype=dtype)
    if args.pt_model_path:
        transformer.load_state_dict(
            torch.load(join(args.pt_model_path, "image_variation.bin"), map_location='cpu'), strict=False)

        transformer.load_lora_adapter(join(args.pt_model_path, "pytorch_lora_weights.safetensors"))
        transformer.fuse_lora(lora_scale=1.0)
        transformer.unload_lora()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")

    pipe = FluxFillPipeline(
        scheduler=scheduler,
        vae=vae,
        transformer=transformer,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
    )

    # tianyi text_to_image
    pipe_ty = FluxImg2ImgPipeline.from_pretrained(
        args.ty_base_model_path,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        torch_dtype=dtype).to(device, dtype=dtype)
    pipe_ty.load_lora_weights(args.ty_lora_model_path)
    pipe_ty.fuse_lora(lora_scale=0.75)
    pipe_ty.unload_lora_weights()

    for i, (k, v) in enumerate(img_dict.items()):
        if len(v) < 2:
            continue

        item = random.choice(v)
        src_img = read_image(args, item['image'])
        src_img = resize_image(src_img, size=args.resolution)
        # 重绘参考图
        ref_img = pipe_ty(
            prompt=item['caption'],
            image=src_img,
            height=args.resolution[0],
            width=args.resolution[1],
            strength=0.75,
            num_inference_steps=30,
            guidance_scale=3.5,
            num_images_per_prompt=1,
            return_dict=False,
        )[0][0]

        # 生成图像
        v.remove(item)
        out_gt_imgs = [src_img]
        out_imgs = [ref_img]
        for gt_item in v:
            gt_img = read_image(args, gt_item['image'])
            gt_img = resize_image(gt_img, size=args.resolution)
            out_gt_imgs.append(gt_img.copy())

            print(f"{i + 1}/{len(img_dict)}: {k}")
            seed = get_fixed_seed(-1)
            generator = get_torch_generator(seed)

            out, pred_x0_list = pipe(
                prompt=gt_item["caption"],
                image=ref_img,
                tgt_image=gt_img,
                height=args.resolution[0],
                width=args.resolution[1],
                strength=0.9,
                num_inference_steps=30,
                guidance_vector=30.0,
                num_images_per_prompt=1,
                generator=generator,
                latent_guidance_step=7,
                v_scale=0.2,
                clip_mu=True,
                # return_gif=True,
            )
            # export_to_gif(pred_x0_list, "img1.gif", fps=8)
            out = out[0]
            out = np.split(np.array(out), 2, axis=1)[0]
            out_imgs.append(out)

        out_gt_imgs = np.concatenate(out_gt_imgs, axis=1)
        out_imgs = np.concatenate(out_imgs, axis=1)
        out_img = np.concatenate(
            [out_gt_imgs, out_imgs], axis=0)
        Image.fromarray(out_img).save(join(args.out_dir, k + '.png'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
