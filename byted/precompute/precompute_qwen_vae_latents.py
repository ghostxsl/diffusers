# Copyright (c) wilson.xu. All rights reserved.
import argparse
import os
from os.path import join, splitext, basename, exists
import torch
import numpy as np
from tqdm import tqdm

from diffusers.models import AutoencoderKLQwenImage
from diffusers.image_processor import VaeImageProcessor
from diffusers.data.utils import load_file
from diffusers.data.outer_vos_tools import load_or_download_image


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--save_dir",
        default="output",
        type=str,
        help="Directory to save.")
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="/mnt/bn/ttcc-algo-bytenas/xsl/Qwen-Image/vae",
    )

    parser.add_argument(
        "--resolution",
        type=str,
        default="1280x720",
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

    os.makedirs(args.save_dir, exist_ok=True)

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


def get_aspect_ratio_size(size, max_size=(1024, 1024), mod_value=16):
    w, h = size
    max_area = max_size[0] * max_size[1]
    aspect_ratio = h / w
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    return (width, height)


@torch.no_grad()
def qwen_encode_vae_image(vae, image, latents_mean, latents_std, sample_mode="sample"):
    if sample_mode == "sample":
        image_latents = vae.encode(image).latent_dist.sample()
    elif sample_mode == "argmax":
        image_latents = vae.encode(image).latent_dist.mode()
    else:
        raise Exception("Could not access latents of provided `sample_mode`")

    image_latents = (image_latents - latents_mean) * latents_std

    return image_latents


def main(args):
    device = torch.device(args.device)
    dtype = args.dtype

    image_processor = VaeImageProcessor()
    vae = AutoencoderKLQwenImage.from_pretrained(args.vae_model_path).to(device, dtype=dtype)
    vae.requires_grad_(False)
    latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)).to(device, dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)

    # load data list
    img_list = load_file(args.input_file)
    if isinstance(img_list, dict):
        img_list = [v for v in img_list.values()]

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks

        img_list = img_list[args.rank::args.num_ranks]

    for item in tqdm(img_list):
        try:
            # 1. load poster image
            poster_url = item[-4]
            image = load_or_download_image(poster_url)
            size = get_aspect_ratio_size(image.size, max_size=args.resolution)
            image = image.resize(size, 1)
            init_image = image_processor.preprocess(image, width=size[0], height=size[1])
            # 1.1 flux image vae encode
            init_image = init_image.unsqueeze(2).to(vae.device, dtype=vae.dtype)
            init_latents = qwen_encode_vae_image(vae, init_image, latents_mean, latents_std)
            init_latents = init_latents.squeeze(2).cpu()

            # 2. load reference image
            cond_image = load_or_download_image(item[-1])
            cond_image = image_processor.preprocess(cond_image, width=cond_image.width, height=cond_image.height)
            # 2.1 reference image vae encode
            cond_image = cond_image.unsqueeze(2).to(vae.device, dtype=vae.dtype)
            cond_latents = qwen_encode_vae_image(vae, cond_image, latents_mean, latents_std)
            cond_latents = cond_latents.squeeze(2).cpu()

            save_name = basename(poster_url) + '.latents'
            save_latents = {
                "image_latents": init_latents,
                "cond_image_latents": cond_latents,
            }
            torch.save(save_latents, join(args.save_dir, save_name))
        except Exception as e:
            print(e)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
