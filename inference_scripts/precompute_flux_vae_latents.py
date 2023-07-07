# Copyright (c) wilson.xu. All rights reserved.
import argparse
import os
from os.path import join, splitext, basename, exists
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.data.utils import load_file, json_save
from diffusers.utils.vip_utils import load_image
from diffusers.data.vos_client import VOSClient
from diffusers.data.outer_vos_tools import download_pil_image


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--img_dir",
        default=None,
        type=str,
        help="Directory to image.")
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
        "--vae_flux_model_path",
        type=str,
        default="/apps/dat/cv/wty/models/flux/FLUX.1-Fill-dev/vae",
    )

    parser.add_argument(
        "--resolution",
        type=str,
        default="1024",
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

    if args.vos_pkl is None:
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


@torch.no_grad()
def flux_encode_vae_image(vae, image, generator=None):
    image_latents = vae.encode(image).latent_dist.sample(generator=generator)

    image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor

    return image_latents


def main(args):
    device = torch.device(args.device)
    dtype = args.dtype

    image_processor = VaeImageProcessor()
    vae = AutoencoderKL.from_pretrained(args.vae_flux_model_path).to(device, dtype=dtype)

    if args.vos_pkl is not None:
        img_list = load_file(args.vos_pkl)
        args.vos_client = VOSClient()
    else:
        assert exists(args.img_dir)
        img_list = os.listdir(args.img_dir)
        img_list = [join(args.img_dir, a) for a in img_list if splitext(a)[
            -1].lower() in ['.png', '.jpg', '.jpeg']]
        img_list = sorted(img_list)

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        total_num = len(img_list)
        stride = int(total_num / args.num_ranks)

        start_idx = stride * args.rank
        end_idx = stride * (args.rank + 1) if args.rank + 1 < args.num_ranks else len(img_list)
        img_list = img_list[start_idx: end_idx]

    damaged = []
    for name in tqdm(img_list):
        try:
            image = read_image(args, name)
            image = pad_image(image)
            image = image.resize(args.resolution[::-1], 1)

            init_image = image_processor.preprocess(
                image, height=args.resolution[0], width=args.resolution[1])

            # flux
            flux_latents = flux_encode_vae_image(
                vae,
                init_image.to(vae.device, dtype=vae.dtype))
            flux_latents = flux_latents.cpu()

            save_name = splitext(basename(name))[0] + '.pt'
            if args.vos_pkl is not None:
                args.vos_client.upload_vos_pt(
                    flux_latents, join(args.out_dir, save_name)
                )
            else:
                torch.save(flux_latents, join(args.out_dir, save_name))
        except:
            damaged.append(name)

    if args.rank is not None:
        json_save(damaged, f"./damaged_{args.rank}.json")
    else:
        json_save(damaged, "./damaged.json")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
