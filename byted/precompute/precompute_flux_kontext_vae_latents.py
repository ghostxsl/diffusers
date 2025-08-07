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
from diffusers.data.utils import load_file
from diffusers.utils.vip_utils import load_image


PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


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
        "--vae_flux_model_path",
        type=str,
        default="/mnt/bn/ttcc-algo-bytenas/zjn/models/FLUX.1-Kontext-dev/vae",
    )

    parser.add_argument(
        "--resolution",
        type=str,
        default="1024x1024",
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


def prepare_latent_image_ids(height, width, device=torch.device('cpu'), dtype=torch.float32):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


def get_aspect_ratio_size(size, max_size=(1024, 1024), mod_value=16):
    w, h = size
    max_area = max_size[0] * max_size[1]
    aspect_ratio = h / w
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    return (width, height)


@torch.no_grad()
def flux_encode_vae_image(vae, image):
    image_latents = vae.encode(image).latent_dist.mode()

    image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor

    return image_latents


def main(args):
    device = torch.device(args.device)
    dtype = args.dtype

    image_processor = VaeImageProcessor()
    vae = AutoencoderKL.from_pretrained(args.vae_flux_model_path).to(device, dtype=dtype)

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
            image = load_image(item['poster'])
            size = get_aspect_ratio_size(image.size, max_size=args.resolution)
            image = image.resize(size, 1)
            init_image = image_processor.preprocess(image, width=size[0], height=size[1])
            # 1.1 flux image vae encode
            init_latents = flux_encode_vae_image(vae, init_image.to(vae.device, dtype=vae.dtype))
            init_latents = init_latents.cpu()
            # 1.2 latents image ids
            latent_ids = prepare_latent_image_ids(
                init_latents.shape[2] // 2, init_latents.shape[3] // 2)

            # 2. load reference image
            cond_image = load_image(item['bg'])
            aspect_ratio = cond_image.width / cond_image.height
            _, image_width, image_height = min(
                (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
            )
            cond_image = cond_image.resize((image_width, image_height), 1)
            cond_image = image_processor.preprocess(cond_image, width=image_width, height=image_height)
            # 2.1 reference image vae encode
            cond_latents = flux_encode_vae_image(vae, cond_image.to(vae.device, dtype=vae.dtype))
            cond_latents = cond_latents.cpu()
            # 2.2 reference image ids
            cond_ids = prepare_latent_image_ids(
                cond_latents.shape[2] // 2, cond_latents.shape[3] // 2)
            cond_ids[:, 0] = 1

            save_name = basename(item['poster']).split('_')[0] + '.latents'
            save_latents = {
                "image_latents": init_latents,
                "cond_image_latents": cond_latents,
                "latent_image_ids": latent_ids,
                "cond_image_ids": cond_ids,
            }
            torch.save(save_latents, join(args.save_dir, save_name))
        except Exception as e:
            print(e)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
