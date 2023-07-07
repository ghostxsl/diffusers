# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, basename
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch

from diffusers import AutoencoderKL
from diffusers import AutoencoderDC
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.vip_utils import *
from diffusers.data.vos_client import VOSClient
from diffusers.data.utils import *
from aistudio.data.outer_vos_tools import download_pil_image


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--img_dir",
        default=None,
        type=str,
        help="Directory to reference image.")
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
        "--vae_1_5_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/vae_ft_mse",
    )
    parser.add_argument(
        "--vae_xl_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/Kolors-diffusers/vae",
    )
    parser.add_argument(
        "--vae_flux_model_path",
        type=str,
        default="/apps/dat/cv/wty/models/flux/FLUX.1-dev/vae",
    )
    parser.add_argument(
        "--vae_sana_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/dc-ae-f32c32-sana-1.0",
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


def sd_1_5_xl_encode_vae_image(vae, image, generator=None):
    image_latents = vae.encode(image).latent_dist.sample(generator=generator)

    image_latents = vae.config.scaling_factor * image_latents

    return image_latents


def flux_encode_vae_image(vae, image, generator=None):
    image_latents = vae.encode(image).latent_dist.sample(generator=generator)

    image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor

    return image_latents


@torch.no_grad()
def main(args):
    device = args.device
    dtype = args.dtype

    if args.vos_pkl is not None:
        img_list = load_file(args.vos_pkl)
        args.vos_client = VOSClient()
    else:
        assert exists(args.img_dir)
        img_list = os.listdir(args.img_dir)
        img_list = [join(args.img_dir, a) for a in img_list if splitext(a)[
            -1].lower() in ['.png', '.jpg', '.jpeg']]
        img_list = sorted(img_list)

    vae_1_5 = AutoencoderKL.from_pretrained(
        args.vae_1_5_model_path).to(device, dtype=dtype)

    vae_xl = AutoencoderKL.from_pretrained(
        args.vae_xl_model_path, variant="fp16").to(device, dtype=torch.float32)

    vae_flux = AutoencoderKL.from_pretrained(
        args.vae_flux_model_path).to(device, dtype=dtype)

    vae_sana = AutoencoderDC.from_pretrained(
        args.vae_sana_model_path).to(device, dtype=torch.float32)

    image_processor = VaeImageProcessor()

    seed = get_fixed_seed(42)
    generator = get_torch_generator(seed)

    for file in tqdm(img_list):
        image = read_image(args, file)
        image = pad_image(image)
        image = image.resize(args.resolution[::-1], 1)

        init_image = image_processor.preprocess(
            image, height=args.resolution[0], width=args.resolution[1])

        # sd_1.5
        sd_1_5_latents = sd_1_5_xl_encode_vae_image(
            vae_1_5,
            init_image.to(vae_1_5.device, dtype=vae_1_5.dtype),
            generator)
        sd_1_5_image = vae_1_5.decode(sd_1_5_latents / vae_1_5.config.scaling_factor, return_dict=False)[0]
        sd_1_5_image = image_processor.postprocess(sd_1_5_image)[0]

        # sd_xl
        sd_xl_latents = sd_1_5_xl_encode_vae_image(
            vae_xl,
            init_image.to(vae_xl.device, dtype=vae_xl.dtype),
            generator)
        sd_xl_image = vae_xl.decode(sd_xl_latents / vae_xl.config.scaling_factor, return_dict=False)[0]
        sd_xl_image = image_processor.postprocess(sd_xl_image)[0]

        # flux
        flux_latents = flux_encode_vae_image(
            vae_flux,
            init_image.to(vae_flux.device, dtype=vae_flux.dtype),
            generator)
        flux_latents = (flux_latents / vae_flux.config.scaling_factor) + vae_flux.config.shift_factor
        flux_image = vae_flux.decode(flux_latents, return_dict=False)[0]
        flux_image = image_processor.postprocess(flux_image)[0]

        # sana
        sana_image = vae_sana.forward(
            init_image.to(vae_sana.device, dtype=vae_sana.dtype), return_dict=False)[0]
        sana_image = image_processor.postprocess(sana_image)[0]

        out_img = np.concatenate([image, sd_1_5_image, sd_xl_image, flux_image, sana_image], axis=1)
        Image.fromarray(out_img).save(join(args.out_dir, basename(file)))

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
