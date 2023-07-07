# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, basename
import numpy as np
from PIL import Image
import torch

from diffusers.pipelines import FluxKontextPipeline
from diffusers.utils import export_to_video, export_to_gif
from diffusers.data.utils import load_file
from diffusers.data.outer_vos_tools import download_pil_image
from diffusers.utils.vip_utils import load_image


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--img_path",
        default=None,
        type=str)
    parser.add_argument(
        "--out_img",
        default="flux_kontext.png",
        type=str)

    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/apps/dat/file/llm/model/FLUX.1-Kontext-dev",
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


def rm_image_border(image, upper=240, lower=15):
    image = np.array(image)
    # 转换成灰度图
    mean_img = np.mean(image, axis=-1)
    # 裁剪白边
    x = np.where(np.mean(mean_img, axis=0) < upper)[0]
    y = np.where(np.mean(mean_img, axis=1) < upper)[0]
    if len(x) > 1 and len(y) > 1:
        x1, y1, x2, y2 = int(x[0]), int(y[0]), int(x[-1]) + 1, int(y[-1]) + 1
    else:
        raise Exception("The whole picture is white, check the input image.")
    mean_img = mean_img[y1:y2, x1:x2]
    image = image[y1:y2, x1:x2]

    # 裁剪黑边
    x = np.where(np.mean(mean_img, axis=0) > lower)[0]
    y = np.where(np.mean(mean_img, axis=1) > lower)[0]
    if len(x) > 1 and len(y) > 1:
        x1, y1, x2, y2 = int(x[0]), int(y[0]), int(x[-1]) + 1, int(y[-1]) + 1
    else:
        raise Exception("The whole picture is black, check the input image.")
    image = image[y1:y2, x1:x2]

    return Image.fromarray(image)


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

    pipe = FluxKontextPipeline.from_pretrained(
        args.base_model_path, torch_dtype=dtype).to(device)

    input_image = load_image(args.img_path)
    size = get_aspect_ratio_size(input_image.size)
    prompt = ""

    output = pipe(
        image=input_image,
        prompt=prompt,
        height=size[1],
        width=size[0],
        guidance_scale=2.5,
        num_inference_steps=30,
    ).images[0]

    output = output.resize(input_image.size, resample=1)
    output.save(args.out_img)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
