# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, basename
import random
import numpy as np
from PIL import Image
import torch
import pandas as pd

from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from diffusers.utils.vip_utils import *
from diffusers.data.vos_client import VOSClient
from diffusers.data.utils import *
from diffusers.data.outer_vos_tools import download_pil_image, upload_pil_image

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--img_dir",
        default="/home/llms/flf2v_test_img",
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--out_dir",
        default="output",
        type=str)

    parser.add_argument(
        "--resolution",
        type=str,
        default="720",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )

    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/home/llms/Wan-FLF2V-14B-720P-dif",
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default="/apps/dat/cv/xsl/exp_animate/wan2.1_flf2v/pytorch_lora_weights.safetensors",
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
        "--weight_dir",
        default=None,
        type=str,
        help="Directory to weights.")
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

    if not exists(args.out_dir):
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


def long_side_resize(image, size=720):
    w, h = image.size
    max_l = max(w, h)
    ratio = size / max_l
    new_w = size if w == max_l else int(w * ratio)
    new_h = size if h == max_l else int(h * ratio)
    return image.resize((new_w, new_h), 1)


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


def read_image(vos_client, img):
    if img.startswith('http'):
        img = img.replace('https://a.vpimg2.com/', 'http://a-appsimg.vip.vip.com/')
        img = download_pil_image(img)
    else:
        img = vos_client.download_vos_pil(img)
    return load_image(img)


def write_to_csv(results, dst):
    df = pd.DataFrame(results)
    df.to_csv(dst, mode='a', index=False, header=not os.path.exists(dst))


def main(args):
    device = args.device
    dtype = args.dtype
    args.vos_client = VOSClient()

    img_list = sorted(os.listdir(args.img_dir))
    img_dict = {}
    for name in img_list:
        k = name.split('_')[0]
        if k not in img_dict:
            img_dict[k] = [join(args.img_dir, name)]
        else:
            img_dict[k].append(join(args.img_dir, name))

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks

        img_keys = sorted(list(img_dict.keys()))
        img_keys = img_keys[args.rank::args.num_ranks]
        img_dict = {k: img_dict[k] for k in img_keys}

    pipe = WanImageToVideoPipeline.from_pretrained(
        args.base_model_path, torch_dtype=dtype).to(device)
    if args.lora_model_path:
        pipe.load_lora_weights(args.lora_model_path)
        pipe.fuse_lora(lora_scale=1.0)
        pipe.unload_lora_weights()

    for i, (k, v) in enumerate(img_dict.items()):
        print(f"{i + 1}/{len(img_dict)}: {k}")

        first_frame = load_image(v[0])
        first_frame = pad_image(long_side_resize(first_frame, size=args.resolution[0]))
        last_frame = load_image(v[1])
        last_frame = pad_image(long_side_resize(last_frame, size=args.resolution[1]))

        # prompt = "A model is showing off the clothes they are wearing."
        prompt = "一位模特注视着镜头，缓缓转身"

        output = pipe(
            image=first_frame,
            last_image=last_frame,
            prompt=prompt,
            height=args.resolution[0],
            width=args.resolution[1],
            guidance_scale=5.5,
            num_frames=80 + 1,
            num_inference_steps=30,
        ).frames[0]
        save_name = join(args.out_dir, f"{k}.mp4")
        export_to_video(output, save_name, fps=16)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
