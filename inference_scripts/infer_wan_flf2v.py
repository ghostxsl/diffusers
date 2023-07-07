# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, basename
import numpy as np
from PIL import Image
import torch

from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, export_to_gif


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--img_dir",
        default="/apps/dat/cv/xsl/flf2v_test_img",
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--out_dir",
        default="output_lora",
        type=str)

    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/Wan-FLF2V-14B-720P-dif",
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default="/apps/dat/cv/xsl/exp_animate/wan2.1_flf2v_lora/pytorch_lora_weights.safetensors",
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

    if not exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.device = torch.device(args.device)
    if args.dtype == 'fp16':
        args.dtype = torch.float16
    elif args.dtype == 'bf16':
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float32

    return args


def get_aspect_ratio_size(size, max_size=(720, 960), mod_value=16):
    w, h = size
    max_area = max_size[0] * max_size[1]
    aspect_ratio = h / w
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    return (width, height)


def main(args):
    device = args.device
    dtype = args.dtype

    img_list = sorted(os.listdir(args.img_dir))
    img_dict = {}
    for name in img_list:
        k = int(name.split('_')[0])
        if k not in img_dict:
            img_dict[k] = [join(args.img_dir, name)]
        else:
            img_dict[k].append(join(args.img_dir, name))
    img_dict = {k: img_dict[k] for k in sorted(list(img_dict.keys()))}

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

        first_frame = Image.open(v[0])
        size = get_aspect_ratio_size(first_frame.size)

        first_frame = first_frame.resize(size, 1)
        last_frame = Image.open(v[1]).resize(size, 1)

        # prompt = "A model is showing off the clothes they are wearing."
        prompt = splitext(basename(v[0]))[0].split('_')[-1]
        if len(prompt) == 1:
            prompt = "一位年轻女性注视着镜头，缓慢运动"
        print(prompt)

        output = pipe(
            image=first_frame,
            last_image=last_frame,
            prompt=prompt,
            height=size[1],
            width=size[0],
            guidance_scale=5.0,
            num_frames=48 + 1,
            num_inference_steps=30,
        ).frames[0]

        export_to_video(output, join(args.out_dir, f"{k}.mp4"), fps=16)
        # out_pils = [Image.fromarray(a) for a in np.uint8(output * 255)]
        # export_to_gif(out_pils, join(args.out_dir, f"{k}.gif"), fps=16)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
