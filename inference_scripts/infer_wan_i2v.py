# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, basename
import numpy as np
from PIL import Image
import torch

from diffusers import WanImageToVideoPipeline
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video, export_to_gif
from diffusers.data.utils import load_file
from diffusers.data.outer_vos_tools import download_pil_image
from diffusers.utils.vip_utils import load_image


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--img_dir",
        default=None,
        type=str)
    parser.add_argument(
        "--input_file",
        default="/apps/dat/cv/xsl/test_zjn_i2v_caption.csv",
        type=str)
    parser.add_argument(
        "--out_dir",
        default="output",
        type=str)

    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/apps/dat/file/llm/model/Wan2.1-I2V-14B-720P-Diffusers",
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default="/apps/dat/cv/xsl/exp_animate/checkpoint-3000/pytorch_lora_weights.safetensors",
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

    img_dict = {}
    if args.img_dir:
        img_list = sorted(os.listdir(args.img_dir))
        for name in img_list:
            k = int(name.split('_')[0])
            if k not in img_dict:
                img_dict[k] = [join(args.img_dir, name)]
            else:
                img_dict[k].append(join(args.img_dir, name))
        img_dict = {k: img_dict[k] for k in sorted(list(img_dict.keys()))}
    elif args.input_file:
        img_list = load_file(args.input_file)
        for i, line in enumerate(img_list):
            img_dict[i] = line
    else:
        raise Exception(f"`img_dir` and `input_file` are both None.")

    pipe = WanImageToVideoPipeline.from_pretrained(
        args.base_model_path, torch_dtype=dtype).to(device)
    flow_shift = 4.5  # 5.0 for 720P, 3.0 for 480P
    scheduler = UniPCMultistepScheduler(
        prediction_type='flow_prediction',
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=flow_shift)
    pipe.scheduler = scheduler

    if args.lora_model_path:
        pipe.load_lora_weights(args.lora_model_path)
        pipe.fuse_lora(lora_scale=1.0)
        pipe.unload_lora_weights()

    prompt = "一位年轻女性注视着镜头，缓慢运动"
    for i, (k, v) in enumerate(img_dict.items()):
        print(f"{i + 1}/{len(img_dict)}: {k}")

        if args.img_dir:
            first_frame = Image.open(v[0])
            if len(splitext(basename(v[0]))[0].split('_')[-1]) > 1:
                prompt = splitext(basename(v[0]))[0].split('_')[-1]
        else:
            first_frame = read_image(args, v[0])
            prompt = v[1]

        first_frame = rm_image_border(first_frame)
        size = get_aspect_ratio_size(first_frame.size)
        print(prompt)

        output = pipe(
            image=first_frame,
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
