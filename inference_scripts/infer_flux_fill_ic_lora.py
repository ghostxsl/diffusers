# Copyright (c) wilson.xu. All rights reserved.
import argparse
import random
import os
from os.path import join, splitext, basename
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision.transforms import CenterCrop

from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
    CLIPTextModel,
    T5EncoderModel,
)
from diffusers import AutoencoderKL
from diffusers.models.vip.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.vip.pipeline_flux_fill_vip import FluxFillVIPPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.vip_utils import *
from diffusers.data.vos_client import VOSClient
from diffusers.data.utils import *
from diffusers.data.outer_vos_tools import download_pil_image


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--vos_pkl",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--out_dir",
        default="output_fill_ic",
        type=str,
        help="Directory to save.")

    parser.add_argument(
        "--resolution",
        type=str,
        default="896x672",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--gpt4v",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--llava",
        action="store_true",
        default=False,
    )

    # Model Path
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/apps/dat/cv/wty/models/flux/FLUX.1-Fill-dev",
    )
    parser.add_argument(
        "--transformer_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default=None,
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

    args.vos_client = VOSClient()
    args.centercrop = CenterCrop(args.resolution)

    return args


def pad_image(img, pad_values=(255, 255, 255), size=None):
    w, h = img.size
    img = np.array(img)
    pad_border = [[0, 0], [0, 0], [0, 0]]
    if size is not None:
        # padding成指定size
        assert isinstance(size, (list, tuple)) and len(size) == 2
        oh, ow = size

        if oh > h:
            pad_ = oh - h
            pad_border[0] = [pad_ // 2, pad_ - pad_ // 2]
        if ow > w:
            pad_ = ow - w
            pad_border[1] = [pad_ // 2, pad_ - pad_ // 2]

        if pad_values == 'border':
            img = cv2.copyMakeBorder(
                img,
                pad_border[0][0], pad_border[0][1],
                pad_border[1][0], pad_border[1][1],
                cv2.BORDER_REPLICATE)
        else:
            img = cv2.copyMakeBorder(
                img,
                pad_border[0][0], pad_border[0][1],
                pad_border[1][0], pad_border[1][1],
                cv2.BORDER_CONSTANT, value=pad_values)
    else:
        # padding成1:1方图
        if w > h:
            pad_ = w - h
            pad_border = ((pad_ // 2, pad_ - pad_ // 2), (0, 0), (0, 0))
        elif h > w:
            pad_ = h - w
            pad_border = ((0, 0), (pad_ // 2, pad_ - pad_ // 2), (0, 0))

        if pad_values == 'border':
            img = cv2.copyMakeBorder(
                img,
                pad_border[0][0], pad_border[0][1],
                pad_border[1][0], pad_border[1][1],
                cv2.BORDER_REPLICATE)
        else:
            img = cv2.copyMakeBorder(
                img,
                pad_border[0][0], pad_border[0][1],
                pad_border[1][0], pad_border[1][1],
                cv2.BORDER_CONSTANT, value=pad_values)

    return Image.fromarray(img), pad_border


def resize_image(args, img, size):
    w, h = img.size
    oh, ow = size
    ow = int(oh / h * w)
    img = img.resize((ow, oh), 1)

    if img.width > size[1]:
        img = args.centercrop(img)
    else:
        img, _ = pad_image(img, size=size)

    return img


def read_image(args, img):
    if args.vos_pkl is not None:
        if img.startswith('http'):
            img = img.replace('https://a.vpimg2.com/', 'http://a-appsimg.vip.vip.com/')
            img = download_pil_image(img)
        else:
            img = args.vos_client.download_vos_pil(img)
    return load_image(img)


def main(args):
    device = args.device
    dtype = args.dtype

    img_dict = load_file(args.vos_pkl)
    if args.gpt4v:
        img_dict = random.sample(img_dict, k=100)
        img_dict = {a[0]: eval(a[1]) for a in img_dict}
    elif args.llava:
        img_dict = random.sample(img_dict, k=100)
        img_dict = {a[0]: eval(a[1])['result'] for a in img_dict}

    # Load vae and image_encoder
    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae").to(device, dtype=dtype)

    # Load the tokenizer
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.base_model_path, subfolder="tokenizer")
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.base_model_path, subfolder="tokenizer_2")

    # import correct text encoder class
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.base_model_path, subfolder="text_encoder").to(device, dtype=dtype)
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.base_model_path, subfolder="text_encoder_2").to(device, dtype=dtype)

    # Load transformer and scheduler
    if args.transformer_model_path:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.transformer_model_path).to(device, dtype=dtype)
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.base_model_path, subfolder="transformer").to(device, dtype=dtype)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")

    pipe = FluxFillVIPPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        transformer=transformer,
    )
    if args.lora_model_path is not None:
        pipe.load_lora_weights(args.lora_model_path)
        pipe.fuse_lora(lora_scale=1.0)
        pipe.unload_lora_weights()

    candidate_prompt = [
        "Standing posture, left hand on the shoulder, right hand slightly raised and touching the cloth.",
        "Standing posture, the person is standing with their hands naturally hanging down at their sides.",
        "Standing posture, hands in pockets, body slightly turned to the left.",
        "Standing posture, left hand touching the hair.",
        "Standing posture, left hand making a V sign, right hand touching hair.",
        "Standing posture, right hand making a V sign gesture.",
        "Standing posture, left hand raised to head level, right hand down by the side.",
        "Standing posture, left hand lightly touching the right arm which is bent at the elbow.",
        "Standing posture, arms crossed in front of chest.",
        "Standing posture, body slightly turned to the side, left hand naturally hanging down, right hand slightly bent resting on the waist.",
        "Seated posture, right hand touching hair.",
        "Seated posture, right hand touching the chin.",
        "Seated posture, left hand resting on the thigh, right hand making a V sign gesture.",

        "Walking posture, left leg in mid-step, looking downward to the left, hair covering part of the face.",
        "Standing posture, left hand on hip, right arm hanging by the side.",
        "Standing posture, left hand touching hair.",
        "Standing posture, left hand raised to chest level, right hand down by the side.",
        "Standing posture, body slightly turned to the side, head slightly tilted downward, hands clasped together in front of chest.",
        "Standing posture, body slightly turned to the side, left hand holding a bag down.",
        "Standing posture, left leg bent and lifted, right hand holding a bag down.",
        "Seated posture, left leg crossed over right, left hand resting on the thigh.",
        "Seated on the floor, left arm extended outward, legs crossed at the ankles.",
        "Seated posture, left leg crossed over right, right hand resting on right leg",

        "Standing posture, upper-body photograph.",
        "Seated posture, upper-body photograph.",
        # "Standing posture, full-body photograph.",
    ]
    global_prompt = "[Pose Transition], This group of images illustrates the pose transitions of the same person wearing the same outfit in similar scenes."

    for i, (k, v) in enumerate(img_dict.items()):
        save_name = f"{splitext(basename(k))[0]}.jpg" if args.gpt4v or args.llava else f"{k}.jpg"
        print(f"{i + 1}/{len(img_dict)}: {save_name}")

        prompts_2 = []
        if args.gpt4v:
            src_prompt_2 = f"This group of images illustrates the {v['Gender']}, {v['Character']}; {v['Background']};"
            src_prompt_2 += f" [IMAGE1] {v['Posture']};"

            candidate_ = random.sample(candidate_prompt, k=args.num_images)
            for j in range(args.num_images):
                prompts_2.append(src_prompt_2 + f" [IMAGE2] {candidate_[j]};")

            src_img = read_image(args, k)

        elif args.llava:
            src_prompt_2 = f"This group of images illustrates the {v['gender']}, {v['group']}, {v['clothes']}; {v['background']};"
            src_prompt_2 += f" [IMAGE1] {v['posture']};"

            candidate_ = random.sample(candidate_prompt, k=args.num_images)
            for j in range(args.num_images):
                prompts_2.append(src_prompt_2 + f" [IMAGE2] {candidate_[j]};")

            src_img = read_image(args, k)

        else:
            item1 = random.choice(v)
            src_prompt_2 = f"This group of images illustrates the {item1['llava']['gender']}, {item1['llava']['group']}, {item1['llava']['clothes']}; {item1['llava']['background']};"
            src_prompt_2 += f" [IMAGE1] {item1['llava']['posture']};"

            for j, idx in enumerate(random.sample(range(len(v)), args.num_images)):
                item = v[idx]
                prompts_2.append(src_prompt_2 + f" [IMAGE2] {item['llava']['posture']};")

            src_img = read_image(args, item1['image'])

        src_img = resize_image(args, src_img, args.resolution)

        out_imgs = []
        for prompt_2 in prompts_2:
            out_img = pipe(
                prompt=global_prompt,
                prompt_2=prompt_2,
                image=src_img,
                height=args.resolution[0],
                width=args.resolution[1],
                num_inference_steps=50,
                guidance_scale=35,
                num_images_per_prompt=1,
                generator=torch.Generator("cpu").manual_seed(42),
                max_sequence_length=512,
            )[0]
            out_imgs.append(np.split(np.array(out_img), 2, axis=1)[1])

        out_imgs.insert(0, src_img)
        out_imgs = np.concatenate(out_imgs, axis=1)
        Image.fromarray(out_imgs).save(join(args.out_dir, save_name))


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
