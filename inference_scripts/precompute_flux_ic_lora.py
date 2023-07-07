# Copyright (c) wilson.xu. All rights reserved.
import argparse
import os
from os.path import join, splitext, basename, exists
import random
from itertools import combinations
import torch
from torchvision.transforms import CenterCrop
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
    CLIPTextModel,
    T5EncoderModel,
)

from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.vip_utils import load_image
from diffusers.data.utils import load_file, pkl_save
from diffusers.data.vos_client import VOSClient
from diffusers.data.outer_vos_tools import download_pil_image


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--vos_pkl",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--out_vos_dir",
        default="output",
        type=str,
        help="Directory to save.")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="/apps/dat/cv/wty/models/flux/FLUX.1-dev",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="768x576",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )
    parser.add_argument(
        "--num_images",
        type=str,
        default="2x2",
    )
    parser.add_argument(
        "--crop_human",
        action="store_true",
        default=False,
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

    parser.add_argument(
        "--out_pkl",
        default="output_vos.pkl",
        type=str,
        help="Path to save list on vos.")

    args = parser.parse_args()

    if len(args.resolution.split("x")) == 1:
        args.resolution = [int(args.resolution),] * 2
    elif len(args.resolution.split("x")) == 2:
        args.resolution = [int(r) for r in args.resolution.split("x")]
    else:
        raise Exception(f"Error `resolution` type({type(args.resolution)}): {args.resolution}.")

    if len(args.num_images.split("x")) == 1:
        args.num_images = [1, int(args.num_images)]
    elif len(args.num_images.split("x")) == 2:
        args.num_images = [int(r) for r in args.num_images.split("x")]
    else:
        raise Exception(f"Error `num_images` type({type(args.num_images)}): {args.num_images}.")

    args.device = torch.device(args.device)
    if args.dtype == 'fp16':
        args.dtype = torch.float16
    elif args.dtype == 'bf16':
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float32

    return args


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


def resize_image(args, img):
    w, h = img.size
    oh, ow = args.resolution
    ow = int(oh / h * w)
    img = img.resize((ow, oh), 1)

    if img.width > args.resolution[1]:
        img = args.centercrop(img)
    else:
        img, _ = pad_image(img, size=args.resolution)

    return img


def crop_human_bbox(det_bbox, img_size, crop_size=(768, 576), pad_bbox=10):
    x1, y1, x2, y2 = det_bbox
    h, w = img_size
    ch, cw = crop_size

    x1 = 0 if x1 - pad_bbox < 0 else x1 - pad_bbox
    y1 = 0 if y1 - pad_bbox < 0 else y1 - pad_bbox
    x2 = w if x2 + pad_bbox > w else x2 + pad_bbox
    y2 = h if y2 + pad_bbox > h else y2 + pad_bbox

    bh, bw = y2 - y1, x2 - x1
    ratio_h, ratio_w = ch / bh, cw / bw

    # 长边resize
    if ratio_h < ratio_w:
        # 按高 resize
        ow = int(bh / ch * cw)
        expand_w = ow - bw

        x1 -= int(expand_w / 2)
        if x1 < 0:
            x1 = 0

        x2 += (expand_w - int(expand_w / 2))
        if x2 > w:
            x2 = w

        return [x1, y1, x2, y2], [bh, ow]
    elif ratio_h > ratio_w:
        # 按宽 resize
        oh = int(bw / cw * ch)
        expand_h = oh - bh

        y1 -= int(expand_h / 2)
        if y1 < 0:
            y1 = 0

        y2 += (expand_h - int(expand_h / 2))
        if y2 > h:
            y2 = h

        return [x1, y1, x2, y2], [oh, bw]
    else:
        return [x1, y1, x2, y2], [bh, bw]


def crop_human_image(args, img, pose, pad_bbox=10):
    if len(pose['body']['bboxes']) == 0:
        return resize_image(args, img)

    w, h = pose['width'], pose['height']
    bbox = np.int32(pose['body']['bboxes'][0] * np.array([w, h, w, h]))

    x1, y1, x2, y2 = bbox
    x1 = min(max(x1, 0), w)
    x2 = min(max(x2, 0), w)
    y1 = min(max(y1, 0), h)
    y2 = min(max(y2, 0), h)

    if x2 - x1 < 0.1 * w or y2 - y1 < 0.1 * h:
        return resize_image(args, img)

    crop_bbox, out_size = crop_human_bbox([x1, y1, x2, y2], (h, w), args.resolution, pad_bbox=pad_bbox)
    x1, y1, x2, y2 = crop_bbox
    crop_img = np.array(img)[y1: y2, x1: x2]
    crop_img, _ = pad_image(Image.fromarray(crop_img), size=out_size)

    crop_img = crop_img.resize(args.resolution[::-1], Image.LANCZOS)
    return crop_img


def concat_and_crop_image(args, img_list):
    out_img = []
    for i in range(args.num_images[0]):
        row_img = []
        for j in range(args.num_images[1]):
            item = img_list[i * args.num_images[1] + j]
            img = read_image(args, item['image'])
            if args.crop_human:
                pose = args.vos_client.download_vos_pkl(item['pose'])
                img = crop_human_image(args, img, pose)
            else:
                img = resize_image(args, img)
            row_img.append(img)
        row_img = np.concatenate(row_img, axis=1)
        out_img.append(row_img)
    out_img = np.concatenate(out_img, axis=0)

    return Image.fromarray(out_img)


def main(args):
    device = torch.device(args.device)
    dtype = args.dtype
    args.vos_client = VOSClient()
    args.centercrop = CenterCrop(args.resolution)

    # Load vae
    image_processor = VaeImageProcessor()
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device, dtype=dtype)

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    tokenizer_two = T5TokenizerFast.from_pretrained(args.pretrained_model_path, subfolder="tokenizer_2")

    # import correct text encoder classes
    text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(device, dtype=dtype)
    text_encoder_two = T5EncoderModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder_2").to(device, dtype=dtype)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # make clip text embeddings
    global_prompt = "[Pose Transition], This group of images illustrates the pose transitions of the same person wearing the same outfit in similar scenes."
    with torch.no_grad():
        pooled_input_ids = tokenizer_one(
            global_prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        ).input_ids
        pooled_input_ids = pooled_input_ids.to(device)
        pooled_prompt_embeds = text_encoder_one(
            pooled_input_ids, output_hidden_states=False).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.cpu()
        args.vos_client.upload_vos_pt(
            pooled_prompt_embeds, join(args.out_vos_dir, "pooled_prompt_embeds.text")
        )

    # load data list
    img_list = load_file(args.vos_pkl)

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        total_num = len(img_list)
        stride = int(total_num / args.num_ranks)

        start_idx = stride * args.rank
        end_idx = stride * (args.rank + 1) if args.rank + 1 < args.num_ranks else len(img_list)
        img_list = img_list[start_idx: end_idx]

    out_list = []
    nums = args.num_images[0] * args.num_images[1]
    for k, v in tqdm(img_list.items()):
        concat_idx = list(combinations(range(len(v)), nums))
        if len(concat_idx) > 100:
            concat_idx = random.sample(concat_idx, k=100)

        for idx in concat_idx:
            save_name = f"{k}_" + "_".join([str(a) for a in idx])

            item = {}
            try:
                # text embeddings
                caption_llava = f"This group of images illustrates the {v[idx[0]]['llava']['gender']}, {v[idx[0]]['llava']['group']}, {v[idx[0]]['llava']['clothes']}; {v[idx[0]]['llava']['background']};"
                for i, ind in enumerate(idx):
                    caption_llava += f" [IMAGE{i + 1}] {v[ind]['llava']['posture']};"

                with torch.no_grad():
                    input_ids = tokenizer_two(
                        caption_llava,
                        padding="max_length",
                        max_length=args.max_sequence_length,
                        truncation=True,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_tensors="pt",
                    ).input_ids
                    input_ids = input_ids.to(device)
                    prompt_embeds = text_encoder_two(
                        input_ids, output_hidden_states=False)[0]
                    prompt_embeds = prompt_embeds.cpu()

                if all(['gpt_4v' in v[i] for i in idx]):
                    caption_gpt = f"This group of images illustrates {v[idx[0]]['gpt_4v']['Character']}; {v[idx[0]]['gpt_4v']['Background']};"
                    for i, ind in enumerate(idx):
                        caption_gpt += f" [IMAGE{i + 1}] {v[ind]['gpt_4v']['Posture']};"

                    with torch.no_grad():
                        input_ids = tokenizer_two(
                            caption_gpt,
                            padding="max_length",
                            max_length=args.max_sequence_length,
                            truncation=True,
                            return_length=False,
                            return_overflowing_tokens=False,
                            return_tensors="pt",
                        ).input_ids
                        input_ids = input_ids.to(device)
                        prompt_embeds_gpt = text_encoder_two(
                            input_ids, output_hidden_states=False)[0]
                        prompt_embeds_gpt = prompt_embeds_gpt.cpu()
                        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_gpt])

                args.vos_client.upload_vos_pt(
                    prompt_embeds, join(args.out_vos_dir, save_name + '.text')
                )
                item['prompt_embeds'] = join(args.out_vos_dir, save_name + '.text')

                # vae latents
                temp = [v[i] for i in idx]
                image = concat_and_crop_image(args, temp)
                init_image = image_processor.preprocess(
                    image, height=image.height, width=image.width)
                flux_latents = flux_encode_vae_image(
                    vae, init_image.to(vae.device, dtype=vae.dtype))
                flux_latents = flux_latents.cpu()
                args.vos_client.upload_vos_pt(
                    flux_latents, join(args.out_vos_dir, save_name + '.pt')
                )
                item['latents'] = join(args.out_vos_dir, save_name + '.pt')

                out_list.append(item)
            except Exception as e:
                print(f"{save_name}: {e}")

    pkl_save(out_list, args.out_pkl)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
