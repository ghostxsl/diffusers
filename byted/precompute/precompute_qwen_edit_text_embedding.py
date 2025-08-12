# Copyright (c) wilson.xu. All rights reserved.
import argparse
import os
from os.path import join, splitext, basename, exists
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    Qwen2VLProcessor,
    Qwen2Tokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline
from diffusers.data.utils import load_csv_or_xlsx_to_dict, load_file, json_save
from diffusers.data.outer_vos_tools import load_or_download_image


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--output_file",
        default="output.json",
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--save_dir",
        default="/mnt/bn/creative-algo/xsl/data/gpt_dataset/text_embeddings",
        type=str,
        help="Directory to save.")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="/mnt/bn/creative-algo/xsl/models/Qwen-Image-Edit",
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

    args.device = torch.device(args.device)
    if args.dtype == 'fp16':
        args.dtype = torch.float16
    elif args.dtype == 'bf16':
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float32

    return args


def calculate_dimensions(target_area, ratio, mod_value=14):
    width = round(np.sqrt(target_area * ratio))
    height = round(np.sqrt(target_area / ratio))

    width = int(width // mod_value * mod_value)
    height = int(height // mod_value * mod_value)

    return (width, height)


@torch.no_grad()
def compute_text_embeddings(prompt, image, text_encoding_pipeline, device, dtype):
    prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
        prompt, image, device=device)
    return prompt_embeds.to(dtype), prompt_embeds_mask


def main(args):
    device = torch.device(args.device)
    dtype = args.dtype
    image_area = 224 * 224

    processor = Qwen2VLProcessor.from_pretrained(args.pretrained_model_path, subfolder="processor")
    tokenizer = Qwen2Tokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_path, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=dtype)

    text_encoding_pipeline = QwenImageEditPipeline.from_pretrained(
        args.pretrained_model_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        processor=processor,
        scheduler=None,
    ).to(device)

    # load data list
    try:
        data_list = load_csv_or_xlsx_to_dict(args.input_file)
    except:
        data_list = load_file(args.input_file)

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        data_list = data_list[args.rank::args.num_ranks]

    out = []
    for item in tqdm(data_list):
        try:
            image = load_or_download_image(item['product_image'])
            image_size = calculate_dimensions(image_area, image.width / image.height, mod_value=14)
            image = image.resize(image_size, 1)

            prompt = item['prompt']
            prompt_embeds, prompt_embeds_mask = compute_text_embeddings(
                prompt,
                image,
                text_encoding_pipeline,
                device,
                dtype
            )
            prompt_embeds = prompt_embeds.cpu()
            prompt_embeds_mask = prompt_embeds_mask.cpu()

            text_embeds = {
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
            }
            name = basename(item['poster_url'])
            torch.save(text_embeds, join(args.save_dir, name + '.text'))

            item['text_embedding'] = join(args.save_dir, name + '.text')
            out.append(item)
        except Exception as e:
            print(e)

    json_save(out, args.output_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
