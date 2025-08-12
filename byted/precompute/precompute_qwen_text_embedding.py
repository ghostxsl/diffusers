# Copyright (c) wilson.xu. All rights reserved.
import argparse
import os
from os.path import join, splitext, basename, exists
import torch
from tqdm import tqdm
from transformers import (
    Qwen2Tokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from diffusers.pipelines.qwenimage import QwenImagePipeline
from diffusers.data.utils import load_file


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
        "--pretrained_model_path",
        type=str,
        default="/mnt/bn/ttcc-algo-bytenas/xsl/Qwen-Image",
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


@torch.no_grad()
def compute_text_embeddings(prompt, text_encoding_pipeline, device, dtype):
    prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
        prompt, device=device)
    return prompt_embeds.to(dtype), prompt_embeds_mask


def main(args):
    device = torch.device(args.device)
    dtype = args.dtype

    tokenizer = Qwen2Tokenizer.from_pretrained(
        args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_path, subfolder="text_encoder", torch_dtype=dtype)

    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=dtype)

    text_encoding_pipeline = QwenImagePipeline.from_pretrained(
        args.pretrained_model_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
    ).to(device)

    # load data list
    img_list = load_file(args.input_file)
    if isinstance(img_list, dict):
        img_list = [v for v in img_list.values()]

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks

        img_list = img_list[args.rank::args.num_ranks]

    for item in tqdm(img_list):
        try:
            prompt = item[-3]
            prompt_embeds, prompt_embeds_mask = compute_text_embeddings(
                prompt,
                text_encoding_pipeline,
                device,
                dtype
            )
            prompt_embeds = prompt_embeds.cpu()
            prompt_embeds_mask = prompt_embeds_mask.cpu()

            save_name = basename(item[-4]) + '.text'
            text_embeds = {
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
            }
            torch.save(text_embeds, join(args.save_dir, save_name))

        except Exception as e:
            print(e)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
