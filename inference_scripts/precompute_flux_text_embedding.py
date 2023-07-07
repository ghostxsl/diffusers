# Copyright (c) wilson.xu. All rights reserved.
import argparse
import os
from os.path import join, splitext, basename, exists
import torch
from tqdm import tqdm
from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
    CLIPTextModel,
    T5EncoderModel,
)

from diffusers.data.utils import load_file, json_save
from diffusers.data.vos_client import VOSClient


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
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

    args.device = torch.device(args.device)
    if args.dtype == 'fp16':
        args.dtype = torch.float16
    elif args.dtype == 'bf16':
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float32

    return args


def main(args):
    device = torch.device(args.device)
    dtype = args.dtype
    args.vos_client = VOSClient()

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    tokenizer_two = T5TokenizerFast.from_pretrained(args.pretrained_model_path, subfolder="tokenizer_2")

    # import correct text encoder classes
    text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    text_encoder_two = T5EncoderModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder_2")
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_one.to(device, dtype=dtype)
    text_encoder_two.to(device, dtype=dtype)

    # make clip text embeddings
    with torch.no_grad():
        pooled_input_ids = tokenizer_one(
            "[Pose Transition], This pair of images highlights the pose transitions of the same person in similar scenes.",
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
        if args.vos_pkl is not None:
            args.vos_client.upload_vos_pt(
                pooled_prompt_embeds, join(args.out_dir, "pooled_prompt_embeds.text")
            )
        else:
            torch.save(pooled_prompt_embeds, join(args.out_dir, "pooled_prompt_embeds.text"))

    # load data list
    img_list = load_file(args.vos_pkl)

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        total_num = len(img_list)
        stride = int(total_num / args.num_ranks)

        start_idx = stride * args.rank
        end_idx = stride * (args.rank + 1) if args.rank + 1 < args.num_ranks else len(img_list)
        img_list = img_list[start_idx: end_idx]

    damaged = []
    for k, v in tqdm(img_list.items()):
        for item in v:
            try:
                caption = item["caption"]
                caption = f"[Pose Transition], This pair of images highlights the pose transitions of the same person in similar scenes. " \
                          f"[IMAGE1] {caption} [IMAGE2] In a similar scene, different photo poses from the person in [IMAGE1]. "
                with torch.no_grad():
                    input_ids = tokenizer_two(
                        caption,
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

                save_name = splitext(basename(item['image']))[0] + '.text'
                if args.vos_pkl is not None:
                    args.vos_client.upload_vos_pt(
                        prompt_embeds, join(args.out_dir, save_name)
                    )
                else:
                    torch.save(prompt_embeds, join(args.out_dir, save_name))
            except:
                damaged.append(item['image'])

    if args.rank is not None:
        json_save(damaged, f"./damaged_{args.rank}.json")
    else:
        json_save(damaged, "./damaged.json")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
