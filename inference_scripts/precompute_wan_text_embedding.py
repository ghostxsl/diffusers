# Copyright (c) wilson.xu. All rights reserved.
import argparse
import os
from os.path import join, splitext, basename, exists
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    UMT5EncoderModel,
)

from diffusers.data.utils import load_file
from diffusers.data.vos_client import VOSClient


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file",
        default="/apps/dat/cv/zjn/data/videos/train_100/train_100.json",
        type=str)
    parser.add_argument(
        "--save_dir",
        default="/apps/dat/cv/zjn/data/videos/train_100/text",
        type=str,
        help="Directory to save.")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="/home/llms/Wan-FLF2V-14B-720P-dif",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--use_vos",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--vos_bucket",
        type=str,
        default="public",
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
    if not args.use_vos:
        os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device)
    dtype = args.dtype
    args.vos_client = VOSClient(args.vos_bucket)

    # Load the tokenizers
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")

    # import correct text encoder classes
    text_encoder = UMT5EncoderModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=dtype)

    # load data list
    img_list = load_file(args.input_file)

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks

        img_list = img_list[args.rank::args.num_ranks]

    for item in tqdm(img_list):
        try:
            with torch.no_grad():
                text_inputs = tokenizer(
                    item['text'],
                    padding="max_length",
                    max_length=args.max_sequence_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
                seq_lens = mask.gt(0).sum(dim=1).long()
                prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
                prompt_embeds = prompt_embeds[:, :seq_lens]
                prompt_embeds = torch.cat(
                    [prompt_embeds,
                     prompt_embeds.new_zeros(1, args.max_sequence_length - seq_lens, prompt_embeds.size(2))],
                    dim=1
                )
                prompt_embeds = prompt_embeds.cpu()

            save_name = splitext(basename(item['file_path']))[0] + '.pt'
            if not args.use_vos:
                torch.save(prompt_embeds, join(args.save_dir, save_name))
            else:
                args.vos_client.upload_vos_pt(prompt_embeds, join(args.save_dir, save_name))

        except Exception as e:
            print(e)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
