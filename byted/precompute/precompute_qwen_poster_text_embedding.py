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

from diffusers.pipelines.byted.pipeline_qwenimage_edit import QwenImageEditPipeline
from diffusers.data.utils import load_csv_or_xlsx_to_dict, load_file, json_save


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument("--input_file", default=None, type=str)
    parser.add_argument("--output_file", default=None, type=str)
    parser.add_argument(
        "--save_dir",
        default="text_embeddings",
        type=str,
        help="Directory to save.")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="/mnt/bn/creative-algo/xsl/models/Qwen-Image-Edit-2509",
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
    prompt_embeds = prompt_embeds.to(dtype).cpu()
    prompt_embeds_mask = prompt_embeds_mask.cpu()

    return {
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
    }


def main(args):
    device = torch.device(args.device)
    dtype = args.dtype

    # init pipeline
    tokenizer = Qwen2Tokenizer.from_pretrained(
        args.pretrained_model_path, subfolder="tokenizer")
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
        processor=None,
        scheduler=None,
    ).to(device)

    # load data list
    try:
        data_list = load_csv_or_xlsx_to_dict(args.input_file)
    except Exception:
        data_list = load_file(args.input_file)

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        data_list = data_list[args.rank::args.num_ranks]

    out = []
    error_out = []
    for item in tqdm(data_list):
        try:
            prompt_keys = [a for a in item.keys() if 'prompt' in a]
            text_embeddings = []
            for k in prompt_keys:
                text_embedding = compute_text_embeddings(
                    item[k],
                    text_encoding_pipeline,
                    device,
                    dtype
                )
                text_embeddings.append(text_embedding)

            if len(text_embeddings) > 0:
                name = basename(item['poster_url'])
                torch.save(text_embeddings, join(args.save_dir, name + '.text'))
                item["text_embedding"] = join(args.save_dir, name + ".text")
                out.append(item)
                if len(out) % 10 == 0:
                    json_save(out, args.output_file)
        except Exception as e:
            print(e)
            error_out.append((item, str(e)))

    json_save(out, args.output_file)
    print(len(error_out))
    print(error_out)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
