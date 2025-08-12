# Copyright (c) wilson.xu. All rights reserved.
import argparse
import os
import torch

from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--transformer_model_path",
        type=str,
        default="/mnt/bn/creative-algo/xsl/xsl-subject-1126/checkpoint-30000",
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default="/mnt/bn/creative-algo/xsl/xsl-poster-1201/checkpoint-30000/pytorch_lora_weights.safetensors",
    )
    parser.add_argument(
        "--save_dir",
        default="/mnt/bn/creative-algo/xsl/qwen_poster_transformer",
        type=str,
        help="Directory to save.")

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


def main(args):
    device = args.device
    dtype = args.dtype
    # Load transformer
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.transformer_model_path, torch_dtype=dtype)
    transformer.requires_grad_(False)
    transformer.to(device, dtype=dtype)
    # Load lora
    transformer.load_lora_adapter(args.lora_model_path)
    transformer.fuse_lora(lora_scale=1.0)
    transformer.unload_lora()
    transformer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
