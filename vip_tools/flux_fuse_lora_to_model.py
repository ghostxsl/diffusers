# Copyright (c) wilson.xu. All rights reserved.
import argparse
import torch

from diffusers.models.vip.transformer_flux import FluxTransformer2DModel


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/apps/dat/cv/wty/models/flux/FLUX.1-dev/transformer",
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default="/apps/dat/cv/xsl/exp_animate/checkpoint-12000/pytorch_lora_weights.safetensors",
    )
    parser.add_argument(
        "--out_dir",
        default="flux_fuse_lora",
        type=str,
        help="Directory to save.")

    parser.add_argument(
        "--device",
        default='cpu',
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
    device = args.device
    dtype = args.dtype

    # Load transformer and scheduler
    transformer = FluxTransformer2DModel.from_pretrained(args.base_model_path)
    transformer = transformer.to(device=device, dtype=dtype)

    transformer.load_lora_adapter(args.lora_model_path)
    transformer.fuse_lora(lora_scale=1.0)
    transformer.unload_lora()

    transformer.save_pretrained(args.out_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
