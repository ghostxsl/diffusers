# Copyright (c) wilson.xu. All rights reserved.
import argparse
import os
from os.path import join, splitext, basename
from tqdm import tqdm
import torch

from diffusers.pipelines.byted.pipeline_qwenimage_edit import QwenImageEditPipeline
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from diffusers.data.utils import load_file, json_save
from diffusers.data.outer_vos_tools import load_or_download_image, decode_pil_bytes
from diffusers.data.byted.tos import _gen_name, get_file_from_tos


aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472)
    }
positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
    "zh": "，超清，4K，电影级构图",  # for chinese prompt
}
negative_prompt = "Vague, unclear, overexposure, low quality."


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file",
        default="/mnt/bn/creative-algo/xsl/data/atoms_dataset/test_v6.json",
        type=str)
    parser.add_argument(
        "--save_dir",
        default="/mnt/bn/creative-algo/xsl/qwen_plus_atoms",
        type=str)
    parser.add_argument(
        "--output_file",
        default="output_qwen_atoms.json",
        type=str)
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="/mnt/bn/creative-algo/xsl/models/Qwen-Image-Edit-2509",
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
        "--enable_cache", action="store_true", help="Whether to enable cache to boost denoising.")

    parser.add_argument(
        "--rank", default=None, type=int)
    parser.add_argument(
        "--num_ranks", default=None, type=int)

    parser.add_argument(
        "--device", default='cuda', type=str,
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    parser.add_argument(
        "--dtype", default='bf16', type=str,
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
    width, height = (1024, 1024)
    device = args.device
    dtype = args.dtype
    if args.transformer_model_path:
        # Load transformer
        transformer = QwenImageTransformer2DModel.from_pretrained(
            args.transformer_model_path, torch_dtype=dtype
        )
        transformer.requires_grad_(False)
        transformer.to(device, dtype=dtype)
        # Load pipeline
        pipe = QwenImageEditPipeline.from_pretrained(
            args.pretrained_model_path,
            transformer=transformer,
            enable_cache=args.enable_cache,
        )
    else:
        # Load pipeline
        pipe = QwenImageEditPipeline.from_pretrained(
            args.pretrained_model_path, enable_cache=args.enable_cache)
    pipe = pipe.to(dtype=dtype, device=device)

    if args.lora_model_path:
        # Load lora
        pipe.load_lora_weights(args.lora_model_path)
        pipe.fuse_lora(lora_scale=1.0)
        pipe.unload_lora_weights()

    # Load dataset
    data_list = load_file(args.input_file)
    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        data_list = data_list[args.rank::args.num_ranks]

    out = []
    for line in tqdm(data_list):
        try:
            product_img = load_or_download_image(line['ref_image'])
            prompt = line['source_prompt']
            out_img = pipe(
                image_reference=product_img,
                prompt=prompt + positive_magic["en"],
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=40,
                true_cfg_scale=4.0,
            ).images[0]

            name = _gen_name("") + '_gen.jpg'
            out_img.save(join(args.save_dir, name))
            line["generate_image"] = join(args.save_dir, name)

            out.append(line)
            if len(out) % 10 == 0:
                json_save(out, args.output_file)
        except Exception as e:
            print(line)
            print(e)

    # Final save
    json_save(out, args.output_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
