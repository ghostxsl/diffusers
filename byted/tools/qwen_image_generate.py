# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join
import argparse
from tqdm import tqdm
import torch

from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from diffusers.data.utils import load_file, json_save
from diffusers.data.byted.tos import _gen_name


aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
}
positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": "，超清，4K，电影级构图", # for chinese prompt
}
negative_prompt = "Vague, unclear, overexposure, low quality."


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument("--input_file", default=None, type=str, help="Path to image list on vos.")
    parser.add_argument("--output_file", default="output.json", type=str, help="Path to image list on vos.")
    parser.add_argument(
        "--save_dir", default="/mnt/bn/creative-algo/xsl/qwen_image_generate", type=str, help="Directory to save."
    )
    parser.add_argument("--ratio", default="9:16", type=str)

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


def main(args):
    width, height = aspect_ratios[args.ratio]
    device = args.device
    dtype = args.dtype
    pipe = QwenImagePipeline.from_pretrained("/mnt/bn/creative-algo/xsl/models/Qwen-Image", torch_dtype=dtype)
    pipe = pipe.to(device)

    # Load dataset
    data_list = load_file(args.input_file)
    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        data_list = data_list[args.rank::args.num_ranks]

    out = []
    for line in tqdm(data_list):
        try:
            prompt = line["prompt"]
            out_img = pipe(
                prompt=prompt + positive_magic["en"],
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=50,
                true_cfg_scale=4.0,
                ).images[0]

            name = _gen_name("")
            out_img.save(join(args.save_dir, name), format="JPEG")
            line["generate_image"] = join(args.save_dir, name)

            out.append(line)
            if len(out) % 10 == 0:
                json_save(out, args.output_file)
        except Exception as e:
            print(line['source_url'], e)

    # Final save
    json_save(out, args.output_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
