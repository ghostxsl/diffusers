import os
from os.path import join, basename, splitext
import argparse
import torch
import pandas as pd

from diffusers.utils.vip_utils import load_image
from diffusers.data.outer_vos_tools import download_pil_image
from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline


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
        "--base_model_path",
        type=str,
        default="/mnt/bn/ttcc-algo-bytenas/zjn/models/FLUX.1-Kontext-dev",
    )

    parser.add_argument(
        "--resolution",
        type=str,
        default="1280x720",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )

    parser.add_argument(
        "--device",
        default='cuda',
        type=str,
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    parser.add_argument(
        "--dtype",
        default='bf16',
        type=str,
        help="Data type to use (e.g. bf16, fp16, fp32, etc.)")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if len(args.resolution.split("x")) == 1:
        args.resolution = [int(args.resolution),] * 2
    elif len(args.resolution.split("x")) == 2:
        args.resolution = [int(r) for r in args.resolution.split("x")]
    else:
        raise Exception(f"Error `resolution` type({type(args.resolution)}): {args.resolution}.")

    args.device = torch.device(args.device)
    if args.dtype == 'bf16':
        args.dtype = torch.bfloat16
    elif args.dtype == 'fp16':
        args.dtype = torch.float16
    else:
        args.dtype = torch.float32

    return args


def read_image(img):
    if img.startswith('http'):
        img = download_pil_image(img)
    return load_image(img)


def main(args):
    device = args.device
    dtype = args.dtype

    if args.input_file.endswith('.xlsx'):
        df = pd.read_excel(args.input_file)
    elif args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file, encoding='utf-8')
    else:
        raise Exception(f"Error `input_file` type:{args.input_file}")
    data_list = df.to_dict('records')

    # Load Flux.1-Kontext-dev pipeline
    pipe = FluxKontextPipeline.from_pretrained(args.base_model_path, torch_dtype=dtype).to(device)

    for i, line in enumerate(data_list):
        img_with_bg_url = line['bg_res_url']
        try:
            prompt = line['flux_kontext_prompt']
            # Flux-Kontext-dev to generate
            img = read_image(img_with_bg_url)
            out_img = pipe(
                image=img,
                prompt=prompt,
                height=args.resolution[0],
                width=args.resolution[1],
                num_inference_steps=50,
                guidance_scale=2.5,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).images[0]
            name = splitext(basename(img_with_bg_url))[0] + '_kontext.jpg'
            out_img.save(join(args.save_dir, name))

        except Exception as e:
            print(i, img_with_bg_url, e)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
