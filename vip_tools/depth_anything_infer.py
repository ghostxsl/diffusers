# Copyright (c) wilson.xu. All rights reserved.
import argparse
import os
from os.path import join, splitext, basename
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from diffusers.data.utils import load_file
from diffusers.utils.vip_utils import load_image
from diffusers.data.vos_client import VOSClient


def parse_args():
    parser = argparse.ArgumentParser(description="Depth Anything inference script.")
    parser.add_argument(
        "--img_dir",
        default=None,
        type=str,
        help="Directory to image.")
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        help="Directory to save.")
    parser.add_argument(
        "--vos_pkl",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--base_model_path",
        default="/apps/dat/file/llm/model/Depth-Anything-V2-Large-hf",
        # default="/apps/dat/file/llm/model/depth-anything-large-hf",
        type=str)

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

    args = parser.parse_args()

    if args.vos_pkl is None:
        os.makedirs(args.out_dir, exist_ok=True)

    return args


def read_image(args, name):
    if args.vos_pkl is None:
        img = load_image(join(args.img_dir, name))
    else:
        img = args.vos_client.download_vos_pil(name)
        img = load_image(img)

    return img


def main(args):
    device = torch.device(args.device)

    image_processor = AutoImageProcessor.from_pretrained(args.base_model_path)
    model = AutoModelForDepthEstimation.from_pretrained(args.base_model_path).to(device)
    model.requires_grad_(False)

    if args.vos_pkl is None:
        img_list = sorted(os.listdir(args.img_dir))
    else:
        img_list = load_file(args.vos_pkl)
        args.vos_client = VOSClient()

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks

        img_list = img_list[args.rank::args.num_ranks]

    damaged = []
    for name in tqdm(img_list):
        try:
            img = read_image(args, name)

            inputs = image_processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

                post_processed_output = image_processor.post_process_depth_estimation(
                    outputs,
                    target_sizes=[(img.height, img.width)],
                )
                predicted_depth = post_processed_output[0]["predicted_depth"]
                depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
                depth = depth.cpu().numpy() * 255
            depth_img = Image.fromarray(depth.astype("uint8"))

            if args.vos_pkl is None:
                depth_img.save(join(args.out_dir, splitext(name)[0] + '.jpg'))
            else:
                args.vos_client.upload_vos_pil(
                    depth_img,
                    join(args.out_dir, splitext(basename(name))[0] + '.jpg'),
                    format='JPEG',
                )
        except Exception as e:
            damaged.append(name)

    print(damaged)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
