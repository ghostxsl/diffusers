# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, exists, splitext, basename
import argparse
import time
import copy
import random
from tqdm import tqdm
import torch
from torchvision import transforms as tv_transforms
import numpy as np
from PIL import Image

from diffusers.utils.vip_utils import *
from diffusers.data.utils import *
from diffusers.data.transforms import *
from diffusers.data.vos_client import VOSClient


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--vos_pkl",
        default="/apps/dat/cv/xsl/vos_data/train_dress_vos.pkl,/apps/dat/cv/xsl/vos_data/train_upper1_vos.pkl,/apps/dat/cv/xsl/vos_data/train_upper2_vos.pkl,/apps/dat/cv/xsl/vos_data/train_upper3_vos.pkl,/apps/dat/cv/xsl/vos_data/train_upper4_vos.pkl,/apps/dat/cv/xsl/vos_data/train_upper5_vos.pkl",
        type=str,
        help=" ")
    parser.add_argument(
        "--out_dir",
        default="output",
        type=str,
        help="Directory to save.")

    parser.add_argument(
        "--num_display",
        default=100,
        type=int,
        help="")
    parser.add_argument(
        "--resolution",
        type=str,
        default="768x576",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if len(args.resolution.split("x")) == 1:
        args.resolution = [int(args.resolution), ] * 2
    elif len(args.resolution.split("x")) == 2:
        args.resolution = [int(r) for r in args.resolution.split("x")]
    else:
        raise Exception(f"Error `resolution` type({type(args.resolution)}): {args.resolution}.")

    args.vos = VOSClient()

    return args


def get_metadata(dataset_file):
    print("Loading dataset...")
    dataset_file = [d.strip() for d in dataset_file.split(',') if len(d.strip()) > 0]
    out = {}
    for file_path in dataset_file:
        temp = load_file(file_path)
        out.update(temp)
    return out


def main(args):
    image_transforms = tv_transforms.Compose(
        [
            DrawPose(prob_face=0.0),
            PasteMatting(),
            BoxCrop(args.resolution),
            ResizePad(args.resolution, padding=isinstance(args.resolution, int)),
            RandomHorizontalFlip(),
        ]
    )

    data_dict = get_metadata(args.vos_pkl)
    data_keys = list(data_dict.keys())
    random.shuffle(data_keys)

    for i in tqdm(range(args.num_display)):
        temp_list = copy.deepcopy(data_dict[data_keys[i]])
        item = random.choice(temp_list)
        img = load_image(args.vos.download_vos_pil(item["image"]))
        points = args.vos.download_vos_pkl(item["pose"])
        matting = load_image(args.vos.download_vos_pil(item["matting"]))

        if len(temp_list) > 1:
            temp_list.remove(item)
        ref_item = random.choice(temp_list)
        ref_img = load_image(args.vos.download_vos_pil(ref_item["image"]))
        ref_points = args.vos.download_vos_pkl(ref_item["pose"])
        ref_matting = load_image(args.vos.download_vos_pil(ref_item["matting"]))

        data = {
            'image': img,
            'condition': points,
            'matting': matting,
            'reference_image': ref_img,
            'reference_condition': ref_points,
            'reference_matting': ref_matting,
        }
        data = image_transforms(data)

        out_img = np.concatenate(
            [img.resize([args.resolution[0], args.resolution[0]], 1),
            data["image"], data["condition_image"],
            ref_img.resize([args.resolution[0], args.resolution[0]], 1),
            data["reference_image"]], axis=1)
        Image.fromarray(out_img).save(join(args.out_dir, data_keys[i] + '.jpg'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
