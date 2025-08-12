# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms

from diffusers.metrics.fid import FIDMetric
from diffusers.data.utils import load_file
from diffusers.data.outer_vos_tools import load_or_download_image


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--generated_dir",
        default=None,
        type=str,
        help="File for list of generated_image path.")
    parser.add_argument(
        "--gt_file",
        default=None,
        type=str,
        help="File for list of ground-truth image paths (just use for FID).")

    parser.add_argument(
        "--resolution",
        type=str,
        default="640x360",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the dataloader."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading."
    )

    parser.add_argument(
        "--device",
        default='cuda',
        type=str,
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")

    args = parser.parse_args()

    if len(args.resolution.split("x")) == 1:
        args.resolution = [int(args.resolution),] * 2
    elif len(args.resolution.split("x")) == 2:
        args.resolution = [int(r) for r in args.resolution.split("x")]
    else:
        raise Exception(f"Error `resolution` type({type(args.resolution)}): {args.resolution}.")

    args.device = torch.device(args.device)

    return args


class FIDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_list,
        img_size=(224, 224),
    ):
        self.image_list = image_list
        self.img_size = img_size

        self._length = len(self.image_list)

        self.image_transforms = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        img = load_or_download_image(self.image_list[index])
        example = self.image_transforms(img)

        return example


def main(args):
    gen_urls = [join(args.generated_dir, a) for a in os.listdir(args.generated_dir)]
    gt_urls = load_file(args.gt_file)

    fid = FIDMetric(device=args.device)
    # FID: 1. 计算生成图像
    gen_dataset = FIDDataset(image_list=gen_urls, img_size=args.resolution)
    gen_dataloader = torch.utils.data.DataLoader(
        gen_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    pred_act = []
    for batch in tqdm(gen_dataloader):
        pred_act.append(fid.get_activation_batch(batch))
    pred_act = np.concatenate(pred_act, axis=0)
    mu1, sigma1 = fid.calculate_activation_statistics(pred_act)

    # FID: 2. 计算GT
    gt_dataset = FIDDataset(image_list=gt_urls, img_size=args.resolution)
    gt_dataloader = torch.utils.data.DataLoader(
        gt_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    pred_act = []
    for batch in tqdm(gt_dataloader):
        pred_act.append(fid.get_activation_batch(batch))
    pred_act = np.concatenate(pred_act, axis=0)
    mu2, sigma2 = fid.calculate_activation_statistics(pred_act)

    # FID: 3. 计算FID距离
    fid_value = fid.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print("====== FID metric ======")
    print(f"FID score: {round(fid_value, 4)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
