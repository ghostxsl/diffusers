# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, basename
import argparse
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torchvision import transforms

from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from diffusers.metrics.lpips import LPIPSMetric
from diffusers.data.utils import load_file
from diffusers.data.outer_vos_tools import load_or_download_image


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--generated_dir",
        default=None,
        type=str,
        help="Directory for generated image.")
    parser.add_argument(
        "--gt_file",
        default=None,
        type=str,
        help="File for list of ground-truth image paths.")

    parser.add_argument(
        "--rec",
        action="store_true",
        help="Whether to compute reconstruction score."
    )
    parser.add_argument(
        "--lpips",
        action="store_true",
        help="Whether to compute LPIPS score."
    )

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
        default=8,
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


class LPIPSDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            input_list,
            img_size=(224, 224)
    ):
        self.metadata = input_list
        self._length = len(self.metadata)

        self.image_transforms = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        gen_img, gt_img = self.metadata[index]

        gen_img = load_or_download_image(gen_img)
        gen_img = self.image_transforms(gen_img)

        gt_img = load_or_download_image(gt_img)
        gt_img = self.image_transforms(gt_img)

        return gen_img, gt_img


def main(args):
    device = args.device

    gt_urls = load_file(args.gt_file)
    nums = len(gt_urls)
    input_list = []
    for url in gt_urls:
        gen_name = basename(url).replace("_poster.jpg", "_lora.jpg")
        input_list.append([
            join(args.generated_dir, gen_name), url
        ])

    if args.rec:
        # 计算重建误差
        def rec_single_process(gen_url, gt_url):
            gen_img = load_or_download_image(gen_url)
            gen_img = gen_img.resize(args.resolution[::-1], 1)
            gen_img = np.array(gen_img).astype(np.float32)

            gt_img = load_or_download_image(gt_url)
            gt_img = gt_img.resize(args.resolution[::-1], 1)
            gt_img = np.array(gt_img).astype(np.float32)

            ssim = structural_similarity(
                gt_img, gen_img,
                data_range=255,
                channel_axis=2,
                gaussian_weights=True,
                sigma=1.2,
                use_sample_covariance=False,
            )
            psnr = peak_signal_noise_ratio(gt_img.copy(), gen_img.copy(), data_range=255)
            l1 = np.mean(np.abs(gt_img - gen_img))

            return ssim, psnr, l1

        ssim_result, psnr_result, l1_result = [], [], []
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_list = [
                executor.submit(
                    rec_single_process, gen_url, gt_url
                ) for gen_url, gt_url in input_list]
            with tqdm(total=nums) as pbar:
                for future in concurrent.futures.as_completed(future_list):
                    ssim, psnr, l1 = future.result()
                    pbar.update(1)  # Update progress bar

                    ssim_result.append(ssim)
                    psnr_result.append(psnr)
                    l1_result.append(l1)

        print("====== Reconstruction metric ======")
        print(f"SSIM: {np.round(np.mean(ssim_result), 4)}")
        print(f"PSNR: {np.round(np.mean(psnr_result), 4)}")
        print(f"L1: {np.round(np.mean(l1_result), 4)}")


    if args.lpips:
        lpips_dataset = LPIPSDataset(
            input_list=input_list,
            img_size=args.resolution,
        )
        lpips_dataloader = torch.utils.data.DataLoader(
            lpips_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        # 计算LPIPS (net='alex')
        lpips_metric = LPIPSMetric(net='vgg', device=device)

        lpips_result = []
        for gen_img, gt_img in tqdm(lpips_dataloader):
            lpips_result.append(lpips_metric(gen_img, gt_img, normalize=True))

        lpips_result = torch.mean(torch.cat(lpips_result, dim=0)).item()

        print("====== LPIPS metric ======")
        print(f"LPIPS: {round(lpips_result, 4)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
