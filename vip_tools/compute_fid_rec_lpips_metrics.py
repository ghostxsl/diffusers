# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, splitext, exists, basename
import concurrent
from concurrent.futures import ThreadPoolExecutor
import argparse
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from diffusers.metrics.fid import FIDMetric
from diffusers.metrics.lpips import LPIPSMetric
from diffusers.data.vos_client import VOSClient
from diffusers.data.utils import load_file, pkl_load
from diffusers.utils.vip_utils import load_image
from diffusers.data.outer_vos_tools import download_pil_image


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--generated_dir",
        default=None,
        type=str,
        help="Directory to generated.")
    parser.add_argument(
        "--split",
        action="store_true",
        help="Whether to split generated image."
    )
    parser.add_argument(
        "--gt_dir",
        default=None,
        type=str,
        help="Path to image list.")
    parser.add_argument(
        "--gt_pkl",
        default=None,
        type=str,
        help="Path to image list.")

    parser.add_argument(
        "--fid",
        action="store_true",
        help="Whether to compute fid score."
    )
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
        default="512x512",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the dataloader."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="Number of subprocesses to use for data loading."
    )

    parser.add_argument(
        "--device",
        default='cuda',
        type=str,
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    parser.add_argument(
        "--dtype",
        default='fp32',
        type=str,
        help="Data type to use (e.g. fp16, fp32, etc.)")

    args = parser.parse_args()

    args.vos_client = VOSClient()

    if len(args.resolution.split("x")) == 1:
        args.resolution = [int(args.resolution),] * 2
    elif len(args.resolution.split("x")) == 2:
        args.resolution = [int(r) for r in args.resolution.split("x")]
    else:
        raise Exception(f"Error `resolution` type({type(args.resolution)}): {args.resolution}.")

    args.device = torch.device(args.device)
    if args.dtype == 'fp16':
        args.dtype = torch.float16
    elif args.dtype == 'bf16':
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float32

    return args


class FIDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        local_dir=None,
        vos_pkl=None,
        img_size=(512, 512),
        split=False,
    ):
        self.local_dir = local_dir
        self.vos_pkl = vos_pkl
        self.img_size = img_size
        self.split = split

        if local_dir is not None:
            assert exists(local_dir)
            self.metadata = [a for a in os.listdir(local_dir)
                             if splitext(a)[-1].lower() in ['.jpg', '.png', '.jpeg']]
        elif vos_pkl is not None:
            self.metadata = load_file(vos_pkl)
        else:
            raise Exception("error: `generated_dir` and `gt_pkl` are both None.")

        self._length = len(self.metadata)

        self.image_transforms = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor()
        ])

        self.vos = VOSClient()

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if self.local_dir is not None:
            img = load_image(join(self.local_dir, self.metadata[index]))
            if self.split:
                img = np.split(np.array(img), 3, axis=1)[-1]
                img = Image.fromarray(img)
        elif self.vos_pkl is not None:
            img = self.vos.download_vos_pil(self.metadata[index])
        else:
            raise Exception("No img.")

        example = self.image_transforms(img)

        return example


class LPIPSDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            generated_dir,
            gt_dir=None,
            gt_pkl=None,
            img_size=(512, 512),
            split=False):
        self.generated_dir = generated_dir
        self.gt_dir = gt_dir
        self.gt_pkl = gt_pkl
        self.img_size = img_size
        self.split = split

        assert exists(generated_dir)
        gen_list = [a for a in os.listdir(generated_dir)
                    if splitext(a)[-1].lower() in ['.jpg', '.png', '.jpeg']]

        if gt_dir is not None:
            gt_list = [a for a in os.listdir(gt_dir)
                       if splitext(a)[-1].lower() in ['.jpg', '.png', '.jpeg']]
            gt_list = {splitext(a)[0]: a for a in gt_list}
        elif gt_pkl is not None:
            gt_list = load_file(gt_pkl)
            gt_list = {splitext(basename(a))[0]: a for a in gt_list}
        else:
            raise Exception("error: `gt_dir` and `gt_pkl` are both None.")

        self.metadata = [[a, gt_list[splitext(a)[0].split('_to_')[-1]]] for a in gen_list]
        self._length = len(self.metadata)

        self.image_transforms = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor()
        ])

        self.vos = VOSClient()

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        gen_name, gt_name = self.metadata[index]
        gen_img = load_image(join(self.generated_dir, gen_name))
        if self.split:
            gen_img = np.split(np.array(gen_img), 3, axis=1)[-1]
            gen_img = Image.fromarray(gen_img)

        if self.gt_dir is not None:
            gt_img = load_image(join(self.gt_dir, gt_name))
        elif self.gt_pkl is not None:
            gt_img = self.vos.download_vos_pil(gt_name)
        else:
            raise Exception("No img.")

        gen_img = self.image_transforms(gen_img)
        gt_img = self.image_transforms(gt_img)

        return gen_img, gt_img


def read_image(args, img):
    if img.startswith('http'):
        img = img.replace('https://a.vpimg2.com/', 'http://a-appsimg.vip.vip.com/')
        img = download_pil_image(img)
    else:
        img = args.vos_client.download_vos_pil(img)
    return load_image(img)


def main(args):
    device = args.device
    dtype = args.dtype

    if args.fid:
        fid = FIDMetric(device=device)

        # FID: 1. 计算生成图像
        gen_dataset = FIDDataset(
            local_dir=args.generated_dir,
            img_size=args.resolution,
            split=args.split,
        )
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
        if args.gt_dir is not None:
            gt_dataset = FIDDataset(local_dir=args.gt_dir, img_size=args.resolution)
        elif args.gt_pkl is not None:
            gt_dataset = FIDDataset(vos_pkl=args.gt_pkl, img_size=args.resolution)
        else:
            raise Exception("`gt_dir` and `gt_pkl` both are None.")

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


    if args.rec:
        # 计算重建误差
        assert exists(args.generated_dir)
        gen_list = [a for a in os.listdir(args.generated_dir)
                    if splitext(a)[-1].lower() in ['.jpg', '.png', '.jpeg']]

        def rec_single_process(gen_path, gt_path):
            gen_img = Image.open(gen_path)
            if args.split:
                gen_img = np.split(np.array(gen_img), 3, axis=1)[-1]
                gen_img = Image.fromarray(gen_img)
            gen_img = gen_img.resize(args.resolution[::-1], 1)
            gen_img = np.array(gen_img).astype(np.float32)

            gt_img = Image.open(gt_path).resize(args.resolution[::-1], 1)
            gt_img = np.array(gt_img).astype(np.float32)

            ssim = structural_similarity(
                gt_img.copy(), gen_img.copy(),
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
                    rec_single_process,
                    join(args.generated_dir, gen_name),
                    join(args.gt_dir, splitext(gen_name)[0].split('_to_')[-1] + '.jpg')
                ) for gen_name in gen_list]
            with tqdm(total=len(gen_list)) as pbar:
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
            generated_dir=args.generated_dir,
            gt_dir=args.gt_dir,
            gt_pkl=args.gt_pkl,
            img_size=args.resolution,
            split=args.split,
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
