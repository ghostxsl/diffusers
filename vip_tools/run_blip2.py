# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, split, exists
from tqdm import tqdm
import csv
import torch

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from diffusers.data.utils import *
from diffusers.utils.vip_utils import load_image
from diffusers.data.vos_client import VOSClient


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--img_dir",
        default=None,
        type=str,
        help="Path to reference image.")
    parser.add_argument(
        "--vos_pkl",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--out_csv",
        default="output.csv",
        type=str,
        help="File name to save.")

    parser.add_argument(
        "--rank",
        default=None,
        type=int)
    parser.add_argument(
        "--num_ranks",
        default=None,
        type=int)

    parser.add_argument(
        "--model_path",
        default="/apps/dat/file/llm/model/blip2-opt-2.7b-coco",
        type=str,
        help="Directory to weights.")
    parser.add_argument(
        "--device",
        default='cuda',
        type=str,
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    parser.add_argument(
        "--dtype",
        default='fp16',
        type=str,
        help="Data type to use (e.g. fp16, fp32, etc.)")

    args = parser.parse_args()
    out_dir, _ = split(args.out_csv)
    if out_dir and not exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    args.device = torch.device(args.device)
    if args.dtype == 'fp16':
        args.dtype = torch.float16
    else:
        args.dtype = torch.float32

    return args


def read_image(args, name):
    if args.vos_pkl is None:
        img = load_image(join(args.img_dir, name))
    else:
        img = args.vos_client.download_vos_pil(name)
        img = load_image(img)

    return img


def main(args):
    device = args.device
    dtype = args.dtype
    processor = Blip2Processor.from_pretrained(args.model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype).to(device)

    if args.vos_pkl is None:
        img_list = sorted(os.listdir(args.img_dir))
    else:
        img_list = load_file(args.vos_pkl)
        args.vos_client = VOSClient()

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        total_num = len(img_list)
        stride = int(total_num / args.num_ranks)

        start_idx = stride * args.rank
        end_idx = stride * (args.rank + 1) if args.rank + 1 < args.num_ranks else len(img_list)
        img_list = img_list[start_idx: end_idx]

        # save as csv file
        out_dir, out_file = split(args.out_csv)
        out_file = splitext(out_file)[0] + f"_{args.rank}.csv"
        args.out_csv = join(out_dir, out_file)

    done_list = []
    if exists(args.out_csv):
        done_list = set([a[0] for a in load_file(args.out_csv)])
        new_list = []
        for name in img_list:
            if name not in done_list:
                new_list.append(name)
        img_list = new_list

    f = open(args.out_csv, "a", encoding="utf-8", newline="")
    writer = csv.writer(f)
    if len(done_list) == 0:
        writer.writerow(['file_name', 'text'])
    del done_list

    for name in tqdm(img_list):
        try:
            img = read_image(args, name)

            inputs = processor(images=img, return_tensors="pt").to(device, dtype)
            generated_ids = model.generate(**inputs)
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True)[0].strip()

            writer.writerow([name, generated_text])
            f.flush()
        except Exception as e:
            print(e, name)
            continue

    f.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
