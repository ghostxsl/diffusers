import os
from os.path import split, splitext, join
from tqdm import tqdm
import argparse
from diffusers.data.utils import *
from diffusers.data.vos_client import VOSClient
from diffusers.utils.vip_utils import load_image


def parse_args():
    parser = argparse.ArgumentParser(description="llm-vos upload script.")
    parser.add_argument(
        "--data_file",
        default=None,
        type=str,
        help=" ")
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

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    damaged = []
    data_list = load_file(args.data_file)
    vos_client = VOSClient()
    for name, _ in tqdm(data_list):
        try:
            img = load_image(join(args.img_dir, name))

            if splitext(name) in ['.jpg', '.jpeg']:
                vos_client.upload_vos_pil(img, join(args.out_dir, name), format='jpeg')
            else:
                vos_client.upload_vos_pil(img, join(args.out_dir, splitext(name)[0] + '.png'))
        except:
            damaged.append(name)

    print('Done!')
    print(damaged)
