# Copyright (c) wilson.xu. All rights reserved.
import argparse
import time
from os.path import join, splitext, basename
from tqdm import tqdm
import random
import numpy as np
from PIL import Image

from diffusers.data.utils import load_file, csv_save, pad_image
from diffusers.data.outer_vos_tools import load_or_download_image, encode_pil_bytes
from diffusers.data.byted.tos import save_tos

# w, h
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
}
# left, right, top, bottom
safe_ratios={
    "1:1": 0.2,
    "9:16": (0.012, 0.125, 0.2, 0.125),
}


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--output_file",
        default="output.csv",
        type=str,
        help="Path to image list on vos.")

    parser.add_argument(
        "--rank",
        default=None,
        type=int)
    parser.add_argument(
        "--num_ranks",
        default=None,
        type=int)

    args = parser.parse_args()

    return args


def get_gemini_product_image(image, ratio="9:16", margin=0.01):
    w, h = image.size
    W, H = aspect_ratios[ratio]
    if ratio == "9:16":
        left, right, top, bottom = safe_ratios[ratio]
        safe_w, safe_h = int(W * (1 - left - right)), int(H * (1 - top - bottom))
        min_w, min_h = int(W * 0.7), int(H * 0.6)
        if w > safe_w or h > safe_h:
            resize_ratio = min(safe_w / w, safe_h / h)
            w, h = int(w * resize_ratio), int(h * resize_ratio)
            image = image.resize((w, h), 1)
        elif w < min_w and h < min_h:
            resize_ratio = min(min_w / w, min_h / h)
            w, h = int(w * resize_ratio), int(h * resize_ratio)
            image = image.resize((w, h), 1)
        pad_left = random.randint(0, safe_w - w) + int(W * left)
        pad_right = W - pad_left - w
        pad_top = random.randint(0, safe_h - h) + int(H * top)
        pad_bottom = H - pad_top - h
    elif ratio == "1:1":
        safe_ratio = safe_ratios[ratio]
        if w > h:
            safe_w, safe_h = int(W * (1 - margin * 2)), int(H * (1 - safe_ratio))
            if w > safe_w or h > safe_h:
                resize_ratio = min(safe_w / w, safe_h / h)
                w, h = int(w * resize_ratio), int(h * resize_ratio)
                image = image.resize((w, h), 1)
            elif w < int(W * 0.6):
                resize_ratio = int(W * 0.6) / w
                w, h = int(w * resize_ratio), int(h * resize_ratio)
                image = image.resize((w, h), 1)
            pad_left = random.randint(0, safe_w - w) + int(W * margin)
            pad_right = W - pad_left - w
            # 上下随机作为安全区
            pad_top = random.randint(0, safe_h - h) + int(H * safe_ratio) * random.randint(0, 1)
            pad_bottom = H - pad_top - h
        else:
            safe_w, safe_h = int(W * (1 - safe_ratio)), int(H * (1 - margin * 2))
            if w > safe_w or h > safe_h:
                resize_ratio = min(safe_w / w, safe_h / h)
                w, h = int(w * resize_ratio), int(h * resize_ratio)
                image = image.resize((w, h), 1)
            elif h < int(H * 0.6):
                resize_ratio = int(H * 0.6) / h
                w, h = int(w * resize_ratio), int(h * resize_ratio)
                image = image.resize((w, h), 1)
            # 左右随机作为安全区
            pad_left = random.randint(0, safe_w - w) + int(W * safe_ratio) * random.randint(0, 1)
            pad_right = W - pad_left - w
            pad_top = random.randint(0, safe_h - h) + int(H * margin)
            pad_bottom = H - pad_top - h
    else:
        raise Exception(f"Unsupported ratio: {ratio}")

    out_image = pad_image(image, (pad_left, pad_right, pad_top, pad_bottom))
    out_url = save_tos(encode_pil_bytes(out_image, False), folder_name="gemini")
    return out_url


def remove_special_mark(input_str, special_chars = ('™', '®')):
    for char in special_chars:
        input_str = input_str.replace(char, '')
    return input_str


def main(args):
    # load dataset
    data_list = load_file(args.input_file)
    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        data_list = data_list[args.rank::args.num_ranks]

    out = []
    for line in tqdm(data_list):
        try:
            for img_url, img_tag, text_carousel in zip(line['image_urls'], line['image_tags'], line['text_carousels']):
                try:
                    # 1. 卖点文案
                    selling_text = {"product_name": remove_special_mark(text_carousel['product_name'])}
                    if len(text_carousel['primary_selling_points']) > 0:
                        selling_text["primary_selling_points"] = remove_special_mark(
                            text_carousel['primary_selling_points'][0])
                    if len(text_carousel['secondary_selling_points']) > 0:
                        selling_text["secondary_selling_points"] = remove_special_mark(
                            text_carousel['secondary_selling_points'][0])

                    # 2. 商品图
                    subject_url = img_tag['truncated_res']['subject_image_url']
                    image = load_or_download_image(subject_url)
                    # 9:16
                    for i in range(3):
                        try:
                            image_9_16_url = get_gemini_product_image(Image.fromarray(np.array(image)), ratio="9:16")
                            if isinstance(image_9_16_url, str) and image_9_16_url.startswith('http'):
                                break
                            time.sleep(0.1)
                        except Exception as e:
                            time.sleep(0.1)
                            if i == 2:
                                raise Exception(e)
                    # 1:1
                    for i in range(3):
                        try:
                            image_1_1_url = get_gemini_product_image(Image.fromarray(np.array(image)), ratio="1:1")
                            if isinstance(image_1_1_url, str) and image_1_1_url.startswith('http'):
                                break
                            time.sleep(0.1)
                        except Exception as e:
                            time.sleep(0.1)
                            if i == 2:
                                raise Exception(e)

                    out.append({
                        "ad_id": line["ad_id"],
                        "l3_vertical_tag": line["l3_vertical_tag"],
                        "ad_country": line["ad_country"],
                        "external_website_url": line["external_website_url"],
                        "source_url": img_url,
                        "subject_url": subject_url,
                        "product_name": selling_text.get('product_name', ''),
                        "primary_selling_points": selling_text.get('primary_selling_points', ''),
                        "secondary_selling_points": selling_text.get('secondary_selling_points', ''),
                        "image_9:16_url": image_9_16_url,
                        "image_1:1_url": image_1_1_url,
                    })

                    if len(out) % 100 == 0:
                        csv_save(out, args.output_file)
                except Exception as e:
                    print(img_url)
                    print(e)
        except Exception as e:
            print(line)
            print(e)

    # Final save
    csv_save(out, args.output_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
