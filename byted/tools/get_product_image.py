# Copyright (c) wilson.xu. All rights reserved.
import os
os.environ["TCE_PSM"] = "ad.creative.image_core_solution"
from os.path import join, basename
import argparse
import time
from tqdm import tqdm

from diffusers.data.byted.tos import save_tos
from diffusers.data.outer_vos_tools import load_or_download_image, encode_pil_bytes
from diffusers.data.utils import load_csv_or_xlsx_to_dict, load_file, json_save, get_product_and_mask_image
from diffusers.data.byted.clients.creative_ai_capability import image_subject_seg


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--output_file",
        default="output.json",
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--save_image_dir",
        default=None,
        type=str)

    parser.add_argument(
        "--rank",
        default=None,
        type=int)
    parser.add_argument(
        "--num_ranks",
        default=None,
        type=int)

    args = parser.parse_args()

    os.makedirs(args.save_image_dir, exist_ok=True)

    return args


def main(args):
    try:
        data_list = load_csv_or_xlsx_to_dict(args.input_file)
    except Exception:
        data_list = load_file(args.input_file)
    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        data_list = data_list[args.rank::args.num_ranks]

    out = []
    for line in tqdm(data_list):
        try:
            poster_url = line['poster_url']
            mask_url = image_subject_seg(image_urls=[poster_url], only_mask=1, refine_mask=2).success_image_urls[0]

            poster_image = load_or_download_image(poster_url)
            mask = load_or_download_image(mask_url)
            product_image, mask_image = get_product_and_mask_image(poster_image, mask, 10)

            line['poster_mask_url'] = mask_url
            for i in range(3):
                product_url = save_tos(
                    encode_pil_bytes(product_image, to_string=False), folder_name="xsl")
                if product_url is None:
                    time.sleep(0.5)
                    continue
                else:
                    line['product_url'] = product_url
                    break

            # poster_image = poster_image.resize((928, 1664), 1)

            name = basename(poster_url)
            product_image.save(join(args.save_image_dir, name + "_product.jpg"))
            line["ref_image"] = join(args.save_image_dir, name + "_product.jpg")

            poster_image.save(join(args.save_image_dir, name + "_poster.jpg"))
            line["poster_image"] = join(args.save_image_dir, name + "_poster.jpg")
            out.append(line)

        except Exception as e:
            print(line, e)

    json_save(out, args.output_file)


if __name__ == "__main__":
    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"

    args = parse_args()
    main(args)
    print('Done!')
