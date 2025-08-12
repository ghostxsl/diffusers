# Copyright (c) wilson.xu. All rights reserved.
import os
os.environ["TCE_PSM"] = "ad.creative.image_core_solution"
from os.path import join, basename, splitext
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image

from diffusers.data.byted.tos import save_tos
from diffusers.data.outer_vos_tools import load_or_download_image, encode_pil_bytes
from diffusers.data.utils import load_file, json_save, get_bbox_from_mask
from diffusers.data.byted.clients.creative_ai_capability import image_subject_seg

from overpass_ad_creative_image_core_solution.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.capability_thrift import ImageInfo


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


def crop_reference_image(image, mask_image, padding=10):
    mask_image = np.array(mask_image.convert('L'))
    bbox = get_bbox_from_mask(mask_image)

    x1, y1, x2, y2 = bbox
    x1 = max(x1 - padding, 0)
    y1 = max(y1 - padding, 0)
    x2 = min(x2 + padding, image.width)
    y2 = min(y2 + padding, image.height)
    out_img = Image.fromarray(np.array(image)[y1: y2, x1: x2])
    return out_img


def main(args):
    data_list = load_file(args.input_file)
    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        data_list = data_list[args.rank::args.num_ranks]

    out = []
    for line in tqdm(data_list):
        try:
            name = splitext(basename(line["ref_image"]))[0]

            ref_image = load_or_download_image(line["ref_image"])
            if "ref_mask_url" not in line:
                image_info = ImageInfo(Binary=encode_pil_bytes(ref_image, False))
                mask_url = image_subject_seg(image_urls=[], image_infos=[image_info], only_mask=1, refine_mask=2).success_image_infos[0].URL
                line["ref_mask_url"] = mask_url
            else:
                mask_url = line["ref_mask_url"]

            mask = load_or_download_image(mask_url)
            out_ref_img = crop_reference_image(ref_image, mask, 10)
            if "ref_url" not in line:
                ref_url = save_tos(encode_pil_bytes(out_ref_img, to_string=False), folder_name="xsl")
                line["ref_url"] = ref_url
            out_ref_img.save(join(args.save_image_dir, name + '_ref.jpg'))

            gen_image = load_or_download_image(line["gen_image"])
            gen_image.save(join(args.save_image_dir, name + '.jpg'))
            if "gen_url" not in line:
                gen_url = save_tos(encode_pil_bytes(gen_image, to_string=False), folder_name="xsl")
                line["gen_url"] = gen_url

            out.append(line)
            if len(out) % 10 == 0:
                json_save(out, args.output_file)

        except Exception as e:
            print(e)

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
