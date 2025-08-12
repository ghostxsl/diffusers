# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
import time
from collections import Counter
import numpy as np
from tqdm import tqdm
from PIL import Image

from diffusers.data.utils import load_file, json_save, remove_punctuation
from diffusers.data.outer_vos_tools import load_or_download_image, encode_pil_bytes
from diffusers.data.byted.clients.creative_ai_capability import image_subject_seg
from diffusers.data.byted.clients.creative_ai_capability import image_ocr_to_bbox
from overpass_ad_creative_ai_capabilities.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.capability_thrift import ImageInfo


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
        "--rank",
        default=None,
        type=int)
    parser.add_argument(
        "--num_ranks",
        default=None,
        type=int)

    args = parser.parse_args()

    return args


def remove_subject_image(image_url):
    for i in range(3):
        try:
            mask_url = image_subject_seg(
                image_urls=[image_url], only_mask=1, refine_mask=2).success_image_urls[0]
            if isinstance(mask_url, str):
                break
        except Exception as e:
            if i == 2:
                raise Exception(e)

    mask = load_or_download_image(mask_url).convert('L')
    mask = np.float32(mask)[..., None] / 255

    image = load_or_download_image(image_url).convert('RGB')
    image = np.float32(image) * (1 - mask)
    return np.uint8(np.clip(image, 0, 255))


def compare_word_and_char_bag(rec_text, gt_text):
    rec_text = [remove_punctuation(a).lower() for a in rec_text]
    gt_text = [remove_punctuation(a).lower() for a in gt_text]

    # 1. word
    rec_words = [a for a in " ".join(rec_text).split(" ") if len(a) > 0]
    rec_words = Counter(rec_words)
    gt_words = [a for a in " ".join(gt_text).split(" ") if len(a) > 0]
    gt_words = Counter(gt_words)

    word_dict = {'word_tag': (rec_words == gt_words)}
    rec_words.subtract(gt_words)
    word_dict.update({k: v for k, v in dict(rec_words).items() if v != 0})

    # 2. char
    rec_chars = "".join(rec_text).replace(" ", "")
    rec_chars = Counter(rec_chars)
    gt_chars = "".join(gt_text).replace(" ", "")
    gt_chars = Counter(gt_chars)

    char_dict = {'char_tag': (rec_chars == gt_chars)}
    rec_chars.subtract(gt_chars)
    char_dict.update({k: v for k, v in dict(rec_chars).items() if v != 0})

    return word_dict, char_dict


def main(args):
    # load dataset
    data_list = load_file(args.input_file)
    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        data_list = data_list[args.rank::args.num_ranks]

    out = []
    num_gpt_passed_result = 0
    for line in tqdm(data_list):
        try:
            if not line['result_quality_inspection']['overall_evaluation_result']:
                continue

            num_gpt_passed_result += 1
            gen_img = remove_subject_image(line['generate_url'])
            image_info = ImageInfo(Binary=encode_pil_bytes(Image.fromarray(gen_img), False))

            for i in range(3):
                result = image_ocr_to_bbox(image_infos=[image_info])[0]
                if len(result) == 0:
                    time.sleep(0.2)
                    continue
                else:
                    break
            rec_texts = [a['text'] for a in result]

            product_name = line['product_name']
            primary_selling_points = line['primary_selling_points']
            secondary_selling_points = line['secondary_selling_points']
            gt_text = [product_name, primary_selling_points, secondary_selling_points]

            word_dict, char_dict = compare_word_and_char_bag(rec_texts, gt_text)
            line['ocr_v2_rec_texts'] = rec_texts
            line['ocr_v2_word'] = word_dict
            line['ocr_v2_char'] = char_dict

            out.append(line)
        except Exception as e:
            print(line)
            print(e)

    # Final Save
    json_save(out, args.output_file)
    print(num_gpt_passed_result)


if __name__ == "__main__":
    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"
    os.environ["TCE_PSM"] = "ad.creative.image_core_solution"

    args = parse_args()
    main(args)
    print('Done!')
