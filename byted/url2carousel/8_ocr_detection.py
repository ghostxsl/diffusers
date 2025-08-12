# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
import time
from collections import Counter
import numpy as np
from tqdm import tqdm
import paddle
from paddleocr import PaddleOCR

from diffusers.data.utils import load_file, load_csv_or_xlsx_to_dict, json_save, remove_punctuation, full_to_half
from diffusers.data.outer_vos_tools import load_or_download_image, encode_pil_bytes
from diffusers.data.byted.clients.creative_ai_capability import image_subject_seg
from overpass_ad_creative_ai_capabilities.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.capability_thrift import ImageInfo


# python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
# python -m pip install paddleocr


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
        "--type",
        default="image",
        type=str,
        choices=["image", "url"],
    )
    parser.add_argument(
        "--text_detection_model_dir",
        default="/mnt/bn/creative-algo/xsl/models/PP-OCRv5_server_det_infer",
        type=str)
    parser.add_argument(
        "--text_recognition_model_dir",
        default="/mnt/bn/creative-algo/xsl/models/PP-OCRv5_server_rec_infer",
        type=str)

    parser.add_argument(
        "--rank",
        default=None,
        type=int)
    parser.add_argument(
        "--num_ranks",
        default=None,
        type=int)
    parser.add_argument(
        "--device",
        default='gpu',
        type=str,
        help="Device to use (e.g. cpu, gpu, gpu:0, etc.)")

    args = parser.parse_args()

    return args


def remove_subject_image(image_url, type="image"):
    if type == "image":
        image = load_or_download_image(image_url)
        image_info = ImageInfo(Binary=encode_pil_bytes(image, False))
        mask_url = (
            image_subject_seg(image_urls=[], image_infos=[image_info], only_mask=1, refine_mask=2)
            .success_image_infos[0]
            .URL
        )
    else:
        mask_url = image_subject_seg([image_url], only_mask=1, refine_mask=2).success_image_urls[0]

    mask = load_or_download_image(mask_url).convert('L')
    mask = np.float32(mask)[..., None] / 255

    image = load_or_download_image(image_url).convert('RGB')
    image = np.float32(image) * (1 - mask)
    return np.uint8(np.clip(image, 0, 255)), mask_url


def compare_word_and_char_bag(rec_text, gt_text):
    # 全角转半角
    rec_text = [full_to_half(a) for a in rec_text]
    gt_text = [full_to_half(a) for a in gt_text]

    # 移除所有标点符号
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
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_server_det",
        text_detection_model_dir=args.text_detection_model_dir,
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_recognition_model_dir=args.text_recognition_model_dir,
        use_doc_orientation_classify=False,  # Disables document orientation classification model via this parameter
        use_doc_unwarping=False,  # Disables text image rectification model via this parameter
        use_textline_orientation=False,  # Disables text line orientation classification model via this parameter
        text_det_limit_side_len=672,
        text_det_limit_type="max",
        text_det_thresh=0.3,
        text_det_box_thresh=0.6,
        text_det_unclip_ratio=1.2,
        text_rec_score_thresh=0.5,
        device=args.device,
    )

    # load dataset
    try:
        data_list = load_file(args.input_file)
    except Exception:
        data_list = load_csv_or_xlsx_to_dict(args.input_file)
    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        data_list = data_list[args.rank::args.num_ranks]

    out = []
    num_ocr_true = 0
    for line in tqdm(data_list):
        try:
            for i in range(3):
                try:
                    if args.type == "image":
                        gen_img, mask_url = remove_subject_image(line["generate_image"], type=args.type)
                    else:
                        gen_img, mask_url = remove_subject_image(line['generate_url'], type=args.type)
                    if isinstance(gen_img, np.ndarray):
                        break
                except Exception as e:
                    time.sleep(0.2)
                    if i == 2:
                        raise e

            with paddle.no_grad():
                result = ocr.predict(np.array(gen_img)[..., ::-1])

            product_name = line['product_name'] if isinstance(line['product_name'], str) else ''
            primary_selling_points = line['primary_selling_points'] if isinstance(line['primary_selling_points'], str) else ''
            secondary_selling_points = line['secondary_selling_points'] if isinstance(line['secondary_selling_points'], str) else ''
            gt_text = [product_name, primary_selling_points, secondary_selling_points]

            word_dict, char_dict = compare_word_and_char_bag(result[0]['rec_texts'], gt_text)
            line['generate_mask_url'] = mask_url
            line['ocr_rec_texts'] = result[0]['rec_texts']
            line['ocr_word'] = word_dict
            line['ocr_char'] = char_dict

            if line['ocr_char']['char_tag']:
                num_ocr_true += 1
            out.append(line)
        except Exception as e:
            print(line)
            print(e)

    # Final Save
    json_save(out, args.output_file)
    print(num_ocr_true)


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
