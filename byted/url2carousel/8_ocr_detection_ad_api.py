# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
import json
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.utils import load_file, json_save
from diffusers.data.outer_vos_tools import load_or_download_image, encode_pil_bytes
from overpass_ad_creative_qwen_image_v1.clients.rpc.ad_creative_qwen_image_v1 import AdCreativeQwen_Image_V1Client


client = AdCreativeQwen_Image_V1Client(psm='ad.creative.psa', cluster='ocr', idc='sg1')


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file",
        default=None, type=str, help="Path to image list on vos.")
    parser.add_argument(
        "--output_file",
        default="output.json", type=str, help="Path to image list on vos.")
    parser.add_argument(
        "--type",
        default="image", type=str, choices=["image", "url"])
    parser.add_argument(
        "--src_image", action="store_true", default=False)
    parser.add_argument(
        "--max_workers", default=1, type=int)

    args = parser.parse_args()

    return args


def send_request(item):
    if args.src_image:
        params = {
            "image_url": item["image_url"],
            "check_text": "",
        }
        code, msg, resp = client.AiModel(request_body=json.dumps(params))
        res_src = json.loads(resp.result_body)
        gt_text = [item["text"]] + res_src["ocr_text"]

        if args.type == "image":
            gen_img = load_or_download_image(item["generate_image"])
            params = {
                "image_bytes": encode_pil_bytes(gen_img),
                "check_text": ";".join(gt_text),
                "remove_subject": False,
            }
        else:
            params = {
                "image_url": item["generate_url"],
                "check_text": ";".join(gt_text),
                "remove_subject": False,
            }

        code, msg, resp = client.AiModel(request_body=json.dumps(params))
        result = json.loads(resp.result_body)
    else:
        product_name = item["product_name"]
        primary_selling_points = item["primary_selling_points"]
        secondary_selling_points = item["secondary_selling_points"]
        gt_text = [product_name, primary_selling_points, secondary_selling_points]

        if args.type == "image":
            gen_img = load_or_download_image(item["generate_image"])
            params = {
                "image_bytes": encode_pil_bytes(gen_img),
                "check_text": ";".join(gt_text),
                "remove_subject": True,
            }
        else:
            params = {
                "image_url": item["generate_url"],
                "check_text": ";".join(gt_text),
                "remove_subject": True,
            }

        code, msg, resp = client.AiModel(request_body=json.dumps(params))
        result = json.loads(resp.result_body)

    item["generate_mask_url"] = result["mask_url"]
    item["ocr_rec_texts"] = result["ocr_text"]
    item["ocr_word"] = result["word_check"]
    item["ocr_char"] = result["char_check"]
    return item


def main(args):
    # load dataset
    data_list = load_file(args.input_file)

    results = []
    error_results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_url = {executor.submit(send_request, item): item for item in data_list}
        with tqdm(total=len(data_list)) as pbar:
            for future in concurrent.futures.as_completed(future_to_url):
                item = future_to_url[future]
                try:
                    res_item = future.result()
                    pbar.update(1)  # Update progress bar

                    results.append(res_item)
                    if len(results) % 10 == 0:
                        json_save(results, args.output_file)
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    error_results.append({"image_url": item, "error_reason": str(e)})
    # Final Save
    json_save(results, args.output_file)
    print(len(error_results))


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
