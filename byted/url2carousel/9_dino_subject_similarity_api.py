# Copyright (c) wilson.xu. All rights reserved.
import os
os.environ["TCE_PSM"] = "ad.creative.image_core_solution"
import argparse
import json
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.utils import load_file, load_csv_or_xlsx_to_dict, json_save, get_product_and_mask_image
from diffusers.data.outer_vos_tools import load_or_download_image, encode_pil_bytes
from diffusers.data.byted.clients.creative_ai_capability import image_subject_seg
from overpass_ad_creative_qwen_image_v1.clients.rpc.ad_creative_qwen_image_v1 import AdCreativeQwen_Image_V1Client
from overpass_ad_creative_ai_capabilities.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.capability_thrift import ImageInfo


client = AdCreativeQwen_Image_V1Client(psm='ad.creative.psa', cluster='dino', idc='sg1')


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
    parser.add_argument("--subject_seg", action="store_true", default=False)
    parser.add_argument("--max_workers", default=1, type=int,)

    args = parser.parse_args()

    return args


def get_subject_image(image_path, mask_url=None):
    image = load_or_download_image(image_path)
    if mask_url is None:
        image_info = ImageInfo(Binary=encode_pil_bytes(image, False))
        mask_url = image_subject_seg(
            image_urls=[], image_infos=[image_info], only_mask=1, refine_mask=2).success_image_infos[0].URL

    mask = load_or_download_image(mask_url)
    product_image, _ = get_product_and_mask_image(image, mask, 10)
    return product_image


def send_request(item):
    if args.subject_seg:
        gt_image = load_or_download_image(item["product_url"])
        gen_image = load_or_download_image(item["generate_image"])
    else:
        gt_image = load_or_download_image(item["product_url"])
        gen_image = get_subject_image(item["generate_image"], item["generate_mask_url"])

    params = {
        "image_bytes_1": encode_pil_bytes(gt_image),
        "image_bytes_2": encode_pil_bytes(gen_image),
        "subject_seg": args.subject_seg,
    }
    code, msg, resp = client.AiModel(request_body=json.dumps(params))

    result = json.loads(resp.result_body)
    item["subject_similarity"] = result["subject_similarity"]
    item["subject_detail_similarity"] = result["subject_detail_similarity"]
    return item


def main(args):
    # load dataset
    try:
        data_list = load_csv_or_xlsx_to_dict(args.input_file)
    except Exception:
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
    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"

    args = parse_args()
    main(args)
    print('Done!')
