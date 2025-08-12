import os
os.environ["TCE_PSM"] = "ad.creative.image_core_solution"
import argparse
import json
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.utils import csv_save, load_csv_or_xlsx_to_dict
from diffusers.data.byted.clients.creative_data_forge import get_product_info
from diffusers.data.byted.clients.ad_creative_text_solution import generate_psa_sellings_points


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="0121_psa_ingress_filtering_1.csv", type=str)
    parser.add_argument(
        "--output_file", default="0121_psa_ingress_filtering_2.csv", type=str)
    parser.add_argument(
        "--num_workers", default=10, type=int)

    args = parser.parse_args()
    return args


language_map = {
    "ID": "Indonesian",
    "MY": "Malay",
    "PH": "Filipino",
    "TH": "Thai",
    "VN": "Vietnamese",
}


def send_request(line):
    spu_id = line["spu_id"]
    # 获取商品信息
    product_info = get_product_info(product_key=[str(spu_id)], key_type=0)
    line["language"] = product_info[0].language
    line["productRegion"] = product_info[0].productRegion
    line["product_description"] = product_info[0].description
    line["product_name"] = product_info[0].productName
    line["brand_name"] = product_info[0].brand

    # 获得卖点
    req = generate_psa_sellings_points(
        product_id=int(spu_id),
        target_language=language_map.get(line["productRegion"], "English"),
        strategy_version="2002",
        source="TTAM/ImageGenerationAutomationDelivery/PSACarouselAtomGeneration",
        product_description=line["product_description"],
        product_name=line["product_name"],
        brand_name=line["brand_name"],
    )
    if not req.selling_points:
        raise Exception(f"selling_points is None ({req.selling_points})")
    line["selling_points"] = json.dumps(req.selling_points)
    return line


if __name__ == "__main__":
    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"

    args = parse_args()
    print(f"input_file: {args.input_file}, output_file: {args.output_file}, num_workers: {args.num_workers}")

    data = load_csv_or_xlsx_to_dict(args.input_file)

    if os.path.exists(args.output_file):
        done_data = load_csv_or_xlsx_to_dict(args.output_file)
        done_data = {a['spu_id']: a for a in done_data}
        data = [a for a in data if a['spu_id'] not in done_data]
        results = list(done_data.values())
    else:
        results = []

    error_results = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_url = {executor.submit(send_request, item): item for item in data}
        with tqdm(total=len(data)) as pbar:
            for future in concurrent.futures.as_completed(future_to_url):
                item = future_to_url[future]
                try:
                    res_item = future.result()
                    pbar.update(1)  # Update progress bar
                    results.append(res_item)
                    if len(results) % 100 == 0:
                        csv_save(results, args.output_file)
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    error_results.append({'image_url': item, 'error_reason': str(e)})

    csv_save(results, args.output_file)
    print(f"error num: {len(error_results)}")
    print("Done")
