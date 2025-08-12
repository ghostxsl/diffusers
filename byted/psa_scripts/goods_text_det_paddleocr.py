# @author: wilson.xu.
import os
os.environ["TCE_PSM"] = "ad.creative.image_core_solution"
import argparse
import json
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.utils import load_file, json_save, load_csv_or_xlsx_to_dict
from diffusers.data.outer_vos_tools import encode_pil_bytes, load_or_download_image
from overpass_ad_creative_qwen_image_v1.clients.rpc.ad_creative_qwen_image_v1 import AdCreativeQwen_Image_V1Client


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="calalogbg20260315.csv", type=str)
    parser.add_argument(
        "--output_file", default="ocr_calalogbg20260315.json", type=str)
    parser.add_argument(
        "--idc", default="sg1", type=str)
    parser.add_argument(
        "--max_workers", default=1, type=int)

    args = parser.parse_args()
    return args


import re
from collections import Counter
from difflib import SequenceMatcher
from typing import List


def validate_ocr_texts(origin_text: List[str], generate_text: List[str]) -> bool:
    def norm(s: str) -> str:
        return re.sub(r"[^0-9a-z]+", "", s.lower())

    def tokenize(s: str) -> List[str]:
        s_low = s.lower()
        parts = [p for p in re.split(r"[^0-9a-z]+", s_low) if p]
        glued = norm(s)
        tokens = []
        tokens.extend(parts)
        if glued and glued not in tokens:
            tokens.append(glued)
        return tokens

    def sim(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    origin_tokens = []
    for s in origin_text:
        origin_tokens.extend(tokenize(s))
    gen_tokens = []
    for s in generate_text:
        gen_tokens.extend(tokenize(s))

    origin_tokens = [t for t in origin_tokens if t]
    gen_tokens = [t for t in gen_tokens if t]

    # 精确匹配优先（多重集合消耗）
    gen_counter = Counter(gen_tokens)
    missing = []
    for t in origin_tokens:
        if gen_counter[t] > 0:
            gen_counter[t] -= 1
        else:
            missing.append(t)

    if not missing:
        return True

    # 剩余 token：只允许“生成侧更长/粘连”覆盖 origin token，即 tok in g
    leftover_gen = list(gen_counter.elements())
    used = [False] * len(leftover_gen)

    def try_consume_for(tok: str) -> bool:
        # a) 覆盖匹配：origin token 是生成 token 的子串（处理生成侧粘连变长）
        for i, g in enumerate(leftover_gen):
            if used[i]:
                continue
            if tok in g:
                used[i] = True
                return True

        # b) 相似度匹配：仅在长度相近时才允许，避免“短词”误吸收“长词”
        best_i = -1
        best = 0.0
        for i, g in enumerate(leftover_gen):
            if used[i]:
                continue
            # 长度差太大直接跳过
            if min(len(tok), len(g)) == 0:
                continue
            if abs(len(tok) - len(g)) / max(len(tok), len(g)) > 0.2:
                continue
            r = sim(tok, g)
            if r > best:
                best = r
                best_i = i

        if best_i != -1 and best >= 0.90:
            used[best_i] = True
            return True

        return False

    for t in missing:
        if not try_consume_for(t):
            return False

    return True


def ocr_image_text_detection(image, text=""):
    code, msg, resp = client.AiModel(
        request_body=json.dumps(
            {
                "image_bytes": encode_pil_bytes(image),
                "check_text": text,
            }
        )
    )
    if code == 0:
        result = json.loads(resp.result_body)
        if result["StatusCode"] == 0:
            return result
        else:
            raise Exception(f"OCR detection error, code: {result['StatusCode']}, msg: {result['StatusMessage']}")
    else:
        raise Exception(f"Service[OCR] error, code: {code}, msg: {msg}")


def send_request(item):
    ori_url = item["url_1"]
    img = load_or_download_image(ori_url)
    res_ori = ocr_image_text_detection(img)

    gen_url = item["url_2"]
    img = load_or_download_image(gen_url)
    res_gen = ocr_image_text_detection(img)

    check_result = validate_ocr_texts(res_ori["ocr_text"], res_gen["ocr_text"])

    item["url_1_ocr"] = res_ori["ocr_text"]
    item["url_2_ocr"] = res_gen["ocr_text"]
    item["check_result"] = check_result

    return item


def main(data, dst, max_workers):
    results = []
    error_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(send_request, item): item for item in data}
        with tqdm(total=len(data)) as pbar:
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    res_item = future.result()
                    pbar.update(1)  # Update progress bar

                    results.append(res_item)
                    if len(results) % 100 == 0:
                        json_save(results, dst)
                except Exception as e:
                    print(f"An error occurred for {e}")
                    error_results.append({"error_reason": str(e)})

    json_save(results, dst)
    print(len(error_results))


if __name__ == "__main__":
    args = parse_args()
    psm = "ad.creative.psa"
    cluster = "ocr"
    print(f"psm: {psm}, cluster: {cluster}, idc: {args.idc}, max_workers: {args.max_workers}")
    client = AdCreativeQwen_Image_V1Client(psm=psm, cluster=cluster, idc=args.idc, transport="ttheader")

    data = load_csv_or_xlsx_to_dict(args.input_file)
    main(data, args.output_file, args.max_workers)

    print('Done!')
