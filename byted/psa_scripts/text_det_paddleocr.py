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
        "--input_file", default="/mlx_devbox/users/xushangliang/playground/res_psa.csv", type=str)
    parser.add_argument(
        "--output_file", default="ocr_res_psa.json", type=str)
    parser.add_argument(
        "--idc", default="sg1", type=str)
    parser.add_argument(
        "--max_workers", default=1, type=int)

    args = parser.parse_args()
    return args


import re
from collections import Counter
from typing import List, Tuple


def _tokenize_by_separators(text_list: List[str]) -> List[str]:
    """
    严格token：
    - 小写（如果你连大小写都要严格，可去掉lower）
    - 非中英数中文都当分隔符
    """
    out = []
    for s in text_list or []:
        s = (s or "").lower()
        s = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            out.extend([t for t in s.split(" ") if t])
    return out


def _compact(s: str) -> str:
    """
    严格compact：
    - 小写
    - 去空白
    - 去掉非中英数中文
    """
    s = (s or "").lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", s)
    return s


def _consume_once(hay: str, needle: str) -> Tuple[bool, str]:
    idx = hay.find(needle)
    if idx == -1:
        return False, hay
    return True, hay[:idx] + ("#" * len(needle)) + hay[idx + len(needle):]


def build_add_tokens(origin_text: List[str], generate_text: List[str]) -> List[str]:
    """
    新增token序列 = generate_tokens 按次数扣掉 origin_tokens 后剩下的token（保持生成顺序）
    """
    gen_tokens = _tokenize_by_separators(generate_text)
    if not gen_tokens:
        return []

    origin_tokens = _tokenize_by_separators(origin_text)
    if not origin_tokens:          # 原图没OCR到任何token => 直接认为生成侧全是新增
        return gen_tokens

    origin_cnt = Counter(origin_tokens)
    add_tokens = []
    for t in gen_tokens:
        if origin_cnt.get(t, 0) > 0:
            origin_cnt[t] -= 1
        else:
            add_tokens.append(t)
    return add_tokens


def check_custom_phrase_in_add(custom_text: str, add_tokens: List[str]) -> bool:
    """
    严格校验一条custom phrase：
    1) 先用token计数严格匹配（最可靠）
    2) 若失败，再用compact子串consume兜底（解决OCR粘连）
       注意：这是“严格字符一致”的子串匹配，不做任何容错。
    """
    words = [w for w in (custom_text or "").split() if w.strip()]
    words = [_compact(w) for w in words]
    words = [w for w in words if w]
    if not words:
        return False

    need = Counter(words)

    # (1) token级严格次数匹配
    add_cnt = Counter([_compact(t) for t in add_tokens])
    if all(add_cnt.get(w, 0) >= k for w, k in need.items()):
        return True

    # (2) 粘连兜底：在新增token拼成的compact串里严格consume
    hay = _compact(" ".join(add_tokens))
    for w in sorted(need.keys(), key=len, reverse=True):
        for _ in range(need[w]):
            ok, hay = _consume_once(hay, w)
            if not ok:
                return False
    return True


def gate_all(
    origin_text: List[str],
    generate_text: List[str],
    custom_text_list: List[str],
) -> Tuple[bool, List[str]]:
    """
    AND规则：custom_text_list 每条都必须命中。
    返回 (pass_or_not, add_tokens)
    """
    add_tokens = build_add_tokens(origin_text, generate_text)

    if not custom_text_list:
        return False, add_tokens

    ok = all(check_custom_phrase_in_add(c, add_tokens) for c in custom_text_list)
    return ok, add_tokens


def ocr_image_text_detection(image, text="", remove_subject=False):
    code, msg, resp = client.AiModel(
        request_body=json.dumps(
            {
                "image_bytes": encode_pil_bytes(image),
                "check_text": text,
                "remove_subject": remove_subject,
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
    selling_points = item[2]

    image_url = item[0]
    img = load_or_download_image(image_url)
    res_ori = ocr_image_text_detection(img)

    image_url = item[1]
    img = load_or_download_image(image_url)
    res_gen = ocr_image_text_detection(img)

    for i in range(len(selling_points)):
        check_result, add_text_list = gate_all(res_ori["ocr_text"], res_gen["ocr_text"], selling_points[i:i+1])
        if check_result:
            break

    if not check_result:
        print("=" * 10)
        print(add_text_list)
        print(selling_points)
        print("=" * 10)

    item.append(res_ori["ocr_text"])
    item.append(res_gen["ocr_text"])
    item.append(add_text_list)
    item.append(check_result)

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
    out = []
    for item in data:
        gen_urls = json.loads(item['gen_urls'])
        if len(gen_urls) == 0:
            continue
        psa_info = json.loads(json.loads(item['psa_info'])[0])
        if psa_info["strategy"] == "GeminiImage":
            out.append([psa_info["origin_url"], gen_urls[0], psa_info["selling_points"]])
    main(out, args.output_file, args.max_workers)

    print('Done!')
