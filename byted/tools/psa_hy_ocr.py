# @author: wilson.xu.
import os
os.environ["TCE_PSM"] = "ad.creative.image_core_solution"
import re
import argparse
import json
import concurrent
from concurrent.futures import ThreadPoolExecutor

from overpass_ad_creative_qwen_image_v1.clients.rpc.ad_creative_qwen_image_v1 import AdCreativeQwen_Image_V1Client
from tqdm import tqdm

from diffusers.data.utils import load_file, json_save
from diffusers.data.outer_vos_tools import encode_pil_bytes, load_or_download_image


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="gemini3_image_trial_image_addon_hy_ocr.json", type=str)
    parser.add_argument(
        "--output_file", default="result_gemini3_image_trial_image_addon_hy_ocr.json", type=str)
    parser.add_argument(
        "--idc", default="maliva", type=str)
    parser.add_argument(
        "--max_workers", default=1, type=int)

    args = parser.parse_args()
    return args


def check_word_correct(custom_text, vllm_answer):
    """
    检测custom_text中的每个单词是否在vllm_answer中完整出现
    兼容：特殊字符(&、-)、中英文标点、连字符单词、大小写混合
    :param custom_text: 待检测的文本（单词以空格分隔）
    :param vllm_answer: 对照文本
    :return: 所有单词都出现则返回True，否则False
    """
    # 处理空输入
    if not custom_text.strip():
        return False

    # 1. 拆分custom_text为单词列表（处理多空格、首尾空格）
    custom_words = [word.strip() for word in custom_text.split() if word.strip()]
    if not custom_words:
        return False

    # 2. 统一转为小写（消除大小写干扰）
    vllm_lower = vllm_answer.lower()

    # 3. 遍历每个单词，检查是否完整存在（核心修复逻辑）
    for word in custom_words:
        word_lower = word.lower()
        # 转义单词中的特殊字符（-、&、.等）
        escaped_word = re.escape(word_lower)
        # 匹配规则：单词前后是“非单词字符”或文本首尾（兼容连字符/特殊符号）
        # \W 匹配非单词字符（等价于 [^a-zA-Z0-9_]），| 表示“或”，^/$ 表示文本首尾
        pattern = re.compile(r"(^|\W)" + escaped_word + r"(\W|$)")
        # 搜索匹配（忽略匹配到的非单词字符，只确认单词本身存在）
        match = pattern.search(vllm_lower)
        if not match:
            return False

    return True


def send_request(item):
    for k, v in item.items():
        if k == "id":
            continue
        text_str, gen_url = v
        if not isinstance(gen_url, str):
            continue
        img = load_or_download_image(gen_url)
        params= {
            "image_bytes": encode_pil_bytes(img),
            "prompt": "提取图片中的标语。"
        }
        code, msg, resp = client.AiModel(request_body=json.dumps(params))
        res_ocr = json.loads(resp.result_body)["ocr_result"]
        v.append(res_ocr)
        v.append([check_word_correct(t, res_ocr) for t in text_str])

    return item


def main(data, dst, max_workers):
    results = []
    error_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(send_request, item): item for item in data}
        with tqdm(total=len(data)) as pbar:
            for future in concurrent.futures.as_completed(future_to_url):
                item = future_to_url[future]
                try:
                    res_item = future.result()
                    pbar.update(1)  # Update progress bar

                    results.append(res_item)
                    if len(results) % 10 == 0:
                        json_save(results, dst)
                except Exception as e:
                    print(f"An error occurred for {e}")
                    error_results.append({"error_reason": str(e)})

    json_save(results, dst)
    print(error_results)


if __name__ == "__main__":
    args = parse_args()
    psm = "ad.creative.qwen_image_v1"
    cluster = "hy_ocr"
    print(f"psm: {psm}, cluster: {cluster}, idc: {args.idc}, max_workers: {args.max_workers}")
    client = AdCreativeQwen_Image_V1Client(psm=psm, cluster=cluster, idc=args.idc)

    data = load_file(args.input_file)

    main(data, args.output_file, args.max_workers)

    print('Done!')
