# @author: wilson.xu.
import os
os.environ["TCE_PSM"] = "ad.creative.image_core_solution"
import re
import string
from typing import Dict, List, Tuple
import argparse
import json
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.utils import load_file, json_save
from diffusers.data.outer_vos_tools import encode_pil_bytes, load_or_download_image
from overpass_ad_creative_qwen_image_v1.clients.rpc.ad_creative_qwen_image_v1 import AdCreativeQwen_Image_V1Client


# ===================== Core Configuration: Language Character Patterns =====================
LANGUAGE_PATTERNS = {
    # Alphabetic languages (English/Spanish/Portuguese/Indonesian/Malay/Tagalog/Vietnamese)
    "alphabetic": {
        "name": "Alphabetic (EN/ES/PT/ID/MS/VI/PH)",
        "pattern": re.compile(r"[a-zA-Z\u00C0-\u024F]+[a-zA-Z0-9\u00C0-\u024F]*"),
        "countries": ["USA", "Mexico", "Brazil", "Indonesia", "Malaysia", "Philippines", "Singapore (EN)", "Vietnam"],
    },
    # Chinese (Simplified/Traditional)
    "chinese": {
        "name": "Chinese",
        "pattern": re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uF900-\uFAFF]"),
        "countries": ["China", "Singapore (CN)"],
    },
    # Japanese (Kanji/Hiragana/Katakana)
    "japanese": {
        "name": "Japanese",
        "pattern": re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uF900-\uFAFF\u3040-\u309F\u30A0-\u30FF]"),
        "countries": ["Japan"],
    },
    # Thai
    "thai": {"name": "Thai", "pattern": re.compile(r"[\u0E00-\u0E7F]"), "countries": ["Thailand"]},
}


# Noise characters (common punctuation/symbols across target countries)
NOISE_CHARS = (
    # Basic punctuation
    string.punctuation + "，。！？：；"
    "''（）【】《》、*#%·…—±§¶©®™℃℉mlfl.ozgkg"
    # Full-width punctuation
    + "．，、：；？！“”‘’（）【】《》｛｝￥～·"
    # Country-specific noise symbols
    + "++ﾟ✅✓●■▲▼★☆➡️⬅️♀️♂️©️®️™️๑๒๓๔๕๖๗๘๙๐｀´¨ˆ˜˚˙¸"
    + "¡¿¢£¤¥¦§¨ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞß"
    + "₫฿¥$€£¢"
    + "️⃣"
)


def clean_multilingual_text(raw_text: str) -> str:
    """
    Clean multilingual OCR text: remove noise, standardize whitespace
    :param raw_text: Original OCR text
    :return: Cleaned standardized text
    """
    # 1. Unify line breaks/tabs/multi-spaces to single space
    cleaned = re.sub(r"\s+", " ", raw_text).strip()
    # 2. Remove all noise characters (keep valid language chars + spaces)
    cleaned = "".join([c for c in cleaned if c not in NOISE_CHARS])
    # 3. Convert full-width letters/numbers to half-width
    cleaned = re.sub(r"[\uff00-\uffff]", lambda c: chr(ord(c.group(0)) - 0xFEE0), cleaned)
    return cleaned


def count_multilingual_words_chars(
    raw_text: str, keep_brand_num: bool = True, deduplicate_alphabetic: bool = False
) -> Tuple[Dict, str]:
    """
    Count words (alphabetic) / characters (Chinese/Japanese/Thai) for multilingual OCR text
    :param raw_text: Original OCR text
    :param keep_brand_num: Keep brand/model numbers (e.g., 2RB, 3S) in alphabetic words
    :param deduplicate_alphabetic: Deduplicate alphabetic words
    :return: Statistics dict (all English keys) + cleaned text
    """
    # Step 1: Text cleaning
    cleaned_text = clean_multilingual_text(raw_text)

    # Step 2: Initialize statistics (ALL ENGLISH KEYS)
    stats = {
        "total_alphabetic_words": 0,  # Total alphabetic words (EN/ES/PT/ID/MS/VI/PH)
        "total_chinese_chars": 0,  # Total Chinese characters
        "total_japanese_chars": 0,  # Total Japanese characters (exclude duplicate Chinese)
        "total_thai_chars": 0,  # Total Thai characters
        "alphabetic_word_list": [],  # List of valid alphabetic words
        "chinese_char_list": [],  # List of Chinese characters
        "japanese_char_list": [],  # List of Japanese characters
        "thai_char_list": [],  # List of Thai characters
        "detected_languages": [],  # List of detected languages
        "config": {"keep_brand_numbers": keep_brand_num, "deduplicate_alphabetic_words": deduplicate_alphabetic},
        "total_words_chars": 0,  # Total count (words + characters)
    }

    # Step 3: Match alphabetic languages
    alpha_pattern = LANGUAGE_PATTERNS["alphabetic"]["pattern"]
    if not keep_brand_num:
        alpha_pattern = re.compile(r"[a-zA-Z\u00C0-\u024F]+")  # Exclude all numbers

    candidate_words = cleaned_text.split(" ")
    valid_alpha_words = [word for word in candidate_words if alpha_pattern.fullmatch(word)]

    # Deduplicate alphabetic words if needed
    if deduplicate_alphabetic:
        valid_alpha_words = list(set(valid_alpha_words))

    stats["total_alphabetic_words"] = len(valid_alpha_words)
    stats["alphabetic_word_list"] = valid_alpha_words
    if valid_alpha_words:
        stats["detected_languages"].append(LANGUAGE_PATTERNS["alphabetic"]["name"])

    # Step 4: Match Chinese
    cn_pattern = LANGUAGE_PATTERNS["chinese"]["pattern"]
    cn_chars = cn_pattern.findall(cleaned_text)
    stats["total_chinese_chars"] = len(cn_chars)
    stats["chinese_char_list"] = cn_chars
    if cn_chars:
        stats["detected_languages"].append(LANGUAGE_PATTERNS["chinese"]["name"])

    # Step 5: Match Japanese (exclude duplicate Chinese chars)
    jp_pattern = LANGUAGE_PATTERNS["japanese"]["pattern"]
    jp_chars = jp_pattern.findall(cleaned_text)
    jp_unique_chars = [c for c in jp_chars if c not in cn_chars]
    stats["total_japanese_chars"] = len(jp_unique_chars)
    stats["japanese_char_list"] = jp_unique_chars
    if jp_unique_chars:
        stats["detected_languages"].append(LANGUAGE_PATTERNS["japanese"]["name"])

    # Step 6: Match Thai
    th_pattern = LANGUAGE_PATTERNS["thai"]["pattern"]
    th_chars = th_pattern.findall(cleaned_text)
    stats["total_thai_chars"] = len(th_chars)
    stats["thai_char_list"] = th_chars
    if th_chars:
        stats["detected_languages"].append(LANGUAGE_PATTERNS["thai"]["name"])

    # Step 7: Calculate total count (words + characters)
    stats["total_words_chars"] = (
        stats["total_alphabetic_words"]
        + stats["total_chinese_chars"]
        + stats["total_japanese_chars"]
        + stats["total_thai_chars"]
    )

    return stats, cleaned_text


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="/mnt/bn/creative-algo/xsl/label_0121_psa_ingress_filtering_3.json", type=str)
    parser.add_argument(
        "--output_file", default="ocr_label_0121_psa_ingress_filtering_3.json", type=str)
    parser.add_argument(
        "--idc", default="sg1", type=str)
    parser.add_argument(
        "--max_workers", default=5, type=int)

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
    image_url = item["mainimageurl"]
    img = load_or_download_image(image_url)
    params= {
        "image_bytes": encode_pil_bytes(img),
        "prompt": "提取图中的文字。"
    }
    code, msg, resp = client.AiModel(request_body=json.dumps(params))
    res_ocr = json.loads(resp.result_body)["ocr_result"]
    item["ocr_result"] = res_ocr

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
                    if len(results) % 100 == 0:
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
