# Copyright (c) wilson.xu. All rights reserved.
import logging
from typing import List, Sequence, Tuple, Union, Dict
from collections import Counter
from retrying import retry
import re
import os
import json
import base64
from io import BytesIO
import numpy as np
from PIL import Image

import diffusers.data.byted.errno as err
from diffusers.data.byted.decorator import timer
from diffusers.data.outer_vos_tools import load_or_download_image
from diffusers.data.byted.clients.creative_ai_capability import image_subject_seg
from overpass_ad_creative_qwen_image_v1.clients.rpc.ad_creative_qwen_image_v1 import AdCreativeQwen_Image_V1Client
from overpass_ad_creative_ai_capabilities.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.capability_thrift import ImageInfo


# PaddleOCR
pp_client = AdCreativeQwen_Image_V1Client(psm="ad.creative.psa", cluster="ocr", idc="sg1", transport="ttheader")

# HunyuanOCR
hy_client = AdCreativeQwen_Image_V1Client(psm="ad.creative.qwen_image_v1", cluster="hy_ocr", idc="maliva", transport="ttheader")


def encode_pil_bytes(img, to_string=True, format="JPEG", quality=90):
    with BytesIO() as buf:
        img.save(buf, format=format, quality=quality)
        img_bytes = buf.getvalue()

    if to_string:
        return base64.b64encode(img_bytes).decode("utf-8")
    return img_bytes


@timer
def get_image_from_url_or_pil(image_url, image):
    if image_url is None and image is None:
        raise Exception("`image_url` and `image` cannot both be empty.")
    elif isinstance(image_url, str):
        image = load_or_download_image(image_url)
        # print(f"Download image URL: {image_url}")
        return image.convert("RGB"), image_url, None
    elif isinstance(image, Image.Image):
        image_bytes = encode_pil_bytes(image, False)
        return image.convert("RGB"), None, image_bytes
    else:
        raise Exception(f"Wrong type: image_url({type(image_url)}), image_bytes({type(image)})")


@timer
def remove_subject_image(image, image_url, image_bytes):
    if image_url is None and image_bytes is None:
        raise Exception("`image_url` and `image_bytes` cannot both be empty.")
    elif isinstance(image_url, str):
        mask_url = image_subject_seg([image_url], only_mask=1, refine_mask=2).success_image_urls[0]
    elif isinstance(image_bytes, bytes):
        image_info = ImageInfo(Binary=image_bytes)
        mask_url = image_subject_seg(image_urls=[], image_infos=[image_info], only_mask=1, refine_mask=2).success_image_infos[0].URL
    else:
        raise Exception(f"Wrong type: image_url({type(image_url)}), image_bytes({type(image_bytes)})")

    mask = load_or_download_image(mask_url).convert("L")
    # print(f"Mask URL: {mask_url}")

    mask = np.array(mask, dtype=np.float32)[..., None] / 255
    image = np.array(image, dtype=np.float32) * (1 - mask)
    return Image.fromarray(np.uint8(np.clip(image, 0, 255))), mask_url


def get_ocr_image(image_url, image, remove_subject=True):
    img, image_url, image_bytes = get_image_from_url_or_pil(image_url, image)
    if remove_subject:
        try:
            ocr_img, _ = remove_subject_image(img, image_url, image_bytes)
        except err.WithCodeError as e:
            if e.get_status_code() == err.ErrCodeNoBackgroundImage:
                print(f"[image_subject_seg] failed, code: {err.ErrCodeNoBackgroundImage}, msg: Input Image Has No Background.")
                ocr_img = img
            else:
                raise err.WithCodeError(
                    err.ErrorCodeSegmentError,
                    f"[image_subject_seg] failed, code: {err.ErrorCodeSegmentError}, msg: {e.message}",
                )
    else:
        ocr_img = img

    return ocr_img


@timer
@retry(stop_max_attempt_number=1, wait_fixed=500)
def ocr_image_text_detection(image, text=""):
    code, msg, resp = pp_client.AiModel(
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


@timer
def validate_ocr_quality_control(
    image_url: str = None,
    image: Image.Image = None,
    check_text: List[str] = [""],
    remove_subject: bool = True,
    src_image_url: str = None,
    src_image: Image.Image = None,
    return_raw_data: bool = False,
) -> Union[bool, Tuple[bool, dict]]:
    r"""
    OCR Quality Control Service Interface
    Recommended for cases where there is no text in the original image.

    Args:
        image_url (str):
            The URL of the generated image to be detected.
        image (PIL.Image.Image):
            The generated image to be detected (in PIL format), has lower priority than `image_url`.
        check_text (List[str]):
            Text to be detected on the image.
        remove_subject (bool, defaults to `True`):
            Whether to remove the main subject in the image, the default value is True.
            In some cases, the text present on the subject may affect OCR detection.
        src_image_url (str):
            The URL of the original image.
        src_image (PIL.Image.Image):
            The original image (in PIL format), has lower priority than `src_image_url`.
        return_raw_data (bool):
            Whether to return the raw OCR result data along with the boolean check result.

    Returns:
        (bool): `True` if it passes the OCR quality control; otherwise, it is `False`.
        If return_raw_data is True, returns Tuple[bool, dict].
    """
    assert isinstance(check_text, Sequence)

    if src_image_url is not None or src_image is not None:
        src_ocr_img = get_ocr_image(src_image_url, src_image, False)
        result = ocr_image_text_detection(src_ocr_img)
        ocr_img = get_ocr_image(image_url, image, False)
        result = ocr_image_text_detection(ocr_img, text=";".join(check_text + result["ocr_text"]))
        if ocr_img:
            result["image_size"] = ocr_img.size

        if not result["result"]:
            src_ocr_img = get_ocr_image(None, src_ocr_img, True)
            result = ocr_image_text_detection(src_ocr_img)
            ocr_img = get_ocr_image(None, ocr_img, True)
            result = ocr_image_text_detection(ocr_img, text=";".join(check_text + result["ocr_text"]))
            if ocr_img:
                result["image_size"] = ocr_img.size
    else:
        ocr_img = get_ocr_image(image_url, image, remove_subject)
        result = ocr_image_text_detection(ocr_img, text=";".join(check_text))
        if ocr_img:
            result["image_size"] = ocr_img.size

    if return_raw_data:
        return result["result"], result
    return result["result"]


@timer
def hy_ocr_image_text(image, prompt="提取图片中的标语。"):
    params = {"image_bytes": encode_pil_bytes(image), "prompt": prompt}
    code, msg, resp = hy_client.AiModel(request_body=json.dumps(params))
    if code == 0:
        result = json.loads(resp.result_body)
        if result["StatusCode"] == 0:
            logging.info(f"HY OCR result: {resp.result_body}")
            return result["ocr_result"]
        else:
            raise Exception(f"HY OCR detection error, code: {result['StatusCode']}, msg: {result['StatusMessage']}")
    else:
        raise Exception(f"Service[HY OCR] error, code: {code}, msg: {msg}")


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


@timer
def validate_addon_ocr(
    image_url: str = None,
    image: Image.Image = None,
    check_text: List[str] = [""],
    prompt: str = "提取图片中的标语。",
) -> bool:
    assert isinstance(check_text, Sequence)

    img, _, _ = get_image_from_url_or_pil(image_url, image)
    ocr_result = hy_ocr_image_text(img, prompt)

    check_text = " ".join(check_text)
    is_correct = check_word_correct(check_text, ocr_result)

    return is_correct


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
    return True, hay[:idx] + ("#" * len(needle)) + hay[idx + len(needle) :]


def build_add_tokens(origin_text: List[str], generate_text: List[str]) -> List[str]:
    """
    新增token序列 = generate_tokens 按次数扣掉 origin_tokens 后剩下的token（保持生成顺序）
    """
    gen_tokens = _tokenize_by_separators(generate_text)
    if not gen_tokens:
        return []

    origin_tokens = _tokenize_by_separators(origin_text)
    if not origin_tokens:  # 原图没OCR到任何token => 直接认为生成侧全是新增
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


def validate_ocr_quality_control_v2(
    gen_image_url: str = None,
    gen_image: Image.Image = None,
    src_image_url: str = None,
    src_image: Image.Image = None,
    check_text_list: List[str] = [],
) -> Tuple[bool, List[str], Dict]:
    assert isinstance(check_text_list, Sequence)
    # 1. 生成图OCR结果
    gen_ocr_img = get_ocr_image(gen_image_url, gen_image, False)
    result_gen = ocr_image_text_detection(gen_ocr_img)
    generate_text = result_gen["ocr_text"]
    # 2. 原图OCR结果
    if not src_image_url and not src_image:
        origin_text = []
    else:
        src_ocr_img = get_ocr_image(src_image_url, src_image, False)
        result_src = ocr_image_text_detection(src_ocr_img)
        origin_text = result_src["ocr_text"]
    # 3. 计算新增文本
    add_tokens = build_add_tokens(origin_text, generate_text)
    if not check_text_list:
        return False, add_tokens, result_gen
    # 4. 判断是否与check_text_list匹配
    ok = all(check_custom_phrase_in_add(c, add_tokens) for c in check_text_list)
    return ok, add_tokens, result_gen


if __name__ == "__main__":
    from tqdm import tqdm

    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"
    os.environ["TCE_PSM"] = "ad.creative.image_core_solution"

    def json_save(obj, file):
        with open(file, "w") as f:
            json.dump(obj, f)

    def json_load(file):
        with open(file, "r") as f:
            out = json.load(f)
        return out

    data = json_load("gemini_test_1017_us_other.json")

    out = []
    for line in tqdm(data):
        try:
            res = validate_ocr_quality_control(
                image_url=line["res_url"],
                check_text=[line["text"]],
                src_image_url=line["url"],
                remove_subject=False,
            )
            if res:
                out.append(line)
                if len(out) % 10 == 0:
                    json_save(out, "result_us.json")
        except Exception as e:
            print(e)

    json_save(out, "result_us.json")
    print("Done!")
