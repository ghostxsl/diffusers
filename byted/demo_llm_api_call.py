import re
import json
import time
import base64
import logging
import traceback
import requests
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps
import openai

# from azure_mllm import MLLMClient


HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
}


def encode_pil_bytes(img, to_string=True, format='JPEG', quality=90):
    with BytesIO() as buf:
        img.save(buf, format=format, quality=quality)
        img_bytes = buf.getvalue()

    if to_string:
        return base64.b64encode(img_bytes).decode('utf-8')
    return img_bytes


def decode_pil_bytes(img_bytes, from_string=True):
    if from_string:
        img_bytes = base64.b64decode(img_bytes)
    return Image.open(BytesIO(img_bytes))


def download_pil_image(url, retry_times=3):
    for i in range(retry_times):
        try:
            content = requests.get(url, timeout=6, headers=HEADERS).content
            image = Image.open(BytesIO(content))
            return image
        except Exception:
            if i == retry_times - 1:
                raise Exception(traceback.format_exc())
            else:
                time.sleep(0.3)


def load_or_download_image(image_path):
    if isinstance(image_path, str):
        image = download_pil_image(image_path) if image_path.startswith('http') else Image.open(image_path)
    elif isinstance(image_path, Image.Image):
        image = image_path
    else:
        raise ValueError(
            "Incorrect format used for image. Should be a local path to an image, or a PIL image."
        )

    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA":
        # returning an RGB mode image with no transparency
        image = Image.fromarray(np.array(image)[..., :3])
    elif image.mode != "RGB":
        # Fix UserWarning for palette images with transparency
        if "transparency" in image.info:
            image = image.convert("RGBA")
            image = Image.fromarray(np.array(image)[..., :3])
        image = image.convert("RGB")

    return image


def _process_json_response(result, model_name="gemini-2.5-flash"):
    # first, try to extract JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(.*?)```", result, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # if no code block, strip possible markdown delimiters
        json_str = result.strip("```json\n").strip("\n```")

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.warning(f"mllm `{model_name}` initial JSON parsing failed: {e}")
        fixed_result = MLLMClient(
            base_url="https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl",
            model_name="gpt-5-mini-2025-08-07",
            api_key="HNLIrxnEf4bcWW7WyMomdHmBWpNDFJ8p_GPT_AK",
        ).make_raw_request(
            "",
            f"Fix this JSON string's formatting issues to make it valid: {result}, without changing its content. Only return the corrected JSON string with no explanations or additional text. Failing to provide a valid JSON or altering the content will result in severe consequences.",
            max_tokens=2000,
        )
        try:
            return json.loads(fixed_result)
        except json.JSONDecodeError:
            logging.warning(f"`{model_name}` final JSON parsing failed, resp: {fixed_result}")
            raise Exception(f"`{model_name}` final JSON parsing failed, resp: {fixed_result}")


def mllm_make_image_request(
    prompt="",
    image_urls=[],
    image_pils=[],
    thinking_budget=1024,
    max_tokens=4096,
    timeout=60,
    temperature=0.2,
    is_json_response=False,
    is_download_image=False,
    logid="",
    model_name="gemini-2.5-flash",
    api_key="lV9PRXdcOPUV8AgbdrTUtf8E9B0r68Qc_GPT_AK",
    base_url="https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/multimodal/crawl",
):
    try:
        # 0. 创建client
        client = openai.AzureOpenAI(api_key=api_key, azure_endpoint=base_url, api_version="2024-03-01-preview")
        # 1. 组装message
        message = {"role": "user", "content": [{"type": "text", "text": prompt}]}
        if image_urls:
            for image_url in image_urls:
                if is_download_image:
                    image = load_or_download_image(image_url)
                    message["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_bytes(image)}"}})
                else:
                    message["content"].append({"type": "image_url", "image_url": {"url": image_url}})
        elif image_pils:
            for image in image_pils:
                message["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_bytes(image)}"}})
        else:
            raise Exception(f"[{model_name}] Parameters `image_urls` and `image_pils` are both empty")
        # 2. 设置extra_body
        if "gemini" in model_name:
            extra_body = {"thinking": {"budget_tokens": thinking_budget, "include_thoughts": False}}
        else:
            extra_body = None
            temperature = 1.0
        # 3. 请求大模型
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                message,
            ],
            stream=False,
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            extra_body=extra_body,
            extra_headers={"X-TT-LOGID": logid},
        )
        result = json.loads(response.model_dump_json())
        message = result["choices"][0].get("message", {})
        content = message.get("content", "")

        if is_json_response:
            content = _process_json_response(content, model_name)

        return content
    except Exception as e:
        raise e


def gemini_image_generate(
    prompt,
    image_urls=[],
    image_pils=[],
    specify_gen_ratio=True,
    ratio="9:16",
    imageSize="1k",
    max_token=3800,
    thinking_budget=0,
    logid="",
    num_retry=2,
    model_name="gemini-2.5-flash-image",
    ak="lV9PRXdcOPUV8AgbdrTUtf8E9B0r68Qc_GPT_AK",
    base_url="https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/multimodal/crawl",
):
    url = f"{base_url}?ak={ak}"
    # 设置请求头
    headers = {
        "Content-Type": "application/json",
        "X-TT-LOGID": logid,
    }
    # 准备请求体数据
    payload = {
        "stream": False,
        "model": model_name,
        "max_tokens": max_token,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "response_modalities": ["TEXT", "IMAGE"],
        "thinking": {"budget_tokens": thinking_budget, "include_thoughts": False}
    }

    if specify_gen_ratio:
        # payload["generationConfig"] = {"imageConfig": {"aspectRatio": ratio}}
        payload["image_config"] = {
            # 支持以下比例
            # Landscape: 21:9, 16:9, 4:3, 3:2
            # Square: 1:1
            # Portrait: 9:16, 3:4, 2:3
            # Flexible: 5:4, 4:5
            "aspectRatio": ratio,
            "imageSize": imageSize,
            "imageOutputOptions": {"mimeType": "image/png"},
        }

    if image_urls:
        if isinstance(image_urls, str):
            payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": image_urls}})
        else:
            for img_url in image_urls:
                payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": img_url}})
    elif image_pils:
        if isinstance(image_pils, Image.Image):
            payload["messages"][0]["content"].append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_bytes(image_pils)}"}}
            )
        else:
            for image in image_pils:
                payload["messages"][0]["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_bytes(image)}"}}
                )

    for i in range(num_retry):
        # 发送POST请求
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response_json = json.loads(response.text)

        if "usage" not in response_json:
            err_msg = f"`{model_name}` error code({response_json['error']['code']}), message: {response_json['error']['message']}"
            logging.warning(err_msg)
            raise Exception(err_msg)
        if i < num_retry - 1 and "multimodal_contents" not in response_json["choices"][0]["message"]:
            logging.info(f"[{model_name}] retry {i + 1} times")
            continue

        res = {}
        res["usage"] = response_json["usage"]
        for item in response_json["choices"][0]["message"]["multimodal_contents"]:
            if item["type"] == "inline_data":
                base64_img = item["inline_data"]["data"]
                img = decode_pil_bytes(base64_img)
                res["image"] = img

        if i < num_retry - 1 and "image" not in res:
            logging.info(f"[{model_name}] retry {i + 1} times")
            continue

        logging.info(f"[{model_name}] usage: {response_json['usage']}")
        return res


if __name__ == "__main__":
    # 调用多模态gemini-3.1-pro大模型的一个示例
    result_prompt = mllm_make_image_request(
        "翻译一下图中的文字",
        image_urls=["https://p16-oec-sg.ibyteimg.com/tos-alisg-i-aphluv4xwc-sg/8fa9fe853ef549e9bb0c308900aedab8~tplv-aphluv4xwc-origin-jpeg.jpeg?dr=15568&nonce=35024&refresh_token=7315fdc338070c10d871d06a11c4b700&from=1010592719&idc=my&ps=933b5bde&shcp=9b759fb9&shp=3c3c6bcf&t=555f072d"],
        thinking_budget=1024,
        max_tokens=5000,
        timeout=60,
        temperature=1.0,
        is_json_response=False,
        model_name="gemini-3.1-p",
        api_key="pvs0e2G0HEQTQlzj89MxL5764bNwleyW_GPT_AK",
        base_url="https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl",
    )
    print(result_prompt)

    # 调用图像生成gemini-3.1-flash-image模型的一个示例
    gen_result = gemini_image_generate(
        "Preserve the original product exactly as it appears. Position the ring so it is delicately held between the gently pinched thumb and forefinger of an elegant woman's hand, demonstrating its scale and delicacy. The hand features glossy, perfectly manicured nails in a natural nude tone, providing a warm, tactile backdrop that complements the gold and stones. The lighting is soft and warm, casting gentle highlights on the skin and enhancing the natural sparkle of the original ring. The composition is a tight, intimate close-up with a shallow depth of field, softly blurring the background into warm, muted skin tones and soft out-of-focus fabrics, ensuring the visual focus remains completely on the fine details of the ring and the elegant grip of the fingertips. Premium, intimate lifestyle jewelry photography.",
        image_urls=["https://p16-oec-sg.ibyteimg.com/tos-alisg-i-aphluv4xwc-sg/5e8c4db5a98e41f1be1708c776e4b14b~tplv-aphluv4xwc-origin-jpeg.jpeg?dr=15568&nonce=65613&refresh_token=08342664d430123adde69e716a5fc279&from=1010592719&idc=my&ps=933b5bde&shcp=9b759fb9&shp=3c3c6bcf&t=555f072d&t=1776413148462"],
        specify_gen_ratio=True,
        ratio="9:16",
        max_token=3000,
        thinking_budget=0,
        model_name="gemini-3.1-fi",
        ak="XpGAiljn5ZhzjpRn8Eo8BRvj1uWcaCmP_GPT_AK",
        base_url="https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/multimodal/crawl",
    )
    print(gen_result)
