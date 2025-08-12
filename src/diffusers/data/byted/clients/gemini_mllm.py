import re
import json
import logging
import requests
import traceback
from retrying import retry
from PIL import Image
import openai

import diffusers.data.byted.errno as err
from diffusers.data.byted.decorator import timer
from diffusers.data.byted.tos import save_tos, _gen_name
from diffusers.data.outer_vos_tools import encode_pil_bytes, decode_pil_bytes, load_or_download_image
from diffusers.data.byted.clients.azure_mllm import MLLMClient


@timer
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
    base_url="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/multimodal/crawl",
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
        raise err.WithCodeError(err.ErrorCodeOpenaiError, f"`{model_name}` request error: {e}")


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
            base_url="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl",
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


def gemini_image_generate(
    prompt,
    image_urls=[],
    image_pils=[],
    specify_gen_ratio=False,
    ratio="9:16",
    imageSize="1k",
    max_token=3800,
    thinking_budget=0,
    logid="",
    num_retry=2,
    model_name="gemini-2.5-flash-image",
    ak="lV9PRXdcOPUV8AgbdrTUtf8E9B0r68Qc_GPT_AK",
    base_url="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/multimodal/crawl",
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
            raise err.WithCodeError(err.ErrorCodeGenImageError, err_msg)
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


@timer
@retry(stop_max_attempt_number=1, wait_fixed=500)
def gemini_image_generate_module(
    prompt,
    image_urls=[],
    image_pils=[],
    specify_gen_ratio=True,
    ratio="9:16",
    max_token=3800,
    thinking_budget=0,
    logid="",
    model_name="gemini-2.5-flash-image",
    ak="lV9PRXdcOPUV8AgbdrTUtf8E9B0r68Qc_GPT_AK",
    base_url="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/multimodal/crawl",
):
    try:
        result_gemini = gemini_image_generate(
            prompt,
            image_urls,
            image_pils,
            specify_gen_ratio=specify_gen_ratio,
            ratio=ratio,
            max_token=max_token,
            thinking_budget=thinking_budget,
            logid=logid,
            model_name=model_name,
            ak=ak,
            base_url=base_url,
        )
        if "image" not in result_gemini:
            msg = "[gemini_image_generate_module] error: Gemini call failed, result is `None`"
            raise err.WithCodeError(err.ErrorCodeGenImageError, msg)

        gen_url = save_tos(encode_pil_bytes(result_gemini["image"], False), f"{_gen_name('')}.jpg")
        if gen_url is None:
            raise err.WithCodeError(err.ErrCodeTosError, "Image file upload to TOS failed")
        logging.info(
            f"[gemini_image_generate_module] src_url:{image_urls}, gen_url: {gen_url}, "
            f"specify_gen_ratio: {specify_gen_ratio}, ratio: {ratio}, max_token: {max_token}, "
            f"logid: {logid}, generator: {model_name}"
        )
        return gen_url, result_gemini["image"]
    except err.WithCodeError:
        raise
    except Exception as e:
        msg = f"[gemini_image_generate_module] error: {str(e)}, traceback: {traceback.format_exc()}"
        raise err.WithCodeError(err.ErrorCodeGenImageError, msg)
