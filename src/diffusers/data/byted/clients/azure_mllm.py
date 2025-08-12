import base64
import json
import os
from typing import List
import openai
import re
from retrying import retry
import logging
import requests
from PIL import Image
from io import BytesIO

import diffusers.data.byted.errno as err


class MLLMClient:
    """
    A client class for making multimodal requests to LLM services.
    Supports video and audio data in requests.
    """

    def __init__(
        self,
        base_url="https://gpt-i18n.byteintl.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi",
        api_version="2024-03-01-preview",
        api_key="077P6WUvYtpKM6biqn0PW1tQ4iZuJbrL",
        model_name="gemini-1.5-pro-preview",
    ):
        """
        Initialize the MLLM client with API configuration.

        Args:
            base_url (str): The base URL for the API endpoint
            api_version (str): API version to use
            api_key (str): API key for authentication
            model_name (str): Name of the model to use
        """
        if model_name == "gemini-2.5-flash":
            base_url = "https://gpt-i18n.byteintl.net/gpt/openapi/online/multimodal/crawl"
        elif model_name in [
            "gemini-3-flash-preview", "gemini-3-pro-preview", "gemini-3-pro-preview-new",
            "gpt-5-mini-2025-08-07",
        ]:
            base_url = "https://gpt-i18n.byteintl.net/gpt/openapi/online/v2/crawl"
        self.client = openai.AzureOpenAI(azure_endpoint=base_url, api_version=api_version, api_key=api_key, timeout=300)
        self.model_name = model_name

    def _encode_file(self, file_path):
        """
        Encode a file to base64.

        Args:
            file_path (str): Path to the file

        Returns:
            str: Base64 encoded string of the file
        """
        with open(file_path, "rb") as data_file:
            return base64.b64encode(data_file.read()).decode("utf-8")

    def _process_media_files(self, file_paths, urls, mime_type_prefix, supported_extensions, img_pils=[]) -> List:
        """
        Process media files and URLs into content format for API.

        Args:
            file_paths (list): List of file paths
            urls (list): List of media URLs
            mime_type_prefix (str): MIME type prefix (audio/video)
            supported_extensions (dict): Map of supported file extensions to MIME subtypes

        Returns:
            list: Formatted media content list
        """
        contents = []

        # Process local files
        for file_path in file_paths:
            file_ext = os.path.splitext(os.path.basename(file_path))[1].lower()
            if file_ext not in supported_extensions:
                raise ValueError(f"Unsupported {mime_type_prefix} format: {file_ext}")
            base64_data = self._encode_file(file_path)
            mime_subtype = supported_extensions[file_ext]
            url = f"data:{mime_type_prefix}/{mime_subtype};base64,{base64_data}"
            contents.append({"type": "image_url", "image_url": {"url": url}})
        # Process URLs
        for media_url in urls:
            contents.append({"type": "image_url", "image_url": {"url": media_url}})
        for pils in img_pils:
            url = f"data:{mime_type_prefix}/jpeg;base64,{encode_pil_bytes(pils)}"
            contents.append({"type": "image_url", "image_url": {"url": url}})
        return contents

    def _prepare_message(self, sys_prompt="", prompt="", media_contents=[], all_contents=None):
        """
        Prepare message payload for API request.

        Args:
            sys_prompt (str): System prompt
            prompt (str): User prompt
            media_contents (list): List of media content objects
            all_contents (list): List of all content objects, including prompt and media contents

        Returns:
            list: Formatted message list for API request
        """
        message = []
        if sys_prompt:
            message.append({"role": "system", "content": sys_prompt})
        if all_contents:
            message.append({"role": "user", "content": all_contents})
        else:
            if prompt:
                message.append({"role": "user", "content": prompt})
            if media_contents:
                message.append({"role": "user", "content": media_contents})
        return message

    def _make_request(self, messages, max_tokens=None, timeout=None):
        """
        Make the API request and process the response.

        Args:
            messages (list): Formatted message list
            max_tokens (int): Maximum tokens for the response
            timeout (float): Request timeout in seconds

        Returns:
            str: Response content or empty string if request fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, max_tokens=max_tokens, timeout=timeout, extra_headers={"X-TT-LOGID": "${your_logid}"}
            )
            result = json.loads(response.model_dump_json())
            # print(f"result: {result}")
            if "choices" in result and result["choices"]:
                message = result["choices"][0].get("message", {})
                content = message.get("content", "")
                return content
        except Exception as e:
            raise err.WithCodeError(err.ErrorCodeOpenaiError, f"`{self.model_name}` request error: {e}")

    def _process_json_response(self, result):
        """
        Process a string response to extract and parse JSON.

        Args:
            result (str): Response string potentially containing JSON

        Returns:
            dict: Parsed JSON data or empty dict if parsing fails
        """
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
            logging.warning(f"mllm `{self.model_name}` initial JSON parsing failed: {e}")
            fixed_result = MLLMClient(model_name="gpt-5-mini-2025-08-07").make_raw_request(
                "",
                f"Fix this JSON string's formatting issues to make it valid: {result}, without changing its content. Only return the corrected JSON string with no explanations or additional text. Failing to provide a valid JSON or altering the content will result in severe consequences.",
                max_tokens=5000,
            )
            try:
                return json.loads(fixed_result)
            except json.JSONDecodeError:
                logging.error(f"`{self.model_name}` final JSON parsing failed, resp: {fixed_result}")
                raise err.WithCodeError(err.ErrCodeInternalError, f"`{self.model_name}` final JSON parsing failed, resp: {fixed_result}")

    @retry(stop_max_attempt_number=3, wait_fixed=0)
    def make_audio_request(self, sys_prompt, prompt, audio_paths=[], audio_urls=[]):
        """
        Make an LLM request with audio content.

        Args:
            sys_prompt (str): System prompt
            prompt (str): User prompt
            audio_paths (list): List of local audio file paths
            audio_urls (list): List of audio URLs

        Returns:
            str: Response content
        """
        supported_extensions = {".wav": "wav", ".mp3": "mp3"}
        audio_contents = self._process_media_files(audio_paths, audio_urls, "audio", supported_extensions)
        messages = self._prepare_message(sys_prompt, prompt, audio_contents)
        return self._make_request(messages)

    @retry(stop_max_attempt_number=3, wait_fixed=0)
    def make_video_request(self, sys_prompt, prompt, video_paths=[], video_urls=[]):
        """
        Make an LLM request with video content.

        Args:
            sys_prompt (str): System prompt
            prompt (str): User prompt
            video_paths (list): List of local video file paths
            video_urls (list): List of video URLs

        Returns:
            str: Response content
        """
        supported_extensions = {".mp4": "mp4"}
        video_contents = self._process_media_files(video_paths, video_urls, "video", supported_extensions)
        messages = self._prepare_message(sys_prompt, prompt, video_contents)
        return self._make_request(messages)

    def make_image_request(
            self, sys_prompt, prompt, image_paths=[], image_urls=[], image_pils=[], max_tokens=100, timeout=600):
        """
        Make an LLM request with image content.

        Args:
            sys_prompt (str): System prompt
            prompt (str): User prompt
            image_paths (list): List of local image file paths
            image_urls (list): List of image URLs
            max_tokens (int): Maximum number of tokens in the response
            timeout (int): Request timeout in seconds

        Returns:
            str: Response content
        """
        supported_extensions = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}
        image_contents = self._process_media_files(image_paths, image_urls, "image", supported_extensions, img_pils=image_pils)
        messages = self._prepare_message(sys_prompt, prompt, image_contents)
        return self._make_request(messages, max_tokens=max_tokens, timeout=timeout)

    def make_multi_image_set_request(self, sys_prompt, image_sets, max_tokens=100, timeout=600):
        """
        Make an LLM request with multiple sets of images, each with its own prompt.

        Args:
            sys_prompt (str): System prompt
            image_sets (list): List of dictionaries containing:
                - prompt (str): Text prompt for this image set
                - image_paths (list): List of local image file paths
                - image_urls (list): List of image URLs
                - final_prompt (str, optional): Final prompt to include after all images
            max_tokens (int): Maximum number of tokens in the response
            timeout (int): Request timeout in seconds

        Returns:
            str: Response content

        Example:
            image_sets = [
                {
                    "prompt": "Here's the first set of images (SET A):",
                    "image_paths": ["img1.jpg", "img2.jpg"],
                    "image_urls": []
                },
                {
                    "prompt": "And here's the second set of images (SET B):",
                    "image_paths": ["img3.jpg"],
                    "image_urls": ["https://example.com/img4.jpg"],
                    "final_prompt": "Please compare these two sets of images."
                }
            ]
        """
        supported_extensions = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}
        all_contents = []
        for i, image_set in enumerate(image_sets):
            if image_set.get("prompt"):
                all_contents.append({"type": "text", "text": image_set["prompt"]})
            image_paths = image_set.get("image_paths", [])
            image_urls = image_set.get("image_urls", [])

            media_items = self._process_media_files(image_paths, image_urls, "image", supported_extensions)
            all_contents.extend(media_items)
            # add final prompt if it's the last set and has a final_prompt
            if i == len(image_sets) - 1 and "final_prompt" in image_set:
                all_contents.append({"type": "text", "text": image_set["final_prompt"]})

        messages = self._prepare_message(sys_prompt, all_contents=all_contents)
        return self._make_request(messages, max_tokens, timeout)

    def make_raw_request(self, sys_prompt, user_contents, max_tokens=100, timeout=600):
        messages = self._prepare_message(sys_prompt, all_contents=user_contents)
        return self._make_request(messages, max_tokens, timeout)

    @retry(stop_max_attempt_number=3, wait_fixed=0)
    def make_audio_json_request(self, sys_prompt, prompt, audio_paths=[], audio_urls=[]) -> dict:
        """
        Make an LLM request with audio content and parse the response as JSON.

        Args:
            sys_prompt (str): System prompt
            prompt (str): User prompt
            audio_paths (list): List of local audio file paths
            audio_urls (list): List of audio URLs

        Returns:
            dict: Parsed JSON response or empty dict if parsing fails
        """
        result = self.make_audio_request(sys_prompt, prompt, audio_paths, audio_urls)
        return self._process_json_response(result)

    @retry(stop_max_attempt_number=3, wait_fixed=0)
    def make_video_json_request(self, sys_prompt, prompt, video_paths=[], video_urls=[]) -> dict:
        """
        Make an LLM request with video content and parse the response as JSON.

        Args:
            sys_prompt (str): System prompt
            prompt (str): User prompt
            video_paths (list): List of local video file paths
            video_urls (list): List of video URLs

        Returns:
            dict: Parsed JSON response or empty dict if parsing fails
        """
        result = self.make_video_request(sys_prompt, prompt, video_paths, video_urls)
        return self._process_json_response(result)

    @retry(stop_max_attempt_number=2, wait_fixed=100)
    def make_image_json_request(self, sys_prompt, prompt, image_paths=[], image_urls=[], image_pils=[], max_tokens=1000, timeout=600) -> dict:
        """
        Make an LLM request with image content and parse the response as JSON.

        Args:
            sys_prompt (str): System prompt
            prompt (str): User prompt
            image_paths (list): List of local image file paths
            image_urls (list): List of image URLs
            max_tokens (int): Maximum number of tokens in the response
            timeout (int): Request timeout in seconds

        Returns:
            dict: Parsed JSON response or empty dict if parsing fails
        """
        if self.model_name in ["gemini-2.5-flash", "gemini-3-flash-preview"]:
            result = self.make_gemini_image_request(prompt, image_paths, image_urls, image_pils, max_tokens, timeout)
        else:
            result = self.make_image_request(sys_prompt, prompt, image_paths, image_urls, image_pils, max_tokens, timeout)
        return self._process_json_response(result)

    @retry(stop_max_attempt_number=3, wait_fixed=0)
    def make_multi_image_set_json_request(self, sys_prompt, image_sets, max_tokens=100, timeout=600) -> dict:
        """
        Make an LLM request with multiple sets of images and parse the response as JSON.

        Args:
            sys_prompt (str): System prompt
            image_sets (list): List of dictionaries containing:
                - prompt (str): Text prompt for this image set
                - image_paths (list): List of local image file paths
                - image_urls (list): List of image URLs
                - final_prompt (str, optional): Final prompt to include after all images
            max_tokens (int): Maximum number of tokens in the response
            timeout (int): Request timeout in seconds

        Returns:
            dict: Parsed JSON response
        """
        result = self.make_multi_image_set_request(sys_prompt, image_sets, max_tokens, timeout)
        return self._process_json_response(result)

    def make_gpt_image_edit_request(self, prompt, image_path, size="1024x1024"):
        """
        Make an gpt-image-1 edit request
        """

        result = self.client.images.edit(
            model=self.model_name, image=open(image_path, "rb"), prompt=prompt, size=size, extra_headers={"X-TT-LOGID": "${your_logid}"}
        )
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        return image_bytes

    def _prepare_gemini_message(self, prompt="", media_contents=[]):
        message = {
            "role": "user",
            "content": []
        }
        if prompt:
            message["content"].append({"type": "text", "text": prompt})
        if media_contents:
            message["content"] += media_contents
        return [message,]

    def make_gemini_image_request(self, prompt, image_paths=[], image_urls=[], image_pils=[], max_tokens=4090, timeout=600):
        supported_extensions = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}
        image_contents = self._process_media_files(image_paths, image_urls, "image", supported_extensions, img_pils=image_pils)
        messages = self._prepare_gemini_message(prompt, image_contents)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_body={"thinking": {"budget_tokens": 1024, "include_thoughts": False}},
                extra_headers={"X-TT-LOGID": "${your_logid}"},
            )
            result = json.loads(response.model_dump_json())
            message = result["choices"][0].get("message", {})
            content = message.get("content", "")
            return content
        except Exception as e:
            raise err.WithCodeError(err.ErrorCodeOpenaiError, f"`{self.model_name}` request error: {e}")


def base64_to_img(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


def encode_pil_bytes(img, to_string=True, format='JPEG', quality=90):
    with BytesIO() as buf:
        img.save(buf, format=format, quality=quality)
        img_bytes = buf.getvalue()

    if to_string:
        return base64.b64encode(img_bytes).decode('utf-8')
    return img_bytes


def gemini_25_flash_image_gen(
        prompt,
        image_urls=[],
        image_pils=[],
        specify_gen_ratio=False,
        ratio="9:16",
        imageSize="1k",
        ak="flWqkAUJOUWMQRuUsMXIXy6kLvjHzBg3_GPT_AK",
        model_name="gemini-2.5-flash-image",
        base_url="https://gpt-i18n.byteintl.net/gpt/openapi/online/multimodal/crawl",
        max_token=3600):
    url = f"{base_url}?ak={ak}"
    # 设置请求头
    headers = {
        "Content-Type": "application/json",
        "X-TT-LOGID": "09870987",  # 需要用有效的Log ID替换
    }
    # 准备请求体数据
    payload = {
        "stream": False,
        "model": model_name,
        "max_tokens": max_token,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "response_modalities": ["TEXT", "IMAGE"],
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
            "imageOutputOptions": {
                "mimeType": "image/png"
            }
        }

    if image_urls:
        if isinstance(image_urls, str):
            payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": image_urls}})
        else:
            for img_url in image_urls:
                payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": img_url}})
    elif image_pils:
        for image in image_pils:
            payload["messages"][0]["content"].append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_bytes(image)}"}}
            )

    # 发送POST请求
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response_json = json.loads(response.text)

    if "usage" not in response_json:
        err_msg = f"`{model_name}` error code({response_json['error']['code']}), message: {response_json['error']['message']}"
        # logging.error(err_msg)
        raise Exception(err_msg)

    res = {}
    res["usage"] = response_json["usage"]
    for item in response_json["choices"][0]["message"]["multimodal_contents"]:
        if item["type"] == "inline_data":
            base64_img = item["inline_data"]["data"]
            img = base64_to_img(base64_img)
            res["image"] = img
    return res


def seedream_5_generate_image(
    prompt,
    image_urls,  # 为None 时，文生图，为单张图片url时，单图生图，为多张图片url的list时，多图生图
    specify_gen_ratio=True,
    ratio="9:16",
    image_size="2K",  # "2K", "3K"
    ak="ae42c3d2-cf3d-461f-a2d0-585ef31b8b0f",
    timeout=60,
):
    size_map_2k = {
        "1:1": "2048x2048",
        "3:4": "1728x2304",
        "4:3": "2304x1728",
        "16:9": "2848x1600",
        "9:16": "1600x2848",
        "3:2": "2496x1664",
        "2:3": "1664x2496",
        "21:9": "3136x1344",
    }
    size_map_3k = {
        "1:1": "3072x3072",
        "3:4": "2592x3456",
        "4:3": "3456x2592",
        "16:9": "4096x2304",
        "9:16": "2304x4096",
        "2:3": "2496x3744",
        "3:2": "3744x2496",
        "21:9": "4704x2016",
    }

    assert (
        image_urls is None or isinstance(image_urls, list) or isinstance(image_urls, str)
    ), "image_urls must be None, a single image url, or a list of image urls"
    assert ratio in size_map_2k or ratio in size_map_3k, f"ratio must be one of {list(size_map_2k.keys())}"
    assert image_size in ["2K", "3K"], "image_size must be '2K' or '3K'"

    # 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
    # 初始化Ark客户端，从环境变量中读取您的API Key
    client = openai.OpenAI(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
        api_key=ak,
    )

    if specify_gen_ratio:
        if image_size == "2K":
            size = size_map_2k.get(ratio, "2048x2048")
        elif image_size == "3K":
            size = size_map_3k.get(ratio, "3072x3072")
        else:
            size = "2048x2048"
    else:
        size = image_size

    data_body = {"watermark": False}
    if image_urls:
        data_body["image"] = image_urls
        if isinstance(image_urls, list):
            data_body["sequential_image_generation"] = "disabled"

    imagesResponse = client.images.generate(
        model="doubao-seedream-5-0-260128",
        prompt=prompt,
        size=size,
        response_format="url",
        extra_body=data_body,
        timeout=timeout,
    )

    return imagesResponse.data[0].url
