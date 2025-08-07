import base64
import json
import os
from typing import List
import openai
import re
from retrying import retry
import logging

import diffusers.data.errno as err


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

    def _process_media_files(self, file_paths, urls, mime_type_prefix, supported_extensions) -> List:
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
            fixed_result = MLLMClient(model_name="gpt-4o-mini-2024-07-18").make_image_request(
                "",
                f"Fix this JSON string's formatting issues to make it valid: {result}, without changing its content. Only return the corrected JSON string with no explanations or additional text. Failing to provide a valid JSON or altering the content will result in severe consequences.",
                [],
                [],
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

    def make_image_request(self, sys_prompt, prompt, image_paths=[], image_urls=[], max_tokens=100, timeout=600):
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
        image_contents = self._process_media_files(image_paths, image_urls, "image", supported_extensions)
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

    @retry(stop_max_attempt_number=3, wait_fixed=0)
    def make_image_json_request(self, sys_prompt, prompt, image_paths=[], image_urls=[], max_tokens=100, timeout=600) -> dict:
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
        result = self.make_image_request(sys_prompt, prompt, image_paths, image_urls, max_tokens, timeout)
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
