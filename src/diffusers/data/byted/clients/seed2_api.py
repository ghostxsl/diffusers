import json
import re
import openai

from diffusers.data.outer_vos_tools import encode_pil_bytes


class Seed2Client:
    def __init__(self,
                 base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
                 api_key="bd4c2dd5-d0ca-4706-a43d-1ef3d218f290",
                 model="ep-20260306112759-m5lgj",
                 reasoning_effort="high"):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=300)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.last_thinking = None
        self.last_reasoning_tokens = None
        self.mime = {"png": "png", "jpg": "jpeg", "jpeg": "jpeg", "webp": "webp"}
        self.reasoning_levels = ["minimal", "low", "medium", "high"]

    def _build_content(self, prompt, image_urls=None, image_pils=None):
        content = [{"type": "text", "text": prompt}]
        for url in (image_urls or []):
            content.append({"type": "image_url", "image_url": {"url": url}})
        for img in (image_pils or []):
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_bytes(img)}"}})
        return content

    def call(self, prompt, image_urls=None, image_pils=None, max_tokens=4096,
             timeout=600, json_mode=False):
        content = self._build_content(prompt, image_urls, image_pils)

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "timeout": timeout,
        }

        if self.reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)

        self.last_thinking = None
        self.last_reasoning_tokens = None
        self.last_usage = None
        if resp.choices:
            msg = resp.choices[0].message
            raw = json.loads(resp.model_dump_json())
            raw_msg = raw.get("choices", [{}])[0].get("message", {})
            self.last_thinking = raw_msg.get("reasoning_content", None)
            if resp.usage:
                usage = raw.get("usage", {})
                self.last_usage = usage
                details = usage.get("completion_tokens_details") or {}
                self.last_reasoning_tokens = (
                    details.get("reasoning_tokens")
                    or usage.get("reasoning_tokens")
                )
            return msg.content or ""
        return ""

    def call_json(self, prompt, image_urls=None, image_pils=None, max_tokens=4096):
        raw = self.call(prompt, image_urls, image_pils, max_tokens, json_mode=True)
        m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        json_str = m.group(1) if m else raw
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                import json_repair
                return json_repair.loads(json_str)
            except Exception:
                return {"_raw": raw}
