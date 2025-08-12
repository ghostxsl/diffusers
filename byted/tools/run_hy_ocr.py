import json
from openai import OpenAI
from tqdm import tqdm
from typing import Dict, List

from diffusers.data.utils import load_csv_or_xlsx_to_dict, json_save
from diffusers.data.outer_vos_tools import encode_pil_bytes, load_or_download_image


def create_chat_messages(image_path: str, prompt: str):
    img = load_or_download_image(image_path)
    return [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_pil_bytes(img)}"
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]


def process_single_item(client: OpenAI, item: Dict):
    # Extract image path and prompt
    img_path = item['gen_url']
    prompt = f"提取图片中的标语。"

    # Create chat messages
    messages = create_chat_messages(img_path, prompt)

    # Get model response
    response = client.chat.completions.create(
        model="tencent/HunyuanOCR",
        messages=messages,
        temperature=0.0,
        top_p=0.95,
        seed=1234,
        stream=False,
        extra_body={
            "top_k": 1,
            "repetition_penalty": 1.0
        }
    )

    # Update data with model response
    item["vllm_answer"] = response.choices[0].message.content
    return item


def main():
    save_file = "result_hy_ocr.json"

    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        timeout=3600
    )

    data = load_csv_or_xlsx_to_dict("/mnt/bn/creative-algo/xsl/temp/PSASearchGeminiAddOn_log_processed.csv")
    data = [a for a in data if "OCR check not pass" in a["err_msg"]]

    out = []
    for line in tqdm(data):
        try:
            line = process_single_item(client, line)
            # print(line["vllm_answer"])
            out.append(line)
            if len(out) % 10 ==0:
                json_save(out, save_file)

        except Exception as e:
            print(f"Error processing line: {str(e)}")
    # Final save
    json_save(out, save_file)


if __name__ == "__main__":
    main()
    print("Done!")
