# @author: wilson.xu.
import argparse
from tqdm import tqdm
import random
import json
import math
from PIL import Image
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.byted.clients.azure_mllm import MLLMClient, gemini_25_flash_image_gen
from diffusers.data.outer_vos_tools import encode_pil_bytes, load_or_download_image
from diffusers.data.byted.tos import save_tos
from diffusers.data.utils import json_save, load_file, resize_image_by_short_side
from biz.solution.psa.psa_atmos_gen.html_addon_basic_module import HtmlAddOn, SellingPointItem


gpt_client = MLLMClient(model_name="gpt-5-mini-2025-08-07", api_key="7e5RK9vuv5NTXU07CosK9uLotGpltSpD_GPT_AK")
gemini_client = MLLMClient(model_name="gemini-2.5-flash", api_key="3H7OHTGn3JwHvlZHP9JUJzb850gp3TGR_GPT_AK")


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="/mnt/bn/creative-algo/xsl/label_0121_psa_ingress_filtering_3.json", type=str)
    parser.add_argument(
        "--output_file", default="gemini_0121_psa_ingress_filtering_4.json", type=str)
    parser.add_argument(
        "--num_workers", default=10, type=int)

    args = parser.parse_args()
    return args


def add_percentage_padding(image: Image.Image, padding=[0, 0, 0, 0], color=(127, 127, 127)) -> Image.Image:
    """
    为PIL图像添加指定百分比的白色边框

    Args:
        image: PIL.Image.Image - 原始图像对象
        padding: list - 长度为4的列表，依次对应[上, 下, 左, 右]的白边百分比(0-1)

    Returns:
        Image.Image - 添加padding后的新图像
    """
    # 验证输入参数
    if len(padding) != 4:
        raise ValueError("padding必须是长度为4的列表，格式为[上, 下, 左, 右]")
    for p in padding:
        if not (0 <= p <= 1):
            raise ValueError("padding中的每个值必须在0到1之间")

    # 获取原始图像尺寸
    original_width, original_height = image.size

    # 计算四个方向的padding像素数（向上取整确保为整数）
    pad_top = math.ceil(original_height * padding[0])
    pad_bottom = math.ceil(original_height * padding[1])
    pad_left = math.ceil(original_width * padding[2])
    pad_right = math.ceil(original_width * padding[3])

    # 计算新图像的尺寸
    new_width = original_width + pad_left + pad_right
    new_height = original_height + pad_top + pad_bottom

    # 创建白色背景的新图像（RGB模式，像素值255为白色）
    # 如果原始图像有透明度通道，使用RGBA模式
    img_mode = image.mode if image.mode in ["RGB", "RGBA"] else "RGB"
    new_image = Image.new(img_mode, (new_width, new_height), color=color)

    # 将原始图像粘贴到新图像的对应位置
    new_image.paste(image, (pad_left, pad_top))

    return new_image


prompt_check_id = """You are a meticulous AI Image Quality Inspector, focusing on **major, structural failures**, not minor rendering artifacts. Your sole task is to analyze a pair of images ([Original Image] and [Generated Image]) based on a strict set of rules and output your findings in a JSON format.

===== CONTEXT: UNDERSTANDING THE TASK =====
You must first understand the image generation task you are inspecting. An AI model was instructed to **ONLY replace the gray banner areas** at the top and/or bottom of the [Original Image]. The central, non-banner area, which contains all original content, **was explicitly ordered to be kept 100% unchanged**. Your job is to verify if this critical order was violated in a significant way.

===== INPUTS =====
1.  `[Original Image]`: The source image before AI editing.
2.  `[Generated Image]`: The image after AI editing.
3.  `[AI Generation Prompt]`: The exact prompt used to create the [Generated Image]: `{AI_gen_prompt}`.

===== P0 CRITICAL FAILURE RULES =====
You must check for the following two types of critical failures. If EITHER of these rules is violated, the inspection result is "P0".

1.  **Text in Banner (P0 Fail):**
    Carefully examine the top and bottom banner areas of the [Generated Image]. If you detect **ANY** newly generated text, letters, numbers, or recognizable brand marks, this is a P0 failure.

2.  **Central Area Substantially Modified (P0 Fail):**
    Compare the central, non-banner area of the [Generated Image] against the identical area in the [Original Image]. You must **only flag major, substantive changes** that fundamentally alter the image's content or composition.

    **--- What constitutes a "Substantial Change" (P0 Fail): ---**
    *   **Object/Subject Integrity Failure:** The main product, model, or key subject is clearly distorted, missing, swapped for another object, or has major parts added/removed (e.g., a T-shirt becomes a jacket).
    *   **Background Structural Failure:** The original background *within the central area* is completely replaced with a different scene or texture (e.g., a studio background becomes a forest).
    *   **Compositional Vandalism:** New, intrusive objects, large shapes, or disruptive patterns appear in the central area, clearly obscuring the original content.

    **--- What to IGNORE (These are NOT P0 Failures): ---**
    *   **Minor Text/Logo Rendering Variations:** Slight, almost imperceptible shifts in the font weight, sharpness, or rendering of existing text and logos within the central area. The AI may subtly re-render these elements, but as long as the text is still legible and the logo shape is intact, it is **NOT** a failure.
    *   **Imperceptible Artifacts:** Tiny compression artifacts, noise, or subtle color shifts that do not change the identity of any object.

===== JUDGEMENT & OUTPUT =====
Based on your inspection, you must generate a single JSON object. This JSON object **must strictly** contain the following three keys: `"result"`, `"issues"`, and `"reason"`.

*   `"result"`: (String) The value must be either `"Good"` or `"P0"`.
*   `"issues"`: (List of Strings) If the result is `"Good"`, this must be an **empty list `[]`**. If "P0", it must contain one or more of the following strings:
    *   `"Text found in banner"`
    *   `"Central area modified"`
*   `"reason"`: (String) A **very specific and concrete** explanation of the **major** issue(s) in **Chinese**. If the result is `"Good"`, this must be an **empty string `""`**.
    *   Example of a good P0 reason: `"中央区域的背景从室内场景被错误地替换为了室外花园，严重改变了原始图像内容。"`
    *   Example of another good P0 reason: `"产品主体（一件T恤）被错误地修改，增加了不存在的口袋图案。"`

===== EXAMPLES =====

**--- P0 Failure Example (Major Change) ---**
*   **Situation:** The [Generated Image] shows the central area's background has changed from a white wall to a brick wall.
*   **Correct JSON Output:**
    ```json
    {{
      "result": "P0",
      "issues": [
        "Central area modified"
      ],
      "reason": "中央区域的背景从原始的白色墙壁被错误地修改成了砖墙。"
    }}
"""


poster_add_banner = """You are a Senior E-Commerce Art Director and an Expert Prompt Engineer. Your primary mission is to design visually harmonious top and bottom banners for a given e-commerce image, leaving the original central content completely untouched. You will analyze the image and output a single, continuous, and extremely concise image editing prompt, adhering strictly to the "Atomic Command" principle.

===== NON-NEGOTIABLE CORE RULES (PRIORITY ORDER) =====
1.  **Original Image Invariance Mandate (TOP PRIORITY):** The final prompt **must begin** with the following exact and detailed sentence: "**Keep the original, non-gray central image area perfectly preserved, ensuring all its pixels, content, and original aspect ratio remain 100% unchanged without any distortion, cropping, or modification.**"
2.  **Atomic Command & Simplicity Mandate (CRITICAL):** To prevent instruction contamination, the creative description for the banners must be a **single, extremely concise, and self-contained phrase.**
    - It **must not** contain complex, comma-separated clauses, run-on details, or chained descriptions (e.g., "featuring...", "with...", "and also...").
    - The entire design concept must be minimalistic, focusing on a primary texture or color.
3.  **Strict Banner Composition Mandate:** The content within the banners must obey these layout rules:
    - **Absolute Prohibition:** Absolutely no text, letters, numbers, or logos are permitted anywhere in the banners under any circumstances.
    - **Central Field Rule:** The main, central area of the banner must consist **only** of a clean texture, material, abstract color gradient, or similar non-representational background. This area must remain uncluttered.
    - **Side Elements Exception:** If any small, discrete design element (e.g., a simple snowflake icon, a water droplet shape) is used, it **must** be placed on the **far left or far right** of the banner.
4.  **Final "Simple Repetition" Prompt Structure:** The output prompt must strictly follow this highly literal and safe structure:
    - **Part 1 (Protect):** The mandatory invariance sentence from Rule #1.
    - **Part 2 (Apply Top):** A sentence starting with "Replace the top gray banner with..." followed by the **single, concise descriptive phrase** from Rule #2.
    - **Part 3 (Apply Bottom):** A sentence starting with "Replace the bottom gray banner with the identical..." followed by the **exact same single, concise descriptive phrase**.
5.  **Face Protection:** The appropriate face protection clause **must be appended at the very end** of the entire prompt. If a human face is present, include: "If any human face appears, leave it 100% photorealistic, sharp, and completely unaltered." If no face is present, include: "Do not generate any new humans, faces, or body parts."

===== CHAIN OF THOUGHT (INTERNAL ONLY, DO NOT OUTPUT) =====
1.  **Step 1: Identify & Analyze Original Image:** Identify the immutable original image area. Analyze its core subject and aesthetic.
2.  **Step 2: Conceive Atomic Phrase with Strict Composition:** Based on the analysis, conceptualize a banner design and distill it down to **one single, self-contained descriptive phrase.** Crucially, ensure this design adheres to the **Strict Banner Composition Mandate** (clean center, icons only on far sides, no text). Simplicity is paramount.
3.  **Step 3: Construct Prompt via Simple Repetition:** Assemble the final paragraph using the exact structure from Rule #4. First, write the protection sentence. Second, write the replacement command for the top banner using your atomic phrase. Third, repeat the exact same command for the bottom banner. Finally, append the correct face protection clause.

===== OUTPUT REQUIREMENT =====
Output a single, continuous paragraph of **pure natural language description**. Do not use JSON, markdown, or any technical formatting. **Critical: Your primary goal is to generate the simplest, most concise, and safest possible prompt for a literal-minded downstream model.**

===== BUILT-IN EXAMPLE (FOR CLARITY) =====
**Input Image Description:**
- A 3:4 vertical image. The central area shows a sleek, modern water bottle against a studio background. The top and bottom are gray banners. The aesthetic is clean, fresh, and hydrated.
- No human face is present.

**Output Prompt Example:**
Keep the original, non-gray central image area perfectly preserved, ensuring all its pixels, content, and original aspect ratio remain 100% unchanged without any distortion, cropping, or modification. Replace the top gray banner with a clean light blue gradient panel with a single small water droplet icon on the far right. Replace the bottom gray banner with the identical clean light blue gradient panel with a single small water droplet icon on the far right. Do not generate any new humans, faces, or body parts.
"""


def send_request(item):
    image_url = item['mainimageurl']
    src_image = load_or_download_image(image_url)
    selling_points = json.loads(item["selling_points"])
    # src_ratio = src_image.width / src_image.height
    # if src_ratio < 0.8:
    #     raise Exception(f"The aspect ratio({round(src_ratio, 2)}) of the original image is less than 0.8")

    # 0. 增加banner
    add_image = add_percentage_padding(resize_image_by_short_side(src_image), padding=[0.16, 0.17, 0, 0])
    gen_url0 = save_tos(encode_pil_bytes(add_image, False), headers={"Content-Type": "image/jpeg"})

    # 1. prompt生成
    prompt_bg = gpt_client.make_image_request("", poster_add_banner, [], [], image_pils=[add_image], max_tokens=5000, timeout=60)
    item["prompt"] = prompt_bg

    # 2. 图像生成
    result = gemini_25_flash_image_gen(
        prompt_bg,
        image_urls=[],
        image_pils=[add_image],
        specify_gen_ratio=False,
        ratio="3:4",
        model_name="gemini-2.5-flash-image",
        ak="BpaRzJoHfD4aR28PbpiLAwMy3EBb4b1d_GPT_AK",
        max_token=4000,
    )
    gen_url1 = save_tos(encode_pil_bytes(result["image"], False), headers={"Content-Type": "image/jpeg"})
    item["gen_url"] = [gen_url0, gen_url1]

    # 3. 准出校验
    add_image_512 = resize_image_by_short_side(add_image, 512)
    preview_image = load_or_download_image(result["image"])
    res_image_512 = resize_image_by_short_side(preview_image, 512)
    meta_prompt_check_id = prompt_check_id.format(AI_gen_prompt=prompt_bg)
    res_json = gemini_client.make_image_json_request("", meta_prompt_check_id, [], [], image_pils=[add_image_512, res_image_512], max_tokens=4000, timeout=60)
    item["check_msg"] = json.dumps(res_json, ensure_ascii=False)

    # 4. html渲染
    addon = HtmlAddOn()
    # Custom textbox_pos: [center_x, center_y, width, height] (normalized)
    # Position at bottom: center_x=0.5, center_y=0.8, width=0.8, height=0.15
    textbox_poses = [[0.5, 0.936, 1.0, 0.128], [0.5, 0.062, 1.0, 0.125]]
    try:
        result_addon = addon.process_single_image_multi_selling_points(
            image_url=gen_url1,
            selling_points=[SellingPointItem(text=sp, textbox_pos=pos) for sp, pos in zip(selling_points[:2], textbox_poses)],
            image=result["image"],
            img_wh=result["image"].size,
            country=item["country"],
        )
        item["text_preview_url"] = [result_addon["text_preview_url"]]
    except Exception as e:
        raise Exception(f"html render error: {e}")

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
                    print(f"Error: {str(e)}")
                    error_results.append({'image_item': item, 'error_reason': str(e)})

    json_save(results, dst)
    print(f"error num: {len(error_results)}")


if __name__ == "__main__":
    import os
    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"
    os.environ["TCE_PSM"] = "ad.creative.image_core_solution"

    args = parse_args()
    print(f"input_file: {args.input_file}, num_workers: {args.num_workers}")

    data = load_file(args.input_file)
    random.shuffle(data)

    out = []
    for item in tqdm(data):
        if len(out) == 200:
            break
        if not isinstance(item["selling_points"], str) or len(json.loads(item["selling_points"])) < 2:
            continue
        if item["productRegion"] in ["VN", "TH"]:
            out.append(item)

    main(out, args.output_file, args.num_workers)

    print('Done!')
