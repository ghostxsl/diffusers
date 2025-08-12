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


def add_percentage_padding(image: Image.Image, bg_image: Image.Image, padding: list = [0, 0, 0, 0]) -> Image.Image:
    """
    将主体图「嵌入」背景图中，背景图作为完整底图，按百分比计算边框后居中放置主体图
    彻底消除分块拼接的边界问题，实现自然的「边框包裹主体」效果

    Args:
        image: 原始主体图像对象
        bg_image: 背景图（将作为完整底图，resize后包裹主体图）
        padding: [上, 下, 左, 右] 填充百分比(0-1)

    Returns:
        嵌入完成的新图像（背景图完整，主体图居中）
    """
    # 1. 参数校验
    if len(padding) != 4:
        raise ValueError("padding必须是长度为4的列表：[上, 下, 左, 右]")
    for p in padding:
        if not (0 <= p <= 1):
            raise ValueError("padding值必须在0-1之间")
    if not isinstance(bg_image, Image.Image):
        raise TypeError("bg_image必须是PIL Image对象")

    # 2. 计算基础尺寸
    orig_w, orig_h = image.size
    pad_top = math.ceil(orig_h * padding[0])
    pad_bottom = math.ceil(orig_h * padding[1])
    pad_left = math.ceil(orig_w * padding[2])
    pad_right = math.ceil(orig_w * padding[3])

    # 最终画布尺寸 = 主体图尺寸 + 四周padding
    new_w = orig_w + pad_left + pad_right
    new_h = orig_h + pad_top + pad_bottom

    # 3. 核心逻辑：将背景图整体resize到最终画布尺寸（无分块，无拼接边界）
    target_mode = image.mode if image.mode in ("RGB", "RGBA") else "RGB"
    # 背景图先转成和主体图一致的模式，再高质量resize到最终画布大小
    bg_resized = bg_image.convert(target_mode).resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 4. 将主体图粘贴到背景图的中心位置（对应padding的偏移量）
    bg_resized.paste(image, (pad_left, pad_top))

    return bg_resized


poster_texture_map = """You are a top-tier Modern Digital Brand Designer and an expert Text-to-Image Prompt Engineer. Your primary mission is to analyze a given product image and conceive a standalone background image that is visually harmonious, sophisticated, and enhances the product's perceived value.

You will output a single, continuous, and highly descriptive text-to-image prompt. This prompt will be used by a generative AI (like Gemini Image Generation or Midjourney) to create a square, photorealistic background image. This background will later be cropped for banners.

===== NON-NEGOTIABLE CORE RULES =====
1.  **Output is a Standalone T2I Prompt:** Your entire output MUST be a single paragraph of pure, descriptive natural language. It must be directly usable as a text-to-image prompt. Do not add any conversational text, labels, or explanations.
2.  **Aesthetic Goal is Paramount:** The generated background must feel premium, clean, and professional. It must complement the product, not overpower it. Critically, it must AVOID the look of generic, busy, or low-quality stock photos.
3.  **The Creative Palette (CRITICAL CHOICE):** You MUST choose ONLY ONE of the following design categories that best fits the product's aesthetic:
    *   **A) Sophisticated Color Gradient:** Smooth, multi-tone color transitions.
    *   **B) Minimalist Material Surface:** Clean surfaces like brushed metal, matte plastic, fine paper, smooth stone, or concrete.
    *   **C) Abstract Light & Shadow:** Soft bokeh effects, subtle caustics, gentle light rays, or minimalist geometric patterns.
    *   **D) High-Quality Photorealistic Texture:** Realistic and clean textures like light wood grain, linen fabric, etc. (Use with caution, only if it truly fits).
4.  **Composition for Cropping:** The prompt must describe a **uniform, edge-to-edge background**. It is CRITICAL to avoid any single, centered object or element that would be lost during cropping. If any small, discrete icon is used, explicitly state it should be placed on the **far left or far right**.
5.  **Context Neutrality Mandate:** The final T2I prompt must describe ONLY the background/texture itself. It must be **100% "context-neutral" and MUST NOT contain any words that name, describe, or allude to the original product** (e.g., "phone case", "bottle", "designed for...", "to complement a..."). The product analysis is for your internal design process ONLY; it must be completely scrubbed from the final output.
6.  **Strict Prohibitions:** The prompt MUST NOT request any text, letters, numbers, or logos under any circumstances.
7.  **Technical Specification:** The prompt should implicitly or explicitly guide the AI towards a photorealistic, high-resolution, and square-format output. Phrases like "Macro photography," "Photorealistic," "Studio lighting," and "square format" are highly effective.

===== CHAIN OF THOUGHT (INTERNAL ONLY, DO NOT OUTPUT) =====
1.  **Step 1: Analyze Product Aesthetic:** First, deeply analyze the provided image. What is the product? What is its style (e.g., tech, organic, luxury, utilitarian)? What are the dominant colors and materials (e.g., white plastic, blue packaging, metallic elements)?
2.  **Step 2: Choose from the Creative Palette:** Based on your analysis, consciously select the BEST category from Rule #3.
3.  **Step 3: Develop the Core Concept:** Within the chosen category, develop a specific idea. If you chose "Material," what kind? "Brushed aluminum." What color? "Cool gray with a hint of blue."
4.  **Step 4: Construct & Sanitize the T2I Prompt:** Assemble a highly descriptive, single-paragraph prompt. **Crucially, perform a final check to scour the prompt and remove any direct or indirect mention of the original product, ensuring it is purely a background description.** Ensure it adheres to all rules.

===== OUTPUT REQUIREMENT =====
Output a single, continuous paragraph of pure natural language description, ready for a text-to-image model.

===== BUILT-IN EXAMPLE (FOR YOUR REFERENCE) =====
**Input Image (for context):** A photo of a modern, stainless steel water bottle with a minimalist logo. The aesthetic is clean, healthy, and hydrated.

**Example of an Excellent Output Prompt (choosing Category A: Gradient):**
Photorealistic studio background featuring a smooth, clean gradient transitioning from a soft baby blue to a pure white. The lighting is bright and even, creating a feeling of freshness and hydration. Minimalist, uncluttered, professional product backdrop, super high-resolution, square format.

**Example of an Excellent Output Prompt (choosing Category B: Material):**
Photorealistic macro shot of a clean, horizontal brushed aluminum metal surface. A subtle, soft cool blue light wash illuminates the texture from the side, creating a sophisticated and modern tech aesthetic. The background is completely uniform and minimalist. Professional studio lighting, ultra-high detail, square format.
"""


def send_request(item):
    image_url = item['mainimageurl']
    src_image = load_or_download_image(image_url)
    selling_points = json.loads(item["selling_points"])

    # 1. prompt生成
    prompt_bg = gpt_client.make_image_request("", poster_texture_map, [], image_urls=[image_url], max_tokens=4000, timeout=60)
    item["prompt"] = prompt_bg

    # 2. 图像生成
    result = gemini_25_flash_image_gen(
        prompt_bg,
        image_urls=[],
        image_pils=[],
        specify_gen_ratio=True,
        ratio="1:1",
        model_name="gemini-2.5-flash-image",
        ak="BpaRzJoHfD4aR28PbpiLAwMy3EBb4b1d_GPT_AK",
        max_token=3000,
    )
    gen_url1 = save_tos(encode_pil_bytes(result["image"], False), headers={"Content-Type": "image/jpeg"})
    res_img = result["image"]

    # 3. 粘贴上下banner
    add_image = add_percentage_padding(resize_image_by_short_side(src_image), res_img, padding=[0.166, 0.167, 0, 0])
    gen_url2 = save_tos(encode_pil_bytes(add_image, False), headers={"Content-Type": "image/jpeg"})
    item["gen_url"] = [gen_url1, gen_url2]

    # 4. html渲染
    addon = HtmlAddOn()
    # Custom textbox_pos: [center_x, center_y, width, height] (normalized)
    # Position at bottom: center_x=0.5, center_y=0.8, width=0.8, height=0.15
    textbox_poses = [[0.5, 0.0625, 1.0, 0.125], [0.5, 0.9375, 1.0, 0.125]]
    try:
        result_addon = addon.process_single_image_multi_selling_points(
            image_url=gen_url2,
            selling_points=[SellingPointItem(text=sp, textbox_pos=pos) for sp, pos in zip(selling_points[:2], textbox_poses)],
            image=add_image,
            img_wh=add_image.size,
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
