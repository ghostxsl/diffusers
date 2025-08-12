# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import exists
import argparse
import math
import json
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageCms
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.utils import load_csv_or_xlsx_to_dict, load_file, json_save, get_bbox_from_mask
from diffusers.data.outer_vos_tools import load_or_download_image
from diffusers.data.byted.clients.azure_mllm import MLLMClient
from overpass_ad_creative_qwen_image_v1.clients.rpc.ad_creative_qwen_image_v1 import AdCreativeQwen_Image_V1Client


ocr_client = AdCreativeQwen_Image_V1Client(psm='ad.creative.psa', cluster='ocr', idc='sg1')
gpt_client = MLLMClient(model_name="gemini-2.5-flash", api_key="f1d66mV3iH5c651R0diqfBcmh1qzAo8I_GPT_AK")


product_categories = [
    "Automotive & Motorcycle",
    "Baby & Maternity",
    "Beauty & Personal Care",
    "Books, Magazines & Audio",
    "Collectibles",
    "Computers & Office Equipment",
    "Fashion Accessories",
    "Food & Beverages",
    "Health",
    "Home Improvement",
    "Home Supplies",
    "Household Appliances",
    "Jewelry Accessories & Derivatives",
    "Kids' Fashion",
    "Kitchenware",
    "Luggage & Bags",
    "Menswear & Underwear",
    "Muslim Fashion",
    "Other",
    "Pet Supplies",
    "Phones & Electronics",
    "Shoes",
    "Sports & Outdoor",
    "Textiles & Soft Furnishings",
    "Tools & Hardware",
    "Toys & Hobbies",
    "Womenswear & Underwear"
]


PROMPT = """You are a professional image tagging assistant. Strictly follow the requirements below to analyze the user-provided image and output ONLY a valid JSON object (no extra text, explanations, or formatting changes).

## 1. Tag Definition & Judgment Standards
### 1.1 subject_classification
Select **one exact category** from the list (do not modify or add new categories). Judgment basis is clarified as follows:
1. "Automotive & Motorcycle": Automotive vehicles (e.g., cars, trucks), motorcycles, their core parts (e.g., car engines, motorcycle frames), and dedicated accessories (e.g., car seat covers, motorcycle helmets).
2. "Beauty & Health": Beauty products (e.g., lipstick, eyeshadow palettes, facial masks), personal care items (e.g., shampoo, toothpaste, body wash), and health-related products (e.g., vitamin tablets, fitness trackers for health monitoring) – excluding professional medical devices (e.g., thermometers, blood pressure monitors).
3. "Culture, Education & Entertainment": Books, magazines, audio products (e.g., music CDs, audiobook USB drives, audio storage devices), collectibles (e.g., commemorative stamps, limited-edition anime figurines), and entertainment items (e.g., board games, movie DVDs) – excluding sports/outdoor products and electronic entertainment devices (classified under "Phones & Electronics").
4. "Computers & Office Equipment": Computers (e.g., laptops, desktops), large office equipment (e.g., printers, scanners), office stationery (e.g., pens, notebooks, staplers), and their dedicated accessories (e.g., laptop sleeves, printer cartridges).
5. "Fashion Accessories": Fashion embellishments and functional accessories (e.g., scarves, leather belts, hats, sunglasses, gloves, hair clips) – excluding jewelry and derivatives (classified under "Jewelry Accessories & Derivatives").
6. "Food & Beverages": All human-edible food (e.g., potato chips, fresh apples, canned tuna) and beverages (e.g., cola, milk, red wine) – excluding pet food.
7. "Furniture": Indoor and outdoor furniture designed for residential, commercial, or leisure use (e.g., sofas, dining tables, beds, wardrobes, coffee tables, office desks, patio chairs) – including integral components and dedicated accessories (e.g., sofa cushions, table legs, drawer handles).
8. "Home Essentials & Tools": Non-furniture textiles & soft furnishings (e.g., curtains, bed sheets, throw pillows), home supplies (e.g., laundry detergent, toilet paper), home improvement tools (e.g., manual screwdrivers, electric drills), and hardware (e.g., screws, nuts) – excluding Furniture, Household Appliances, and Kitchenware (all classified as separate categories).
9. "Household Appliances": Electric/electronic home devices for daily living, cooking, cleaning, or comfort (e.g., refrigerators, microwaves, electric kettles, washing machines, air conditioners, vacuum cleaners, rice cookers) – including their dedicated accessories (e.g., microwave turntables, vacuum cleaner filters, rice cooker inner pots).
10. "Jewelry Accessories & Derivatives": Decorative accessories made of various materials (e.g., necklaces, earrings, bracelets, rings, brooches) and their derivative products (e.g., jewelry boxes, cleaning kits, replacement parts such as chains, clasps).
11. "Kitchenware": Tools, utensils, and containers for kitchen food preparation, cooking, serving, or storage (e.g., non-stick pots, chef’s knives, cutting boards, frying pans, dinnerware sets, food storage containers, measuring cups) – excluding electric/electronic kitchen appliances (classified under "Household Appliances").
12. "Luggage & Bags": Bags and luggage for daily use, travel, or storage (e.g., backpacks, suitcases, handbags, tote bags, crossbody bags, duffel bags).
13. "Maternity & Kids": Maternal supplies (e.g., pregnancy leggings, baby strollers), kids’ fashion (e.g., children’s dresses, toddler shoes), and kids’ toys & hobbies (e.g., plastic building blocks, plush dolls).
14. "Menswear & Underwear": Men’s clothing (e.g., shirts, jackets, trousers) and men’s underwear (e.g., briefs, boxers, undershirts).
15. "Muslim Fashion": Muslim-specific clothing and apparel (e.g., abayas, hijabs, thobes) – including corresponding undergarments for men and women.
16. "Pet Supplies": Pet-specific products, including pet food (e.g., dog kibble, cat canned food), pet toys (e.g., rubber chew toys), pet grooming tools (e.g., pet brushes), pet beds, and pet gear (e.g., litter boxes, pet collars, pet carriers, pet fences).
17. "Phones & Electronics": Mobile phones, consumer electronics (e.g., wireless headphones, digital cameras, smartwatches, tablets, smart speakers), and their dedicated accessories (e.g., phone cases, charging cables, screen protectors) – excluding office electronic equipment (classified under "Computers & Office Equipment").
18. "Shoes": Adult shoes for daily wear, casual use, or specific occasions (e.g., sneakers, leather shoes, high heels, sandals, boots) – excluding kids’ shoes (classified under "Maternity & Kids") and sports-specific shoes (classified under "Sports & Outdoor").
19. "Sports & Outdoor": Sports equipment (e.g., basketballs, yoga mats, tennis rackets), outdoor gear (e.g., camping tents, hiking boots), and functional apparel for sports/outdoor activities (e.g., running leggings, swimming trunks, hiking jackets) – including sports-specific shoes (e.g., running shoes, basketball shoes, hiking boots) and excluding daily casual sport-style clothing (classified under "Menswear/Womenswear").
20. "Womenswear & Underwear": Women’s clothing (e.g., dresses, blouses, skirts) and women’s underwear (e.g., bras, panties, shapewear).
21. "Other": Items that do not fit any above categories (e.g., industrial machine parts, unclassified miscellaneous items such as random hardware scraps).

### 1.2 is_clothing_product
Output boolean value "true" or "false" (lowercase, no quotes). Judgment standard:
Only when the product subject in the image is upper garments, lower garments, or one-piece clothing (e.g., shirts, trousers, dresses, jumpsuits) is it "true". Excludes accessories such as socks, hats, shoes, bags, jewelry, scarves, and belts. It is "false" for non-clothing products or images with no recognizable products.

### 1.3 background_classification
Select **only one** from: "Solid-color Background", "Virtual Background", "Real Background" (use exact strings, no variations). Definitions are clarified:
- "Solid-color Background": Single-color background with no texture, patterns, or additional elements (e.g., pure white, solid black, flat gray backgrounds).
- "Virtual Background": Non-real shooting backgrounds synthesized through digital technology, such as 3D-rendered scenes, cartoon-style backgrounds, and scenes generated by digital effects.
- "Real Background": Real environment backgrounds captured in actual shooting, such as real indoor scenes (e.g., living rooms, studios), outdoor scenes (e.g., streets, natural landscapes), and real prop scenes (e.g., real wooden tables, brick walls).

### 1.4 has_product
Output boolean value "true" or "false" (lowercase, no quotes). Judgment standard:
The image contains a **product subject** with prominent, unambiguous features (e.g., clear product shape, distinct commercial packaging, obvious association with a specific usage scenario) – these features must align with the characteristics of standard e-commerce product images.

### 1.5 product_fully_visible
Output boolean value "true" or "false" (lowercase, no quotes). Core criterion:
The product (including all its key structural parts) is 100% fully displayed with no truncation. Judge by whether the product’s edges or key parts are cut off by the image frame (e.g., a laptop with a truncated screen edge = false; a T-shirt with no cut-off sleeves/hem = true; a cup with a truncated handle = false).

### 1.6 product_quantity
Select **only one** from: "single", "multiple", "not_applicable" (use exact strings, no variations). Definitions are clarified:
- "single": Image contains one individual, distinguishable product (e.g., one smartphone, one T-shirt, one coffee mug).
- "multiple": Image contains two or more identical/different products (e.g., three notebooks, one laptop + one mouse, two bottles of shampoo).
- "not_applicable": Image has no recognizable product (i.e., when `has_product` is "false") OR product cannot be quantified (e.g., shapeless bulk goods like a pile of flour, a heap of rice with no clear individual boundaries).

### 1.7 human_parts_visible
Identify any visible human presence and output a single, most representative string tag. The selection process must follow a strict logical order. This tag does not consider the face.
Judgment Logic & Hierarchy: First, determine the number of people in the image:
- If no human is present, the tag must be "none".
- If multiple people (two or more) are present, the tag must be "other" (e.g., an image displaying couple's outfits or a group of models).

If only a single person is present, you must select the tag based on the following strict priority order. Choose the first tag from the top of the list that accurately describes the visible portion:
1. "full_body": The person is visible from head to toe, or at least 90% of their body is shown, e.g., a model showcasing a full-length dress or a complete suit.
2. "upper_body": The upper part of the body is visible, typically from the waist up (includes torso, shoulders, arms), e.g., displaying a T-shirt, necklace, or jacket.
3. "lower_body": The lower part of the body is visible, typically from the waist down, and must include the thigh, e.g., showing how jeans, a skirt, or shorts fit.
4. "hand_only": A close-up shot showing only the hand(s) or wrist(s), e.g., a hand displaying a ring, watch, or holding a small product like nail polish. If a significant part of the arm is visible, "upper_body" takes priority.
5. "foot": A close-up shot showing the foot, ankle, or calf area, e.g., a focused view on shoes, socks, or an anklet. If the thigh is also visible, "lower_body" takes priority.
6. "other": Use this tag if a single person is visible, but the part shown does not fit any of the specific categories above (e.g., a shot of only an elbow, a shoulder, or the back of the head).

Possible Output Tags: "full_body", "upper_body", "lower_body", "hand_only", "foot", "other", "none".

### 1.8 has_human_face
Output boolean value "true" or "false" (lowercase, no quotes). Judgment standard:
"True" only if the image contains a real human face (excluding cartoon/pattern/doll faces) with at least two clear core features (eyes, nose, mouth, cheeks, eyebrows) visible – occlusion by items like hats, sunglasses, or masks is allowed as long as the face itself is recognizable. "False" for single features, heavily blurred/unrecognizable features, non-real faces, or no face-related elements at all.

### 1.9 can_stand_upright
Output boolean value "true" or "false" (lowercase, no quotes). Judgment standard (including special cases):
Determine whether the **main objects in the image** (products, humans, or other key subjects) can remain stably upright or placed on a flat ground surface without any external support (e.g., no hands holding, no brackets, no adhesive) and without toppling over. Key factors to consider:
- Object shape: Whether the object has a stable support surface (e.g., a cube has a flat, large support surface = favorable for stability; a pencil standing on its tip has a tiny support surface = unfavorable).
- Center of gravity: Whether the object’s center of gravity falls within the range of its support surface (e.g., an upright chair with gravity centered over its four legs = stable; a tilted chair with gravity outside the leg range = unstable).
- Placement angle: Whether the object is placed at an angle that causes gravity to shift (e.g., a water bottle placed vertically = stable; a water bottle tilted 45° = unstable).

**Special cases (must be applied first if applicable):**
1. If the image is shot from a **top-down perspective** (bird’s-eye view), and the main object can be stably placed/rested on the ground when judged from this perspective (e.g., a plate viewed from above – its flat bottom can stably contact the ground = true), the object is considered stable.
2. If the main object is a **standing human figure** (with upright posture, feet contacting the ground, no leaning/crouching/lying), the object is considered stable = true.

### 1.10 text_content_judgment
Output boolean value "true" or "false" (lowercase, no quotes). Judgment scope:
Only text **outside the subject** (words, numbers, symbols, logos, etc.) is counted. Ignore text directly printed, engraved, or attached to the subject itself (e.g., brand logos printed on a T-shirt, text on a book’s cover, serial numbers on a phone = not counted; a price tag next to a product, a logo on the background wall = counted).

### 1.11 text_occlusion_judgment
Output boolean value "true" or "false" (lowercase, no quotes). Judgment standard:
Determine whether the text **outside the product subject** (as defined in 1.10) obscures any part of the product (e.g., a price tag covering a product’s button = true; text on the background wall with no overlap with the product = false). If the image has no recognizable product (`has_product` = false) OR no text outside the subject (`text_content_judgment` = false), return false.

### 1.12 is_poster
Output true or false. This tag identifies images that are designed as posters for promotion, advertisement, or announcements.
Key Rule: The image must contain significant textual elements (copy). An image without any text is always false.
Judgment Standard: The text is not just a simple label or brand logo, but is integrated with visuals in a deliberate design layout to convey a specific marketing message, event details, or a slogan.
Examples (true): Movie posters, concert flyers, product sale advertisements, public service announcements.
Examples (false): A photo of a t-shirt with a text logo, an image of a product with a small watermark, any image with no text at all.

### 1.13 image_clarity
Select **only one** from: "Clear", "Blurry" (use exact strings, no variations). Judgment standard:
Based on the sharpness of the product (core subject) – not the background. Examples: clear texture of a fabric product, distinguishable buttons on a shirt, readable small text on product packaging = "Clear"; blurred product edges, unrecognizable product details (e.g., indistinguishable fabric texture), unreadable text on the product = "Blurry".

### 1.14 grid_style_image_collage
Output true or false. This tag identifies images created by combining two or more separate images into a single grid layout.
Key Criterion: Look for clear, straight dividing lines or "seams" that separate distinct sub-images.
Includes Simple Cases: This explicitly includes simple layouts such as a two-panel side-by-side split (left/right) or a two-panel top/bottom split.
Important Exclusion: Do NOT tag true for "split-screen" designs that show different views of the same single subject within one continuous frame (e.g., left side shows the front of a dress, right side shows the back of the same dress). This is considered a design layout, not a collage of separate images.
Simple Test: If you could theoretically "cut" along the dividing lines and get multiple, completely independent pictures, then it is true.

## 2. Required JSON Structure
Include all 14 core tags (no missing keys). Use exact key names, no changes; key order aligns with the "Tag Definition & Judgment Standards" sequence above:
{
    "subject_classification": "SELECTED_CATEGORY",
    "is_clothing_product": BOOLEAN_VALUE,
    "background_classification": "SELECTED_BACKGROUND",
    "has_product": BOOLEAN_VALUE,
    "product_fully_visible": BOOLEAN_VALUE,
    "product_quantity": "SELECTED_QUANTITY",
    "human_parts_visible": SELECTED_PARTS,
    "has_human_face": BOOLEAN_VALUE,
    "can_stand_upright": BOOLEAN_VALUE,
    "text_content_judgment": BOOLEAN_VALUE,
    "text_occlusion_judgment": BOOLEAN_VALUE,
    "is_poster": BOOLEAN_VALUE,
    "image_clarity": "SELECTED_CLARITY",
    "grid_style_image_collage": BOOLEAN_VALUE
}

## 3. Example of Valid Output (Aligned with Integrated Structure)
{
    "subject_classification": "Womenswear & Underwear",
    "is_clothing_product": true,
    "background_classification": "Real Background",
    "has_product": true,
    "product_fully_visible": true,
    "product_quantity": "single",
    "human_parts_visible": "upper_body",
    "has_human_face": false,
    "can_stand_upright": false,
    "text_content_judgment": true,
    "text_occlusion_judgment": false,
    "is_poster": false,
    "image_clarity": "Clear",
    "grid_style_image_collage": false
}

## 4. Mandatory Compliance Rules
- No trailing commas in the JSON object.
- Use double quotes (") for all strings and keys (do not use single quotes).
- Output ONLY the JSON object (no preamble, comments, follow-up text, or formatting adjustments).
- For uncertain items, select the most appropriate option based on the above clarified definitions (do not leave fields empty or use ambiguous values).
- For `can_stand_upright`, prioritize special cases (top-down perspective, standing human) over general shape/gravity/angle judgments when applicable."""


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file", default=None, type=str)
    parser.add_argument(
        "--output_file", default="output.json", type=str)

    parser.add_argument("--num_workers", default=10, type=int)

    args = parser.parse_args()

    return args


def ocr_image_text_detection(image_url, text="", remove_subject=True):
    code, msg, resp = ocr_client.AiModel(
        request_body=json.dumps(
            {
                "image_url": image_url,
                "check_text": text,
                "remove_subject": remove_subject,
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


def merge_intervals(intervals):
    # 处理空输入
    if not intervals:
        return []

    # 按区间的起始值从小到大排序
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    # 初始化结果列表，放入第一个区间
    merged = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        # 取出结果列表中最后一个已合并的区间
        last = merged[-1]

        # 如果当前区间的起始值 <= 最后一个区间的结束值，说明有重叠，需要合并
        if current[0] <= last[1]:
            # 合并后的区间结束值取两个区间的最大值
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            # 无重叠，直接加入结果列表
            merged.append(current)

    return merged


def calculate_text_height_ratio(image_url, ocr_result):
    img = load_or_download_image(image_url)
    h = img.height
    # 计算文本行高占比
    rec_boxes = ocr_result['rec_boxes']
    height_intervals = [(a[1], a[3]) for a in rec_boxes]
    height_intervals = merge_intervals(height_intervals)
    text_h = sum([a[1] - a[0] for a in height_intervals])
    return text_h / h


def get_subject_ratio(mask_url):
    mask_image = load_or_download_image(mask_url)
    mask_image = np.array(mask_image.convert('L'))
    bbox = get_bbox_from_mask(mask_image)
    x1, y1, x2, y2 = bbox
    w_ratio = round((x2 - x1) / mask_image.shape[1], 4)
    h_ratio = round((y2 - y1) / mask_image.shape[0], 4)
    subject_ratio = max(w_ratio, h_ratio)
    return subject_ratio, bbox


def calculate_color_difference(image_url, mask_url) -> float:
    """
    计算mask内外区域的颜色差异度（贴近人眼感知）
    # ΔE < 2: 人眼几乎无法区分
    # 2 < ΔE < 10: 有明显差异但不显著
    # ΔE > 10: 差异明显，容易区分

    参数:
        image_url: 图像URL或本地路径
        mask_url: 掩码图像URL或本地路径

    返回:
        float - 颜色差异度（值越大，内外颜色越容易区分，范围通常0-100+）
    """
    img = load_or_download_image(image_url)
    mask = load_or_download_image(mask_url)
    # 确保图像和掩码尺寸一致
    if img.size != mask.size:
        raise ValueError("图像和掩码的尺寸必须一致")

    # 确保图像是RGB模式（处理RGBA等情况）
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 转换掩码为二值化numpy数组（1表示mask内，0表示mask外）
    mask_gray = mask.convert("L")
    mask_np = np.array(mask_gray)
    mask_binary = (mask_np > 0).astype(np.uint8)
    total_pixels = mask_binary.size

    # 统计mask内外的像素数量（避免空区域）
    inner_count = np.sum(mask_binary)
    outer_count = total_pixels - inner_count
    if inner_count < 100 or outer_count < 100:  # 过滤像素太少的区域（避免偶然误差）
        raise ValueError("mask内或外的有效像素太少，无法计算差异度")

    # 将RGB图像转换到CIE Lab颜色空间（更符合人眼对颜色差异的感知）
    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")
    transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
    img_lab = ImageCms.applyTransform(img, transform)
    img_lab_np = np.array(img_lab, dtype=np.float32)  # shape: (H, W, 3)，通道：L(亮度), a(绿-红), b(蓝-黄)

    # 提取mask内和外的像素（Lab空间）
    inner_pixels = img_lab_np[mask_binary == 1]  # shape: (N, 3)
    outer_pixels = img_lab_np[mask_binary == 0]  # shape: (M, 3)

    # -------------------------- 新增：剔除极端值 --------------------------
    def filter_extreme_values(pixels, percentile=2):
        """剔除像素中按L通道排序的前后指定百分比极端值"""
        n = len(pixels)
        if n == 0:
            return pixels

        # 计算需要剔除的像素数量（四舍五入取整）
        k = int(round(n * percentile / 100))
        if k <= 0:  # 当像素数量较少时，可能无需剔除
            return pixels

        # 按L通道（亮度）排序（人眼对亮度更敏感）
        sorted_indices = np.argsort(pixels[:, 0])  # 按第一列（L通道）排序
        # 剔除前k个和后k个极端值
        filtered = pixels[sorted_indices[k : n - k]]
        return filtered

    # 对背景区域分别剔除前后2%的极端值
    filtered_outer = filter_extreme_values(outer_pixels, percentile=2)

    # 检查过滤后像素数量是否足够
    if len(filtered_outer) < 100:
        raise ValueError("剔除极端值后，有效像素数量不足，无法计算差异度")
    # ----------------------------------------------------------------------

    # 计算过滤后区域的平均颜色（Lab空间）
    inner_mean = np.mean(inner_pixels, axis=0)  # (L1, a1, b1)
    outer_mean = np.mean(filtered_outer, axis=0)  # (L2, a2, b2)

    # 计算Lab空间中两个颜色的欧氏距离（人眼感知的差异度）
    delta_e = math.sqrt(
        (inner_mean[0] - outer_mean[0]) ** 2
        + (inner_mean[1] - outer_mean[1]) ** 2
        + (inner_mean[2] - outer_mean[2]) ** 2
    )

    return delta_e


def process_data_ocr_subject_color(image_url, key="src"):
    # 1. ocr推理
    result = ocr_image_text_detection(image_url)
    # 1.1 按空格分词
    ocr_text = []
    for line in result["ocr_text"]:
        ocr_text += line.split(" ")
    out = {
        f"{key}_mask_url": result["mask_url"],
        f"{key}_ocr_text": ocr_text,
        f"{key}_ocr_result": result["ocr_result"],
    }
    height_ratio = calculate_text_height_ratio(image_url, result["ocr_result"])
    out[f"{key}_text_height_ratio"] = height_ratio
    # 2. 计算主体占比
    subject_ratio, bbox = get_subject_ratio(result["mask_url"])
    out[f"{key}_subject_ratio"] = subject_ratio
    out[f"{key}_bbox"] = bbox
    # 3. 计算对比度
    delta_e = calculate_color_difference(image_url, result["mask_url"])
    out[f"{key}_delta_e"] = delta_e
    return out


def label_single_image(image_url):
    result = gpt_client.make_image_json_request("", PROMPT, [], [image_url], 4095, timeout=60)
    return result


def send_request(item):
    src_url = item["original_image_url"]
    gen_url = item["generated_image_url"]

    # 1. ocr检测 + 主体 + 对比度
    res_src_ocr = process_data_ocr_subject_color(src_url, "src")
    item.update(res_src_ocr)
    res_gen_ocr = process_data_ocr_subject_color(gen_url, "gen")
    item.update(res_gen_ocr)

    # 2. 图像打标签
    item["src_label"] = label_single_image(src_url)
    item["gen_label"] = label_single_image(gen_url)

    return item


def main(args):
    # load dataset
    try:
        data_list = load_csv_or_xlsx_to_dict(args.input_file)
    except Exception:
        data_list = load_file(args.input_file)

    if exists(args.output_file):
        done_data = load_file(args.output_file)
        done_data = {a['generated_image_url']: a for a in done_data}
        out = list(done_data.values())
        data_list = [a for a in data_list if a["generated_image_url"] not in done_data]
    else:
        out = []

    error_results = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_url = {executor.submit(send_request, item): item for item in data_list}
        with tqdm(total=len(data_list)) as pbar:
            for future in concurrent.futures.as_completed(future_to_url):
                item = future_to_url[future]
                try:
                    result_item = future.result()
                    pbar.update(1)  # Update progress bar

                    out.append(result_item)
                    if len(out) % 10 == 0:
                        json_save(out, args.output_file)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    error_results.append({'image_url': item, 'error_reason': str(e)})

    # Final Save
    json_save(out, args.output_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
