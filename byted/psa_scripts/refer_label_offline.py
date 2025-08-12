# @author: wilson.xu.
from os.path import exists
import argparse
from tqdm import tqdm
import json
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.byted.clients.gemini_mllm import mllm_make_image_request
from diffusers.data.utils import load_csv_or_xlsx_to_dict, load_file, csv_save


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="/mlx_devbox/users/xushangliang/playground/label_psa_reference_pic.csv", type=str)
    parser.add_argument(
        "--output_file", default="1_label_psa_reference_pic.csv", type=str)

    parser.add_argument("--num_workers", default=10, type=int)

    args = parser.parse_args()
    return args


PROMPT = """# Prompt - Tagging Reference Ad Images for E-commerce Design Retrieval

You are building a reference library of high-performing e-commerce advertisement images.
Your task is to analyze a single ad image and assign a small set of standardized tags that are useful for later creative retrieval and design transfer.

The purpose of these tags is **not** business catalog management.
The purpose is to support:
1. reference image retrieval,
2. visual pattern transfer,
3. later LLM-based background redesign for other products.

Therefore, focus on how the product is visually expressed in advertising, not on overly fine-grained retail taxonomy.

---

## Tagging Instructions

Analyze the image and output tags for the following dimensions:

### 1. visual_product_group
Select exactly **one** category from this fixed list:

- Jewelry & Small Luxury
- Beauty, Personal Care & Wellness
- Fashion & Wearables
- Home Living
- Furniture & Large Home Fixtures
- Kitchen & Dining
- Electronics & Appliances
- Food, Beverage & Pet Consumables
- Kids, Toys & Pet Supplies
- Automotive, Sports, Tools & Functional Goods
- Other

Use the category that best reflects the product’s **advertising presentation logic**, not necessarily the most detailed retail category.

Guidance:
- Jewelry, rings, earrings, necklaces, bracelets, brooches, jewelry boxes, small premium gift-like items → Jewelry & Small Luxury
- Skincare, cosmetics, toothpaste, whitening pens, shampoo, supplements, wellness products → Beauty, Personal Care & Wellness
- Clothing, shoes, bags, hats, scarves, sunglasses, belts → Fashion & Wearables
- Home supplies, soft furnishings, daily home products, decor objects, candles, storage, cleaning products → Home Living
- Sofa, bed, table, chair, cabinet and other furniture → Furniture & Large Home Fixtures
- Cookware, knives, plates, kitchen containers, non-electric kitchen tools → Kitchen & Dining
- Phones, tablets, headphones, smart devices, office electronics, household appliances → Electronics & Appliances
- Snacks, drinks, packaged food, pet food, edible consumables → Food, Beverage & Pet Consumables
- Baby products, toys, children’s goods, non-food pet accessories → Kids, Toys & Pet Supplies
- Automotive goods, sports equipment, tools, hardware, outdoor functional goods → Automotive, Sports, Tools & Functional Goods

---

### 2. visual_style
Select **one or two** styles from this fixed list:

- Minimal Clean
- Premium Luxury
- Soft Feminine
- Clinical Professional
- Natural Organic
- Lifestyle Warm
- Modern Tech
- Playful Bright
- Giftable Festive
- Fresh Refreshing
- Rich Editorial
- Cute Friendly

Choose the style(s) that best describe the overall advertising look and visual appeal of the image.

Brief guidance:
- Minimal Clean: clean, sparse, simple, white space, controlled
- Premium Luxury: elegant, expensive-looking, refined, metallic, polished
- Soft Feminine: gentle, creamy, pastel, soft and feminine
- Clinical Professional: scientific, hygienic, efficacy-oriented, trustworthy
- Natural Organic: botanical, earthy, fresh, natural materials
- Lifestyle Warm: homey, warm, relatable, daily-life ambience
- Modern Tech: cool-toned, sleek, futuristic, digital, structured
- Playful Bright: colorful, cheerful, youthful, eye-catching
- Giftable Festive: celebratory, present-like, ceremonial, holiday-oriented
- Fresh Refreshing: airy, water-light, minty, cooling, transparent
- Rich Editorial: magazine-like, dramatic, highly styled, design-forward
- Cute Friendly: adorable, soft, childlike, pet-friendly, approachable

---

### 3. scene_construction_level
Select exactly **one** from this fixed list:

- Plain / Pure
- Graphic / Abstract
- Styled Surface
- Environmental Scene

Definitions:
- Plain / Pure: plain white, plain solid color, very simple gradient, almost no scene structure
- Graphic / Abstract: geometric shapes, arcs, lines, abstract blocks, design graphics, but not a real physical setting
- Styled Surface: pedestal, tabletop, platform, fabric surface, tray, mirror surface, or any staged support plane without a full real-life room scene
- Environmental Scene: recognizable real environment such as bathroom, kitchen, bedroom, vanity, living room, outdoor setting, studio room-like space

---

### 4. visual_busyness
Select exactly **one** from this fixed list:

- Low
- Medium
- High

Definitions:
- Low: very few elements, highly restrained, strong visual simplicity
- Medium: some supporting elements and layering, but still controlled
- High: many visible elements, props, decorations, or strong visual activity

Judge by the overall visual information load, not just whether the image is “pretty.”

---

### 5. focal_presentation
Select exactly **one** from this fixed list:

- Isolated Product Focus
- Product with Simple Support
- Product in Styled Display
- Product in Lifestyle Context

Definitions:
- Isolated Product Focus: product alone, almost fully isolated, little or no support
- Product with Simple Support: product remains dominant, with minimal supporting base or light visual framing
- Product in Styled Display: clearly art-directed display with pedestal, props, decorative composition
- Product in Lifestyle Context: product presented as part of an everyday or contextual scene

This tag is meant to describe how the product is presented to the viewer.

---

### 6. is_clothing_product
Output boolean value `true` or `false` (lowercase, no quotes).

Judgment standard:
- `true`: Only when the product subject in the image is upper garments, lower garments, or one-piece clothing (e.g., shirts, trousers, dresses, jumpsuits). 
- `false`: Excludes accessories such as socks, hats, shoes, bags, jewelry, scarves, and belts. It is also `false` for non-clothing products or images with no recognizable products.

---

### 7. background_classification
Select exactly **one** from this fixed list (use exact strings, no variations):

- Solid-color Background
- Virtual Background
- Real Background

Definitions:
- Solid-color Background: Single-color background with no texture, patterns, or additional elements (e.g., pure white, solid black, flat gray backgrounds).
- Virtual Background: Non-real shooting backgrounds synthesized through digital technology, such as 3D-rendered scenes, cartoon-style backgrounds, and scenes generated by digital effects.
- Real Background: Real environment backgrounds captured in actual shooting, such as real indoor scenes (e.g., living rooms, studios), outdoor scenes (e.g., streets, natural landscapes), and real prop scenes (e.g., real wooden tables, brick walls).

---

### 8. is_poster
Output boolean value `true` or `false` (lowercase, no quotes).

This tag identifies images that are designed as posters for promotion, advertisement, or announcements.

Definitions & Key Rules:
- The image **must** contain significant textual elements (copy). An image without any text is always `false`.
- `true`: The text is not just a simple label or brand logo, but is integrated with visuals in a deliberate design layout to convey a specific marketing message, event details, or a slogan. (Examples: Movie posters, concert flyers, product sale advertisements, public service announcements).
- `false`: A photo of a t-shirt with a text logo, an image of a product with a small watermark, or any image with no text at all.

---

## Important Rules
- Do not guess hidden product features not visible in the image.
- Focus on the visible advertising presentation.
- Use only the provided labels; do not invent new categories.
- If multiple labels seem possible, choose the one that is most useful for later visual retrieval and design transfer.
- Be consistent and conservative.

---

## Output Format
Return your answer in valid JSON only, using this exact schema:

{
  "visual_product_group": "",
  "visual_style": ["", ""],
  "scene_construction_level": "",
  "visual_busyness": "",
  "focal_presentation": "",
  "is_clothing_product": false,
  "background_classification": "",
  "is_poster": false
}"""


def send_request(item):
    image_url = item['image_url']
    result_json = mllm_make_image_request(
        PROMPT,
        image_urls=[image_url],
        thinking_budget=2048,
        max_tokens=5000,
        timeout=60,
        temperature=0.1,
        is_json_response=True,
        model_name="gemini-3.1-fl",
        api_key="XpGAiljn5ZhzjpRn8Eo8BRvj1uWcaCmP_GPT_AK",
        base_url="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl",
    )
    item['reference_label'] = json.dumps(result_json, ensure_ascii=False)
    return item


def main(data, dst, max_workers):
    if exists(dst):
        done_data = load_file(dst)
        done_data = {a["image_url"]: a for a in done_data}
        data = [a for a in data if a["image_url"] not in done_data]
        results = list(done_data.values())
    else:
        results = []
    error_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(send_request, item): item for item in data}
        with tqdm(total=len(data)) as pbar:
            for future in concurrent.futures.as_completed(future_to_url):
                item = future_to_url[future]
                try:
                    result_item = future.result()
                    pbar.update(1)  # Update progress bar

                    results.append(result_item)
                    if len(results) % 100 == 0:
                        csv_save(results, dst)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    error_results.append({'item': item, 'error_reason': str(e)})

    csv_save(results, dst)
    print(len(error_results))


if __name__ == "__main__":
    args = parse_args()
    print(f"input_file: {args.input_file}, num_workers: {args.num_workers}")

    data = load_csv_or_xlsx_to_dict(args.input_file)
    sorted_data = sorted(data, key=lambda x: x["dollar_cost"], reverse=True)

    main(sorted_data, args.output_file, args.num_workers)

    print('Done!')
