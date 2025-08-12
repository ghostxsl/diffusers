# @author: wilson.xu.
from os.path import exists
import argparse
from tqdm import tqdm
import json
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.byted.clients.azure_mllm import MLLMClient
from diffusers.data.utils import load_csv_or_xlsx_to_dict, load_file, json_save


# gpt_client = MLLMClient(model_name="gpt-5-mini-2025-08-07")
gpt_client = MLLMClient(model_name="gemini-2.5-flash", api_key="f1d66mV3iH5c651R0diqfBcmh1qzAo8I_GPT_AK")


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="/mnt/bn/creative-algo/xsl/1014_psa_data_1k_test_urls.csv", type=str)
    parser.add_argument(
        "--output_file", default="label_1014_psa_data_1k_test_urls.json", type=str)

    parser.add_argument("--num_workers", default=10, type=int)

    args = parser.parse_args()
    return args


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

### 1.4 human_parts_visible
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

### 1.5 has_human_face
Output boolean value "true" or "false" (lowercase, no quotes). Judgment standard:
"True" only if the image contains a real human face (excluding cartoon/pattern/doll faces) with at least two clear core features (eyes, nose, mouth, cheeks, eyebrows) visible – occlusion by items like hats, sunglasses, or masks is allowed as long as the face itself is recognizable. "False" for single features, heavily blurred/unrecognizable features, non-real faces, or no face-related elements at all.

### 1.6 can_stand_upright
Output boolean value "true" or "false" (lowercase, no quotes). Judgment standard (including special cases):
Determine whether the **main objects in the image** (products, humans, or other key subjects) can remain stably upright or placed on a flat ground surface without any external support (e.g., no hands holding, no brackets, no adhesive) and without toppling over. Key factors to consider:
- Object shape: Whether the object has a stable support surface (e.g., a cube has a flat, large support surface = favorable for stability; a pencil standing on its tip has a tiny support surface = unfavorable).
- Center of gravity: Whether the object’s center of gravity falls within the range of its support surface (e.g., an upright chair with gravity centered over its four legs = stable; a tilted chair with gravity outside the leg range = unstable).
- Placement angle: Whether the object is placed at an angle that causes gravity to shift (e.g., a water bottle placed vertically = stable; a water bottle tilted 45° = unstable).
**Special cases (must be applied first if applicable):**
1. If the image is shot from a **top-down perspective** (bird’s-eye view), and the main object can be stably placed/rested on the ground when judged from this perspective (e.g., a plate viewed from above – its flat bottom can stably contact the ground = true), the object is considered stable.
2. If the main object is a **standing human figure** (with upright posture, feet contacting the ground, no leaning/crouching/lying), the object is considered stable = true.

### 1.7 text_content_judgment
Output boolean value "true" or "false" (lowercase, no quotes). Judgment scope:
Only text **outside the subject** (words, numbers, symbols, logos, etc.) is counted. Ignore text directly printed, engraved, or attached to the subject itself (e.g., brand logos printed on a T-shirt, text on a book’s cover, serial numbers on a phone = not counted; a price tag next to a product, a logo on the background wall = counted).

### 1.8 text_occlusion_judgment
Output boolean value "true" or "false" (lowercase, no quotes). Judgment standard:
Determine whether the text **outside the product subject** (as defined in 1.10) obscures any part of the product (e.g., a price tag covering a product’s button = true; text on the background wall with no overlap with the product = false). If the image has no recognizable product (`has_product` = false) OR no text outside the subject (`text_content_judgment` = false), return false.

### 1.9 is_poster
Output true or false. This tag identifies images that are designed as posters for promotion, advertisement, or announcements.
Key Rule: The image must contain significant textual elements (copy). An image without any text is always false.
Judgment Standard: The text is not just a simple label or brand logo, but is integrated with visuals in a deliberate design layout to convey a specific marketing message, event details, or a slogan.
Examples (true): Movie posters, concert flyers, product sale advertisements, public service announcements.
Examples (false): A photo of a t-shirt with a text logo, an image of a product with a small watermark, any image with no text at all.

### 1.10 grid_style_image_collage
Output true or false. This tag identifies **collages of 2+ independent, complete images** arranged into a single grid layout (e.g., side-by-side, top-bottom, 3×3).
Key Criterions:
1. **Clear dividing lines/seams**: Exists to separate distinct sub-images (you could "cut" along lines to get multiple independent pictures).
2. **Minimal/no promotional text**: Sub-images contain little to no text (or only 1-2 short, non-promotional phrases); excludes designs with large blocks of text copy (e.g., product descriptions, sale info) — these are considered posters, not collages.
3. **Independent sub-images**: Each sub-image is a separate, self-contained work.
Simple Test: Can you cut along lines to get 2+ standalone images *with no shared promotional text/design elements*? If yes → true; else → false.

## 2. Required JSON Structure
Include all 10 core tags (no missing keys). Use exact key names, no changes; key order aligns with the "Tag Definition & Judgment Standards" sequence above:
{
    "subject_classification": "SELECTED_CATEGORY",
    "is_clothing_product": BOOLEAN_VALUE,
    "background_classification": "SELECTED_BACKGROUND",
    "human_parts_visible": SELECTED_PARTS,
    "has_human_face": BOOLEAN_VALUE,
    "can_stand_upright": BOOLEAN_VALUE,
    "text_content_judgment": BOOLEAN_VALUE,
    "text_occlusion_judgment": BOOLEAN_VALUE,
    "is_poster": BOOLEAN_VALUE,
    "grid_style_image_collage": BOOLEAN_VALUE
}

## 3. Example of Valid Output (Aligned with Integrated Structure)
{
    "subject_classification": "Womenswear & Underwear",
    "is_clothing_product": true,
    "background_classification": "Real Background",
    "human_parts_visible": "upper_body",
    "has_human_face": false,
    "can_stand_upright": false,
    "text_content_judgment": true,
    "text_occlusion_judgment": false,
    "is_poster": false,
    "grid_style_image_collage": false
}

## 4. Mandatory Compliance Rules
- No trailing commas in the JSON object.
- Use double quotes (") for all strings and keys (do not use single quotes).
- Output ONLY the JSON object (no preamble, comments, follow-up text, or formatting adjustments).
- For uncertain items, select the most appropriate option based on the above clarified definitions (do not leave fields empty or use ambiguous values).
- For `can_stand_upright`, prioritize special cases (top-down perspective, standing human) over general shape/gravity/angle judgments when applicable."""


def send_request(item):
    image_url = item['ad_url']
    result_json = gpt_client.make_image_json_request(
        "", PROMPT, [], [image_url], max_tokens=4000, timeout=60)
    item['src_label'] = result_json
    return item


def main(data, dst, max_workers):
    if exists(dst):
        done_data = load_file(dst)
        done_data = {a["ad_url"]: a for a in done_data}
        data = [a for a in data if a["ad_url"] not in done_data]
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
                    if len(results) % 10 == 0:
                        json_save(results, dst)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    error_results.append({'item': item, 'error_reason': str(e)})

    json_save(results, dst)
    print(error_results)


if __name__ == "__main__":
    args = parse_args()
    print(f"input_file: {args.input_file}, num_workers: {args.num_workers}")

    data = load_file(args.input_file)

    main(data, args.output_file, args.num_workers)

    print('Done!')
