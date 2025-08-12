# @author: wilson.xu.
import argparse
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.byted.clients.azure_mllm import MLLMClient, gemini_25_flash_image_gen
from diffusers.data.outer_vos_tools import encode_pil_bytes
from diffusers.data.byted.tos import save_tos
from diffusers.data.utils import load_file, json_save


gpt_client = MLLMClient(model_name="gpt-5-mini-2025-08-07")


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="", type=str)
    parser.add_argument(
        "--output_file", default="", type=str)
    parser.add_argument(
        "--num_workers", default=10, type=int)

    args = parser.parse_args()
    return args


addon_prompt = """You are an expert Compositing Art Director AI. Your role is twofold: first, to rigorously analyze user inputs against a set of rules; second, to translate your technical decisions into a concise, elegant, and purely descriptive brief for an image generation model.

**--- CORE PRINCIPLES (Your Internal Logic) ---**
1.  **PRODUCT INTEGRITY:** The [PRODUCT IMAGE] is a fixed, high-resolution asset. It must be preserved with 100% fidelity. Your job is to design *around* it.
2.  **SMART TEXT PLACEMENT:** Text never obscures critical product details, faces, or hands. Prioritize natural negative space.
3.  **ASPECT RATIO LOCK:** The final image's aspect ratio must exactly match the original [PRODUCT IMAGE]. No cropping or distortion.

**--- OUTPUT PHILOSOPHY (Crucial for Avoiding Redundancy) ---**
1.  **DESCRIBE, DON'T PRESCRIBE:** Your output is a **description of the final, finished image**, not a list of rules for how to make it.
    *   **Instead of:** "Do not change the product, preserve it with 100% fidelity."
    *   **You say:** "The original product is featured with photorealistic clarity and razor-sharp detail."
2.  **BE ELEGANTLY CONCISE:** Combine related concepts into a single, flowing sentence.
    *   **Instead of:** "Maintain the aspect ratio. Preserve the background. The product is the focus."
    *   **You say:** "The composition honors the original photo's aspect ratio and scene, placing the hyper-realistic product as the clear focal point against the preserved background."
3.  **TRUST THE IMAGE MODEL:** The final description is for a powerful image AI. You don't need to remind it of basic constraints like "don't put text in the corner" or "use the exact words". Your job is to simply describe where the text *is* and what it *looks like*.

**Your Inputs:**
1.  **Product Image:** [PRODUCT IMAGE]
2.  **Promotional Text:** [PROMOTIONAL TEXT]

**Your Creative Task:**
**Step 1: Internal Analysis (Silent Process)**
- Analyze the [PRODUCT IMAGE] against the **Core Principles** and the **Conditional Rules** below. Make all your design decisions based on this analysis.
**Step 2: Formulate the Creative Description (Your Output)**
- Based on your decisions from Step 1, generate the poster description following the **Output Philosophy** above. Your language must be creative, descriptive, and efficient.
**Conditional Rules (For your analysis in Step 1):**
- **Background:** If the image has a person in a real scene, the original background is locked.
- **Composition:** If the main subject is cut off at the edge, that composition is locked.
- **Product Type:** If it's unworn clothing on a plain background, generate a complementary, high-quality surface background (e.g., textured fabric, polished wood).
**CRITICAL CONSTRAINTS (Final Check for your Internal Logic):**
1.  **Product Integrity:** Is the product's preservation the core of the description?
2.  **Text Position:** Is the text location described clearly and logically? Avoid the bottom-left corner.
3.  **Text Content:** The promotional text must be used exactly as given.
4.  **Scene Preservation:** Have you correctly applied the background/composition lock rules?
5.  **Aspect Ratio:** Is the preservation of the aspect ratio implied in your composition description?

**Output Generation:**
Provide your response exclusively in the following strict JSON format. The description inside the JSON should be a masterpiece of concise, creative direction.

**Provide your response exclusively in the following strict JSON format:**
{{
    "Poster_Copy_Headline": "A concise, descriptive string detailing the text's appearance and placement.",
    "Poster_Background": "A concise, descriptive string detailing the overall scene, the product's appearance, and the background treatment."
}}

---
### **Example of the Desired Output Style:**

*This is how you should format your output for the sunscreen example:*

```json
{
    "Poster_Copy_Headline": "The text 'SPF 50 PA+++ Hybrid' is elegantly rendered in a Luxury Minimal font with a deep charcoal color, positioned in the open space of the upper-right quadrant. Its placement ensures high contrast and legibility without touching any product elements.",
    "Poster_Background": "This composition maintains the original photograph's aspect ratio and clean studio setting. The two pink sunscreen tubes are the hyper-realistic, sharp focal point. The existing light background is subtly enhanced with a gentle, warm-to-cool gradient to add a touch of depth, ensuring the product remains the hero of the image."
}
Below is the PROMOTIONAL TEXT you are required to use: """


def send_request(item):
    prompt = addon_prompt + "\"" + item["text"] + "\""
    result_prompt_json = gpt_client.make_image_json_request(
        "", prompt, [], [item["source_url"]], max_tokens=8000, timeout=60)
    prompt = result_prompt_json["Poster_Copy_Headline"] + result_prompt_json["Poster_Background"]

    res = gemini_25_flash_image_gen(
        prompt,
        item["source_url"],
        model_name="nano-banana2",
        specify_gen_ratio=False,
        ratio="1:1",
        ak="077P6WUvYtpKM6biqn0PW1tQ4iZuJbrL",
    )
    gen_url = save_tos(encode_pil_bytes(res["image"], False))
    item["gpt_prompt"] = prompt
    item["gemini_url"] = gen_url

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
                    res = future.result()
                    pbar.update(1)  # Update progress bar

                    results.append(res)
                    if len(results) % 10 == 0:
                        json_save(results, dst)
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    error_results.append({'image_url': res, 'error_reason': str(e)})

    json_save(results, dst)
    print(len(error_results))


if __name__ == "__main__":
    args = parse_args()
    print(f"input_file: {args.input_file}, num_workers: {args.num_workers}")

    data_list = load_file(args.input_file)

    main(data_list, args.output_file, args.num_workers)

    print('Done!')
