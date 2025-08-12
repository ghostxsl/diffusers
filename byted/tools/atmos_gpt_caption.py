# Copyright (c) wilson.xu. All rights reserved.
import argparse
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.byted.clients.azure_mllm import MLLMClient
from diffusers.data.utils import load_file, json_save


gpt_client = MLLMClient(model_name="gpt-5-mini-2025-08-07")


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument("--input_file", default=None, type=str)
    parser.add_argument("--save_file", default="output.json", type=str)
    parser.add_argument("--max_workers", default=10, type=int)

    args = parser.parse_args()
    return args


PROMPT = """### ROLE ###
You are an expert Reverse Prompt Engineer, acting as a creative director. Your task is to deconstruct a product poster into its core visual elements. Your language must be concise, direct, and evocative, combining the fluency of natural language with the efficiency of an AI prompt.
### TASK ###
Analyze the provided product poster and generate a single, valid JSON object. Your descriptions should be direct and impactful, avoiding conversational filler or redundant words. Focus on capturing the essential visual information needed to recreate the image. Do not output any text or explanation outside of this JSON object.
### OUTPUT FORMAT & CONSTRAINTS ###
Your output MUST be a single, valid JSON object with four keys: "style_description", "product_description", "text_layout_description", and "background_description".

Here are the detailed content requirements for each key:
1.  **"style_description"**: (Type: String - Concise Natural Language)
    *   **Objective**: Write a concise, powerful description of the poster's overall visual identity. Use short, active sentences that combine key descriptive phrases.
    *   **Instruction**: Immediately state the core aesthetic and mood. Directly describe the lighting, color palette, and composition style.
    *   **Example**: "A modern and minimalist advertisement. The mood is calm and sophisticated. It uses a muted color palette and soft, even studio lighting. The composition is clean, balanced, and centered on the product."
2.  **"product_description"**: (Type: String - Concise Natural Language)
    *   **Objective**: Provide a direct and vivid description of the product itself.
    *   **Instruction**: Start immediately with the subject (e.g., "A photorealistic shot of..."). Use short, active sentences to describe its form, material, and presentation. Omit any introductory phrases like "The image features...".
    *   **Critical Constraint**: You MUST ignore any text, logos, or labels printed on the product's own surface or packaging. Focus only on its physical form.
    *   **Example**: "A photorealistic shot of a sleek wireless headphone. It has a matte blue finish with polished metallic accents. The headphone floats in mid-air, angled perfectly to showcase its modern design."
3.  **"text_layout_description"**: (Type: String - Structured Description)
    *   **Objective**: This field is an exception and remains structured for clarity. It provides clear, unambiguous instructions for text overlays.
    *   **Constraint**: Analyze only promotional text separate from the product. If none exists, this field MUST be an empty string (`""`).
    *   **Structure**: For each text block, write one simple sentence detailing its: **Position**, **Font Style** (`serif`, `sans-serif`, `script`, `bold`, etc.), **Font Size** (`small`, `medium`, `large`), **Font Color**, and **Text Content** (in double quotes).
    *   **Example**: `The headline "A New Dawn" is at the top-center in a large, white, elegant serif font. The tagline "Experience the light" is at the bottom-center in a small, white, clean sans-serif font.`
4.  **"background_description"**: (Type: String - Concise Natural Language)
    *   **Objective**: Write a concise, direct description of the background.
    *   **Instruction**: Immediately describe the setting and its main features. Use short, descriptive sentences to detail textures, secondary elements, and the color scheme.
    *   **Example**: "The background is a solid, dark charcoal grey with a smooth matte surface. Subtle, glowing blue circuit lines add a futuristic feel. A shallow depth of field keeps the background slightly blurred and the product in sharp focus."

Output example:
```json
{
  "style_description": "A commercial poster with a natural and organic aesthetic. The mood is fresh and serene. It uses an earthy color palette of greens and browns with soft, natural sunlight. The composition is minimalist and centered.",
  "product_description": "A photorealistic shot of a cosmetic bottle. The bottle is made of translucent green plastic and has a black pump dispenser. It stands upright on a piece of rustic wood.",
  "text_layout_description": "The headline \"NATURE'S SECRET\" is at the top-center in a large, white, vintage serif font. The supporting line \"100% Organic Ingredients\" is at the bottom-center in a small, white, sans-serif font.",
  "background_description": "The background shows lush green foliage, soft and out-of-focus. This creates a beautiful bokeh effect and adds a sense of depth. The setting feels like a serene, sunlit forest."
}
```"""


def send_request(item):
    result_json = gpt_client.make_image_json_request(
        "", PROMPT, [item['gen_image']], max_tokens=4000, timeout=60)
    return result_json


def main(args):
    data = load_file(args.input_file)

    results = []
    error_results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_url = {executor.submit(send_request, line): line for line in data}
        with tqdm(total=len(data)) as pbar:
            for future in concurrent.futures.as_completed(future_to_url):
                line = future_to_url[future]
                try:
                    res_json = future.result()
                    pbar.update(1)  # Update progress bar

                    line['gpt_res'] = res_json
                    line['gpt_prompt'] = res_json['style_description'] + res_json['product_description'] +\
                                         res_json['text_layout_description'] + res_json['background_description']
                    results.append(line)
                    if len(results) % 10 == 0:
                        json_save(results, args.save_file)
                except Exception as e:
                    print(f"An error occurred for {line['gen_image']}: {e}")
                    error_results.append({"image_url": line['gen_image'], "error_reason": str(e)})

    json_save(results, args.save_file)
    print(error_results)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
