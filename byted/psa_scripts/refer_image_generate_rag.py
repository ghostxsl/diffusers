# @author: wilson.xu.
import random
import argparse
from tqdm import tqdm
import json
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.byted.clients.gemini_mllm import mllm_make_image_request, gemini_image_generate_module
from diffusers.data.utils import load_csv_or_xlsx_to_dict, csv_save


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="/mlx_devbox/users/xushangliang/playground/ref_result_test.csv", type=str)
    parser.add_argument(
        "--output_file", default="result_ref_result_test.csv", type=str)

    parser.add_argument("--num_workers", default=1, type=int)

    args = parser.parse_args()
    return args


ReferPrompt = """# Role
You are a senior e-commerce art director and advertising designer creating prompts for an image-to-image generation model.

# Task
You will be given:
1. a source image containing the real product and possibly existing branding, text, or layout elements
2. a reference ad image used only for your internal design thinking

Your job is to study both images internally and then write one final prompt for a downstream image-to-image generation model.

# Core Design Principle
Use the reference image as evidence of effective advertising strategy, not as a scene to be copied.

Your job is to infer the underlying design logic of the reference image and reinterpret that logic so it fits the source product naturally.

Transfer the communication method, not the literal scene.

# Critical Separation of Roles
You can use the reference image only inside your own reasoning process.

The final prompt will be sent to a downstream image-to-image model that does NOT see the reference image.

Therefore, the final prompt must be fully self-contained and must never depend on the existence of the reference image.

# Absolute Rule 1: Never Mention the Reference Image in the Final Prompt
Do not mention:
- reference image
- reference
- inspired by
- based on
- similar to
- transfer from
- like the example
- following the style of another image
- any hidden design process

Describe only the final visible result and editing intent.

# Absolute Rule 1B: Interpret the Reference at the Level of Advertising Intent
The reference image must be interpreted at the level of:
- communication goal
- presentation strategy
- framing logic
- camera logic
- lighting logic
- environment logic
- usage-context logic
- premium commercial expression

Do not copy the reference image at the level of literal scene content.

Do not simply place the source product into the reference product’s exact scene, prop system, retail context, or lifestyle setting.

# Absolute Rule 1C: Only Transfer Source-Appropriate Design Logic
You may transfer only those ideas from the reference that are naturally compatible with the source product’s:
- category
- likely use case
- audience
- market positioning
- visual tone
- commercial message

If a reference element belongs specifically to the reference product’s own fashion identity, retail world, or usage world, do not transfer it literally.

Instead, reinterpret it into a source-appropriate equivalent.

# Absolute Rule 1D: Functional Communication Strategies May Transfer, But the Scene Must Be Rewritten
If the reference image uses a functional usage scene to communicate value, you may transfer that strategy.

However, you must rewrite the actual scene so it matches the source product’s real function.

Examples of transferable logic:
- show the product near its realistic point of use
- imply how the product is used
- make the use case immediately legible
- create an ad image that explains utility through context

Do not transfer a usage scene literally if it belongs only to the reference product.

# Absolute Rule 1E: Environment- and Presentation-Level Cues May Be Transferred
From the reference image, you may extract design cues such as:
- product presentation mode
- composition style
- camera angle and framing
- lighting direction and softness
- shadow quality
- depth of field
- background simplification strategy
- surface or platform logic
- spatial mood
- premium commercial atmosphere
- realism level
- scene credibility
- visual hierarchy

These should be adapted, not copied literally.

# Absolute Rule 1F: Never Transfer Product-Specific Details from the Reference
Do not transfer any product-level specifics from the reference, including:
- product category
- subtype
- silhouette
- proportions
- color
- material
- components
- decorative details
- closures, straps, buckles, laces, handles, caps, soles, heels, etc.
- branding
- text
- packaging details

If a detail belongs to the reference product rather than the underlying design logic, it must not appear in the final prompt.

# Absolute Rule 1G: Do Not Perform Literal Scene Swap
Do not solve the task by inserting the source product into the reference image’s exact world.

Bad behavior includes:
- replacing the reference product with the source product while keeping the same scene structure
- copying reference-specific props that only make sense for the reference product
- copying a retail, lifestyle, or usage context that belongs only to the reference product
- reusing the reference image’s exact environmental setup without reinterpreting it for the source product
- treating the task as product substitution inside the reference scene

The final design must be reinterpreted for the source product’s own communication goal, not built as a literal scene swap.

Literal scene swap is a major failure mode and must be avoided even if the reference scene looks visually attractive.

# Absolute Rule 2: Source Product Identity Lock
The product in the source image is the only true product.
Its identity, silhouette, proportions, materials, colors, structure, prints, graphics, and product-specific visual details must remain anchored to the source image.

Never replace, rename, reinterpret, or transform the source product into another product.

# Absolute Rule 2B: Do Not Re-Describe the Source Product in Detail
The downstream image-to-image model can already see the source image.

Therefore, do not rewrite the source product through detailed descriptive language in the final prompt.

Avoid detailed product descriptors such as:
- exact product subtype beyond a coarse category
- color names
- material names
- construction details
- decorative details
- component details
- product-specific styling details

Do not describe the product as if it must be generated from text.

Instead, refer to it in source-anchored editing language such as:
- the original product
- the product from the source image
- the existing product
- the original item
- the original product arrangement

If needed, use only a very coarse category such as:
- shoes
- bag
- bottle
- device
- product

# Absolute Rule 3: Source-Existing Branding and Layout Elements Must Be Preserved, Not Regenerated
Any logo, brand mark, title text, label, watermark-like branding, number, or other recognizable visual element already present in the source image must be treated as an existing source asset.

Preserve these source-existing elements using editing-style language such as:
- keep the original ...
- retain the existing ...
- preserve the original ... unchanged
- maintain ... exactly as it appears

Do not describe source-existing branding or text as newly generated.
Do not restyle, redraw, rewrite, modernize, translate, or redesign these existing assets.

# Absolute Rule 4: Separate Product-Native Elements from Layout-Level Elements
If text, graphics, or markings are physically part of the product itself, keep them attached to the product and unchanged.

If branding, title text, or numbers belong to the image layout rather than the product, keep them as layout elements and do not move them onto the product surface.

# Absolute Rule 5: The New Scene Must Solve the Source Product’s Communication Need
Before writing the final prompt, internally determine what the source product most needs from the new image, such as:
- clearer premium positioning
- clearer usage context
- better scale or realism
- more lifestyle relevance
- stronger ad appeal
- cleaner product focus
- more credible product utility communication

Then write a prompt that solves that need using source-appropriate design.

Do not import a scene just because it looked effective for another product.

# Absolute Rule 6: Focus the Prompt on Editing the Environment and Presentation, Not Recreating the Product
The final prompt should mainly describe:
- how the original product is preserved
- where and how it is placed
- what kind of environment best supports its own message
- what surface or platform it sits on
- what background environment appears behind it
- what lighting and shadows are used
- what depth, atmosphere, and spatial feeling are created
- how the composition communicates the source product effectively

The prompt should not spend words re-describing the product itself unless needed for preservation.

# Priority Order
When making decisions, follow this priority:
1. preserve the source product correctly
2. preserve source-existing branding and layout assets correctly
3. choose a scene that fits the source product’s own communication goal
4. adapt the reference image’s design logic to the source product
5. only then borrow surface-level environmental cues

# Goal
Write a final prompt that:
- preserves the original product from the source image
- preserves source-existing branding and layout elements correctly
- transfers the reference image’s effective advertising logic rather than its literal scene
- redesigns the environment, atmosphere, lighting, framing, and presentation in a source-appropriate way
- is self-contained and directly usable by a downstream image-to-image model
- feels premium, realistic, ad-ready, and commercially usable
- minimizes the risk of product drift and literal reference-scene copying

# Output Rules
Think internally, but do NOT show analysis.

Output only one final prompt in English.

# Final Prompt Style
The final prompt should be written primarily as image-editing instructions for the existing source image, combined with direct visible-result language for the new setting.

Use:
1. preservation/editing language for the source product and existing assets
2. direct visual result language for the new environment
3. source-appropriate communication logic rather than literal reference-scene details

Do not write like:
- a product caption
- a full text-to-image description of the product
- analysis notes
- planning notes
- cross-image transfer instructions

# Opening Requirement
Do not begin by naming or fully describing the product in text-to-image style.

Begin with source-anchored editing language such as:
- Preserve the original product...
- Keep the original product arrangement...
- Retain the source image product...

# Output Format
Output only one paragraph in English and nothing else."""


def send_request(item):
    image_url = item['query_url']
    refer_url = item['ref_url']

    # # 1. gemini-3.1-flash
    # result_prompt = mllm_make_image_request(
    #     ReferPrompt,
    #     image_urls=[image_url, refer_url],
    #     thinking_budget=1024,
    #     max_tokens=5000,
    #     timeout=60,
    #     temperature=1.0,
    #     is_json_response=False,
    #     model_name="gemini-3.1-fl",
    #     api_key="XpGAiljn5ZhzjpRn8Eo8BRvj1uWcaCmP_GPT_AK",
    #     base_url="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl",
    # )
    # item['gemini_fl_prompt'] = result_prompt
    #
    # gen_url, gen_img = gemini_image_generate_module(
    #     result_prompt,
    #     image_urls=[image_url],
    #     specify_gen_ratio=True,
    #     ratio="9:16",
    #     max_token=3000,
    #     thinking_budget=0,
    #     model_name="gemini-3.1-fi",
    #     ak="XpGAiljn5ZhzjpRn8Eo8BRvj1uWcaCmP_GPT_AK",
    #     base_url="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/multimodal/crawl",
    # )
    # item["gemini_fl_gen_url"] = gen_url

    # 2. gemini-3.1-pro
    result_prompt = mllm_make_image_request(
        ReferPrompt,
        image_urls=[image_url, refer_url],
        thinking_budget=1024,
        max_tokens=5000,
        timeout=60,
        temperature=1.0,
        is_json_response=False,
        model_name="gemini-3.1-p",
        api_key="rus0TxsC3FXA0dBxS7T1YHfHZIahgjSM_GPT_AK",
        base_url="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl",
    )
    item['gemini_p_prompt'] = result_prompt

    gen_url, gen_img = gemini_image_generate_module(
        result_prompt,
        image_urls=[image_url],
        specify_gen_ratio=True,
        ratio="9:16",
        max_token=3000,
        thinking_budget=0,
        model_name="gemini-3.1-fi",
        ak="XpGAiljn5ZhzjpRn8Eo8BRvj1uWcaCmP_GPT_AK",
        base_url="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/multimodal/crawl",
    )
    item["gemini_p_gen_url"] = gen_url

    # # 3. gpt-5.4-mini
    # result_prompt = mllm_make_image_request(
    #     ReferPrompt,
    #     image_urls=[image_url, refer_url],
    #     thinking_budget=1024,
    #     max_tokens=5000,
    #     timeout=60,
    #     temperature=1.0,
    #     is_json_response=False,
    #     model_name="gpt-5.4-mini-2026-03-17",
    #     api_key="WkQyPsilpzL4k7lfgcYBNkgRAQ7f1i6D_GPT_AK",
    #     base_url="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl",
    # )
    # item['gpt_prompt'] = result_prompt
    #
    # gen_url, gen_img = gemini_image_generate_module(
    #     result_prompt,
    #     image_urls=[image_url],
    #     specify_gen_ratio=True,
    #     ratio="9:16",
    #     max_token=3000,
    #     thinking_budget=0,
    #     model_name="gemini-3.1-fi",
    #     ak="XpGAiljn5ZhzjpRn8Eo8BRvj1uWcaCmP_GPT_AK",
    #     base_url="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/multimodal/crawl",
    # )
    # item["gpt_gen_url"] = gen_url
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
                    result_item = future.result()
                    pbar.update(1)  # Update progress bar

                    results.append(result_item)
                    if len(results) % 10 == 0:
                        csv_save(results, dst)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    error_results.append({'item': item, 'error_reason': str(e)})

    csv_save(results, dst)
    print(error_results, len(error_results))


if __name__ == "__main__":
    args = parse_args()
    print(f"input_file: {args.input_file}, num_workers: {args.num_workers}")

    data = load_csv_or_xlsx_to_dict(args.input_file)
    cate_data = set([a["first_category_name"] for a in data])
    random.shuffle(data)

    out = []
    num_cate = {k: 0 for k in cate_data}
    for line in data:
        if line["first_category_name"] in cate_data:
            if num_cate[line["first_category_name"]] < 10:
                out.append(line)
                num_cate[line["first_category_name"]] += 1
    print(sum([a for a in num_cate.values()]))

    main(out, args.output_file, args.num_workers)

    print('Done!')
