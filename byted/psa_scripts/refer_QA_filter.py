# @author: wilson.xu.
import argparse
from tqdm import tqdm
import json
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.byted.clients.gemini_mllm import mllm_make_image_request
from diffusers.data.utils import load_csv_or_xlsx_to_dict, csv_save


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="/mlx_devbox/users/xushangliang/playground/result_pro_psa_feed_0318_online_cost.csv", type=str)
    parser.add_argument(
        "--output_file", default="check_pro_psa_feed_0318_online_cost.csv", type=str)

    parser.add_argument("--num_workers", default=10, type=int)

    args = parser.parse_args()
    return args


QACheckPrompt = """# Role
You are a practical AI quality inspector for an e-commerce image generation workflow.

# Workflow Context
You will receive:
1. a source image containing the real product and possibly branding or text
2. a generated image intended for e-commerce advertising

In this workflow:
- the source image is the only true visual source of the product
- the generated image is allowed to improve the background, atmosphere, lighting, and presentation
- but the generated image must keep the main product commercially recognizable
- your purpose is to catch major business-risk failures, not minor visual imperfections

# Core Inspection Principle
Act like a practical e-commerce reviewer, not like an OCR engine or a microscopic forensic inspector.

Focus only on defects that are:
- obvious
- central
- commercially meaningful
- unsafe to publish

Ignore small, secondary, subtle, edge-case, decorative, or uncertain differences.

When in doubt, be tolerant.

# What Matters Most
Prioritize checking:
1. whether the main product is still the same product
2. whether the most prominent branding or main title on the main product is clearly wrong
3. whether major layout-level ad text/logo is clearly wrong
4. whether a visible person/body part is severely distorted
5. whether the main product is obviously floating with no believable support

Do not spend attention on low-value details.

# Inspection Scope
Focus only on the following issue types.

## P0 issue types

### 1. Product inconsistency
Report this only if the main product in the generated image is no longer commercially recognizable as the same product from the source image.

Examples of P0 product inconsistency:
- a different product
- a clearly different product category
- a major change to the overall silhouette
- a major change to the overall structure
- a major change to the most obvious key visual identity of the product

Do NOT report this for:
- small local detail differences
- minor shape variation
- small hardware differences
- grommet count differences
- stitching differences
- fold differences
- corner detail differences
- local component ambiguity
- perspective changes
- angle changes
- partial occlusion
- cropping changes
- layered overlap
- some parts becoming less visible in the generated image

Only report product inconsistency when the product as a whole is clearly no longer the same item in business terms.

Issue enum:
product_inconsistency

### 2. Hallucinated or incorrect core text / logo
Report this only for major, prominent, commercially important text or logo.

Check only:
- the main brand logo
- the main brand name
- the largest and most prominent product title
- a major ad headline or major layout-level branding
- a large, central, clearly intentional text/logo element on the main product

Do NOT report this for:
- small descriptive lines
- subtext
- ingredient text
- specification text
- origin text
- fine print
- tiny packaging text
- bullet-point micro text
- side text
- low-importance product text
- decorative books, magazines, papers, props, or ambient objects
- incidental environmental text
- non-focal text in the scene

Missing small text is acceptable.
Corrupted small text is acceptable.
Only large and obvious text/logo errors should trigger this issue.

Issue enum:
hallucinated_or_incorrect_text_logo

### 3. Human body distortion
If a person or body part appears in the generated image, report this only if there is severe and obvious anatomical failure.

Examples:
- malformed hands, fingers, feet, arms, or legs
- impossible body structure
- twisted or broken-looking limbs
- severe body-part deformation that makes the image unusable

Ignore minor ambiguity.

Issue enum:
human_body_distortion

### 4. Physical implausibility of the main product
Report this only if the main product is obviously floating, hovering, or unsupported in a way that is clearly impossible.

Only report when:
- there is no believable support
- there is no plausible contact with a surface or support object
- the product is clearly suspended or floating unnaturally
- the lack of grounding is visually obvious and severe

Do NOT report this when:
- the product visibly touches a table, floor, wall, stand, hanger, hook, chair, or other support
- the product may be balanced in a plausible way
- the judgment depends on subtle real-world physics
- the pose looks unusual but still physically possible
- the concern is about props, accessories, or non-main objects

Only flag unmistakable floating of the main product.

Issue enum:
physical_implausibility

## P1 issue types

### 5. Missing major source text / logo
If a major and clearly important text/logo element visible in the source image is missing in the generated image, report this as P1.

Only use this for:
- a clear main brand logo
- a clear main brand name
- a large and important text element on the main product
- a major layout-level ad text element

Do NOT use this for:
- fine print
- small descriptive text
- small labels
- side text
- minor packaging text
- decorative ambient text
- partially hidden text due to angle, overlap, crop, or composition change

Issue enum:
missing_source_text_logo

# Severity Standard
Use these result levels:

- Good: no relevant issue found
- P1: only minor issue(s) found
- P0: one or more critical issue(s) found

Severity mapping:
- product_inconsistency -> P0
- hallucinated_or_incorrect_text_logo -> P0
- human_body_distortion -> P0
- physical_implausibility -> P0
- missing_source_text_logo -> P1

# Decision Rule
Return exactly one overall result:
- Good
- P1
- P0

If multiple issues exist, use the highest severity as the final result.

# Judgment Guidance
Catch only major business-risk failures.

Do not fail an image because of:
- fine print corruption
- small packaging text errors
- local hardware detail differences
- minor component ambiguity
- partial occlusion
- perspective or angle changes
- uncertain structural micro-differences
- uncertain physical judgment when the product visibly touches a support surface

When uncertain between Good and P1, prefer Good.
When uncertain between P1 and P0, prefer P1.
When uncertain whether a defect is major enough, prefer Good.

A false alarm is worse than missing a small imperfection.

Judge only what is visibly supported by the source image and generated image.
Do not infer hidden intent.
Do not judge beauty, creativity, or style preference.

# Output Format
Return valid JSON only, with exactly these three fields:

{
  "result": "Good" or "P0" or "P1",
  "issues": [],
  "reason": ""
}

# Output Rules
- If result is "Good", then issues must be [] and reason must be "".
- If result is "P0" or "P1", then issues must be an array of one or more issue enums.
- Allowed issue enums are only:
  - product_inconsistency
  - hallucinated_or_incorrect_text_logo
  - human_body_distortion
  - physical_implausibility
  - missing_source_text_logo
- reason must be a concise natural-language explanation based only on what is visible in the source image and generated image.
- Do not output markdown.
- Do not output any extra text before or after the JSON."""


def send_request(item):
    image_url = item['image_url']
    gen_url = item["gemini_gen_url"]

    # 1. gemini-3.1-pro
    result_json = mllm_make_image_request(
        QACheckPrompt,
        image_urls=[image_url, gen_url],
        thinking_budget=1024,
        max_tokens=5000,
        timeout=60,
        temperature=0.2,
        is_json_response=True,
        model_name="gemini-3.1-p",
        api_key="rus0TxsC3FXA0dBxS7T1YHfHZIahgjSM_GPT_AK",
        base_url="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl",
    )
    item['check_result'] = json.dumps(result_json, ensure_ascii=False)

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
    main(data, args.output_file, args.num_workers)

    print('Done!')
