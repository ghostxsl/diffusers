# @author: wilson.xu.
import argparse
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.byted.clients.azure_mllm import MLLMClient, seedream_5_generate_image
from diffusers.data.outer_vos_tools import encode_pil_bytes, load_or_download_image
from diffusers.data.byted.tos import save_tos, _gen_name
from diffusers.data.utils import load_file, json_save


gpt_client = MLLMClient(model_name="gpt-5-mini-2025-08-07", api_key="7e5RK9vuv5NTXU07CosK9uLotGpltSpD_GPT_AK")


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="/mlx_devbox/users/xushangliang/playground/label_1014_psa_data_1k_test_urls.json", type=str)
    parser.add_argument(
        "--output_file", default="gemini_1014_psa_data_1k_test_urls.json", type=str)
    parser.add_argument(
        "--num_workers", default=10, type=int)

    args = parser.parse_args()
    return args


# 场景联想
image_poster_prompt_v3_6 = """You are a world-class AI Creative Director, Graphic Designer, and Prompt Engineering Strategist. Your mission is to analyze a provided e-commerce poster and, from it, generate a single, powerful, and direct prompt for a secondary AI image generation model (like Gemini 2.5 Flash). Your output must be a crystal-clear, descriptive "order" of the final image.
**Prime Directives: Fidelity, Logic, Creativity, and Safety.**
Your absolute highest priority is to balance asset integrity with creative storytelling while strictly adhering to safety constraints.
1.  **Product Integrity:** The product's original appearance must be perfectly preserved without color shift or distortion. **Crucially, do NOT alter the product's physical mechanism (e.g., do not change a screw cap to a pump; do not change a box shape).**
2.  **Logo Integrity:** Any standalone brand logo must be perfectly preserved and placed without alteration or regeneration.
3.  **Lighting Control:** Lighting must be soft, clean, and neutral to ensure accurate asset color representation. AVOID heavy or colored shadows.
4.  **Conditional Text Handling:** You will ONLY extract and use text if it is clear "Marketing Copy" from the poster background. If no such text exists, the final image must be text-free. If used, text must be a verbatim copy in its original language. DO NOT TRANSLATE.
5.  **Strictly NO Human Faces (Zero Tolerance Policy):**
    *   **CRITICAL SAFETY BAN:** You are explicitly **FORBIDDEN** from generating or describing human faces. This applies universally, but you must be **hyper-vigilant regarding MINORS/CHILDREN**. Generating a child's face is a critical safety violation.
6.  **Creative Enhancement:** You are not just a renderer; you are a storyteller. You are explicitly empowered to enrich the scene by:
    *   **A) Placing the product in a relevant and imaginative usage scenario.**
    *   **B) Adding conceptually related elements that enhance the product's story.** This is crucial for bringing the scene to life.
    *   *Example:* If the product is a high-quality cat food can, you should not just place it on a pedestal. Instead, place it on a clean, modern kitchen floor next to a beautiful, healthy cat (e.g., a British Shorthair) that is looking at the can with interest. If the product is a coffee bean bag, add elements like a steaming ceramic mug, a grinder, or scattered roasted beans.
7.  **The "Ingredient & Prop" Rule (Anti-Hallucination Mandate):**
    *   **NO "Unboxing" or "Replication":** If the input image shows a sealed package (box, bottle, bag), you are **FORBIDDEN** from generating the "content" inside it (e.g., do NOT generate a naked bar of soap next to a soap box; do NOT generate a brewed cup of tea next to a tea box) unless it is explicitly visible in the original image. We do not know what the inside looks like, so guessing creates errors.
    *   **Use "Ingredients" Instead:** When adding elements, choose **Raw Ingredients** or **Generic Props**.
        *   *Example:* For **Ginger Shampoo**, add **fresh ginger roots/leaves**, NOT a blob of shampoo liquid.
        *   *Example:* For **Milk Candy**, add **a glass of milk**, NOT an unwrapped candy.
        *   *Example:* For **Tea**, add **fresh tea leaves/mist**, NOT a cup of colored liquid.
**Your Internal Thought Process (Execute these steps silently):**
1.  **Deep Product Understanding:**
    *   Identify the Core Product and any standalone Logo.
    *   **Crucially, understand its function, target user (person, pet, etc.), and the core feeling it sells (e.g., luxury, convenience, health, joy).** This is the foundation for your creative leap.
2.  **Conceptual Brainstorming & Scene Development:**
    *   Based on your understanding, brainstorm a "micro-moment" or "visual story" for the product.
    *   **Define the Usage Scenario:** Where would this product be used or enjoyed? (e.g., a serene bathroom, a bustling kitchen, a misty forest trail).
    *   **Define Supporting Elements (The "Ingredient Check"):** Select 1-2 powerful elements.
        *   *Self-Correction Check:* Am I trying to generate the product itself (e.g., the soap bar, the liquid)? -> **STOP.**
        *   *Correction:* Switch to an **Ingredient** (e.g., flowers, water, raw food items) or a **Generic Prop** (e.g., a towel, a wooden board).
    *   Select the most compelling and aesthetically pleasing concept.
3.  **Text & Asset Decision:**
    *   Based on Prime Directive #4, decide if text will be used. If so, perform a verbatim transcription.
4.  **Final Prompt Construction:**
    *   Synthesize your creative scene, supporting elements, and text decision into a single, rich, descriptive prompt, rigorously enforcing the "No Face", "Smart Interaction", and "No Unboxing" rules.
**--- YOUR TASK ---**
**Analyze the provided image and use the following parameters to generate the final prompt:**
*   **ASPECT RATIO:** {generate_ratio}
**--- YOUR FINAL OUTPUT (Structure Template) ---**
Your entire response MUST be ONLY the text for the final prompt.
"Photorealistic {generate_ratio} commercial advertisement, professional studio quality, ultra-detailed. Featuring the **product from the reference image and its original brand logo, both perfectly preserved without alteration**, as core assets. The main product is the central hero object, placed within [**Detailed, imaginative description of the usage scenario and its mood, incorporating the supporting elements you conceived: EITHER a hand-held POV shot (only if applicable) OR a premium still-life setup on a surface. Incorporate supporting elements**]. **The composition strictly excludes any human faces.** The overall atmosphere is [**3-4 keywords describing the mood**]. The lighting is **soft, clean, and bright, ensuring accurate product and logo color representation**. The final image is professional, high-resolution, and free of artifacts."
**[Conditional Text Add-on - Include ONLY if applicable]**
" The text '[**Verbatim transcription of Marketing Copy 1, if found**]' and '[**Verbatim transcription of Marketing Copy 2, if found**]' is elegantly integrated, rendered in [**Concise, descriptive keywords for your designed text style**]."
**Do not add any explanations, greetings, or conversational text. Your only output is the final, assembled prompt.**"""


# 虚化背景
poster_blurred_background_prompt = """You are an expert **Advertising Creative Director** and a hyper-logical **AI Prompt Engineer**. Your mission is to analyze a provided e-commerce poster and generate a single, highly-specific, and **CONCISE, KEYWORD-DRIVEN prompt** for a secondary AI image generation model (like Gemini).
**Prime Directives: Absolute Fidelity & Intelligent Staging.**
These rules are non-negotiable.
1.  **Absolute Product & Logo Integrity (HIGHEST PRIORITY):** The product(s) and any logo(s) are SACRED assets. They must be extracted from the reference image and used **AS-IS**.
2.  **Lighting & Realism Control (ULTIMATE CRITICAL RULE):** Your goal is **Stylized Photorealism**, not a physics simulation. You must create an ultra-clean, commercial lighting environment that makes the product "pop" as if it is a perfect graphic element.
    *   **Lighting Principle:** The lighting must be perfectly **even, soft, and completely diffused**, as if the product is inside a giant softbox.
    *   **Forbidden Physical Effects (NON-NEGOTIABLE):** You are **STRICTLY FORBIDDEN** from generating any realistic light physics that distract from the product. This is an absolute rule and includes:
        *   **Any visible cast shadows** from the product onto the surface.
        *   **Any mirror-like reflections** of the product or environment on surfaces.
        *   **Any strong or distracting specular highlights or caustics** (the bright "glints" on shiny surfaces).
    *   **The Goal:** The final image must have the clean, pristine feel of a high-end magazine ad, where the product is the absolute, unblemished hero, free from the "visual noise" of realistic lighting.
3.  **Mandatory Background Reinvention (CRITICAL CREATIVE RULE):** You are **FORBIDDEN** from replicating the source image's background. You must **invent a completely new background**.
4.  **Subtle Thematic Enhancement (AESTHETIC RULE):**
    *   This rule applies **ONLY** to realistic scenes (**Path A: Top-Down** and **Path B: Perspective**).
    *   If the product is **Beauty/Skincare, Food/Beverage, Jewelry, or Pet Supplies**, you may add **1-2 small, subtle, thematic props**.
5.  **Staging Strategy: The Core Decision Based on Physical & Aesthetic Plausibility:**
    Your first and most important task is to perform a **"Real-World Plausibility Analysis"** on the source image's product arrangement. Ask yourself: "Can this exact arrangement be recreated on a realistic physical surface in a way that is both **physically stable AND aesthetically pleasing/professional**?"
    *   **IF THE ANSWER IS "NO" (Implausible or Awkward):**
        *   **Trigger Condition:** You MUST use this path if the arrangement is physically impossible or simply looks awkward and unprofessional in a realistic setting.
        *   **Use Your Judgment and These Examples:**
            *   **Physically Impossible:** A multi-row grid of heavy objects without visible support (e.g., a 2x5 grid of laundry detergent packs).
            *   **Aesthetically Awkward (even if physically possible):**
                *   Multiple thin, flat items that are difficult to stage elegantly (e.g., several screen protectors laid side-by-side on a table).
                *   Naturally formless or flexible items that would look cluttered (e.g., charging cables, ribbons).
                *   Large collections of small, scattered parts (e.g., multiple groups of hardware components).
        *   **Action (Mandatory):** Choose the **Studio Backplate Strategy (Path S)**.
            *   **S1 (For Electronics):** Design a **graphical, abstract studio backdrop** (e.g., "glowing abstract technology lines," "clean minimalist geometric shapes").
            *   **S2 (For all others):** Design a **photorealistic, tangible material backplate** (e.g., "flowing silk fabric," "a sheet of brushed dark aluminum").
            *   **Universal Rule:** No separate physical stands or podiums are allowed in this path.
    *   **IF THE ANSWER IS "YES" (Plausible and Elegant):**
        *   **Trigger Condition:** This applies if the product(s) can be placed realistically and elegantly on a surface (e.g., a single upright bottle, or two self-standing products in a simple side-by-side row).
        *   **Action:** Proceed to choose between the two realistic staging options:
            *   **Path A: Top-Down / Flat Lay Staging:** Best for **single, small items that can lie flat** (e.g., mobile phones, wallets, open makeup palettes).
            *   **Path B: Physics-Aware Perspective Staging:** For all other plausible scenarios.
                *   **B0: Handheld?** Stage in a *new* hand.
                *   **B3: Hangable?** Use a **Complete Suspension System** (e.g., hanger on a rod).
                *   **B4: Large/Floor-standing?** Place on an appropriate *floor*.
                *   **B1: Self-Standing?** Place on a tabletop surface. No extra props.
                *   **B2: Leaning?** Invent an **aesthetically pleasing support object** (e.g., minimalist block, stack of books).
        *   **Background (Path B Only):** The background must be a **heavily blurred (bokeh) realistic scene**.
6.  **Text as an Immutable Asset & Descriptive Language Translation:**
    *   The final prompt must include the `Text & Graphics Asset` line. The `Scene Description` must be a purely visual description, free of internal jargon.
**--- YOUR INTERNAL THOUGHT PROCESS (Execute these steps silently) ---**
1.  Analyze Product(s), layout, and form.
2.  Read text for context.
3.  **Apply the ultimate Lighting & Realism Control principle (no shadows, no reflections) throughout your entire design process.**
4.  Perform the "Real-World Plausibility Analysis".
5.  Decide your primary path: Studio Backplate (S) or Realistic Staging (A/B).
6.  Execute the sub-logic for the chosen path.
7.  Translate your final design into a purely descriptive sentence.
8.  Synthesize the final prompt, ensuring the new, absolute lighting style is explicitly included.
**--- YOUR TASK ---**
**Analyze the provided image and use the following parameters to generate the final prompt:**
*   **ASPECT RATIO:** {generate_ratio}
**--- YOUR FINAL OUTPUT (Structure Template) ---**
Your entire response MUST be ONLY the text for the final prompt, following this exact concise format.
A {generate_ratio} photorealistic commercial photo, [Your chosen shot type, e.g., "Medium close-up shot", "Top-down shot"]. All products and logos from reference image, used as-is, unchanged in appearance or angle. Scene Description: [Translate your strategic choice into a purely visual description here.] Style: [3-4 keywords for mood, e.g., "Minimalist, elegant, premium, futuristic"], clean commercial studio lighting, perfectly even and diffused, shadowless, no cast shadows, no reflections. Text & Graphics Asset: Retain all original text from the reference image, unchanged.
"""


def send_request(item):
    image_url = item[0]
    gen_ratio = "3:4"

    # # 虚化背景
    # meta_prompt_blur_bg = poster_blurred_background_prompt.format(generate_ratio=gen_ratio)
    # prompt_40 = gpt_client.make_image_request("", meta_prompt_blur_bg, [], [image_url], max_tokens=5000, timeout=60)
    # prompt_40 = "Redesign and generate a new poster with reference to the input poster image: " + prompt_40
    # res_url = seedream_5_generate_image(
    #     prompt_40,
    #     image_urls=image_url,
    #     specify_gen_ratio=True,
    #     ratio=gen_ratio,
    #     image_size="2K",
    # )
    # img = load_or_download_image(res_url)
    # img = img.resize((img.width // 2, img.height // 2), 1)
    # gen_url1 = save_tos(encode_pil_bytes(img, False), _gen_name(""))

    # 场景联想(Meta prompt)
    meta_prompt_scene = image_poster_prompt_v3_6.format(generate_ratio=gen_ratio)
    prompt_91 = gpt_client.make_image_request("", meta_prompt_scene, [], [image_url], max_tokens=5000, timeout=60)
    prompt_91 = "Redesign and generate a new poster with reference to the input poster image: " + prompt_91
    res_url2 = seedream_5_generate_image(
        prompt_91,
        image_urls=image_url,
        specify_gen_ratio=True,
        ratio=gen_ratio,
        image_size="2K",
    )
    img2 = load_or_download_image(res_url2)
    img2 = img2.resize((img2.width // 2, img2.height // 2), 1)
    gen_url2 = save_tos(encode_pil_bytes(img2, False), _gen_name(""), headers={"Content-Type": "image/jpeg"})

    item.append([gen_url2, ])
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
                    print(f"An error occurred: {str(e)}")
                    error_results.append({'image_url': item, 'error_reason': str(e)})

    json_save(results, dst)
    print(error_results)
    print(len(error_results))


if __name__ == "__main__":
    args = parse_args()
    print(f"input_file: {args.input_file}, num_workers: {args.num_workers}")

    data = load_file(args.input_file)
    out = []
    for item in data:
        label_dict = item[1]
        if label_dict["human_parts_visible"] not in  ["none", "hand_only", "foot"]:
            continue
        elif label_dict["grid_style_image_collage"]:
            continue
        elif label_dict["is_poster"]:
            if label_dict["subject_classification"] in ["Beauty & Health", "Food & Beverages", "Jewelry Accessories & Derivatives", "Pet Supplies"]:
                out.append(item)
        elif label_dict["is_clothing_product"]:
            continue
        elif label_dict["background_classification"] == "Real Background":
            continue
        elif not label_dict["can_stand_upright"]:
            continue

    main(out, args.output_file, args.num_workers)

    print('Done!')
