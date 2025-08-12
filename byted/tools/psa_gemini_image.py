# @author: wilson.xu.
import argparse
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.byted.clients.azure_mllm import MLLMClient, gemini_25_flash_image_gen
from diffusers.data.outer_vos_tools import encode_pil_bytes
from diffusers.data.byted.tos import save_tos
from diffusers.data.utils import load_file, json_save


gpt_client = MLLMClient(model_name="gpt-5-mini-2025-08-07", api_key="vg2vdRs5eeTY5oC2pABdNuLncpS3g3Bk_GPT_AK")
gemini3_client = MLLMClient(model_name="gemini-3-flash-preview", api_key="7e5RK9vuv5NTXU07CosK9uLotGpltSpD_GPT_AK")


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


abstract_prompt_v1 = """You are a Senior AI Visual Strategist specializing in Commercial Product Photography and Composition.
Your task is to generate a descriptive prompt for **Contextual Background Replacement**.
**--- THE GOAL ---**
Analyze the specific product in the input image and imagine its **ideal real-world usage environment**.
Then, generate a description of that environment as a **Heavily Blurred, High-Key Background (Creamy Bokeh)**.
The product must be "stamped" onto this background in a **Zero-Gravity, Macro Close-Up** composition.
### CRITICAL VISUAL MANDATES (NON-NEGOTIABLE)
1.  **CONTEXTUAL RELEVANCE (NO GENERIC BACKGROUNDS):**
    *   Do NOT use generic "light leaks" or "abstract sparkles" unless appropriate.
    *   You MUST infer the product's function.
    *   *Example:* If it's a **Coffee Cup** -> Background is a "heavily blurred morning kitchen or cafe."
    *   *Example:* If it's a **Running Shoe** -> Background is a "heavily blurred park path or track."
    *   *Example:* If it's **Jewelry** -> Background is a "heavily blurred luxury texture/interior."
2.  **LIGHTING CONTROL (SOFT & EVEN > BRIGHT):**
    *   **PROBLEM TO AVOID:** Do NOT use "intense sunlight," "lens flares," "starbursts," or "blinding backlight." These wash out the product.
    *   **REQUIRED:** Use **"Soft, Diffused, Natural Light."** The background must be bright and airy, but the light source must be invisible and even (cloudy day logic).
3.  **FOREGROUND DOMINANCE (MACRO SHOT):**
    *   The background exists ONLY to provide color context. It must be 100% out of focus.
    *   The product must be the massively dominant visual center.
4.  **FOREGROUND INTEGRITY:** The input product (and all its text/logos) is the immutable foreground.
### PROMPT GENERATION PROCESS (Internal Thought)
1.  **Analyze Product:** What is it? (e.g., A blue Adidas bag).
2.  **Determine Scene:** Where does it belong? (e.g., Outdoors, active lifestyle, city street).
3.  **Abstraction:** How do I describe this scene using ONLY blurred colors and shapes? (e.g., "Muted greens of trees, soft beige of pavement, daylight").
4.  **Check Contrast:** Ensure the background colors contrast well with the product (e.g., If bag is Mint Green -> Background should be warm beige/sand to make it pop, not green).
### OUTPUT RULES
Generate a **single, continuous paragraph** (no line breaks) using this specific structure:
1.  **The Header:** Start strictly with: "**A photorealistic composite featuring the original foreground subject floating directly over a contextually relevant, highly blurred background...**"
2.  **The Contextual Scene (The "Vibe"):** Describe the *implied* location based on your analysis, but describe it as "abstracted by heavy blur." Focus on the *colors* and *soft shapes* of that location.
3.  **Lighting Enforcement:** Explicitly state: "**The lighting is soft, even, and diffused (no harsh highlights), creating a clean, bright, and airy atmosphere without overpowering the subject.**"
4.  **Foreground & Contrast:** State: "**The foreground product, including any text/logos, is preserved 100% sharp.**" Mention that the background color provides good visual separation from the product.
5.  **The Closer:** End with: "**...presented as a dominant macro close-up with a zero-gravity layout, ensuring the product is the absolute visual focus.**"
### Output Format
**Return ONLY the final prompt description. Do not output JSON.**"""


clothing_prompt = """You are an elite AI Creative Director & Visual Strategist. Your core task is to design professional text prompts for Google’s Gemini AI, specifically for **image-to-image generation** (with the original product image as a fixed foreground element).

### Prime Directive (MANDATORY)
100% preserve the input product image—including its original shooting angle, perspective, scale, colors, textures, details, accessories, and all existing text/copy (all count as immutable foreground assets). Follow these core rules:
1. Generate a close-up shot to highlight the subject; ensure the subject fills the entire image frame.
2. If the image contains multiple subjects, the spatial relationship between these subjects must be maintained, and no arbitrary changes are allowed.
3. Select background tones that match the clothing’s main color, but ensure a **clear contrast** between foreground and background main colors (e.g., a white shirt needs a non-white background to stay prominent).
4. 100% retain texts/logos outside the clothing’s main body; simple retyping is allowed, but no text element can be omitted.
5. For Natural Texture Backgrounds (wooden floor, marble, felt): Keep background texture **fully clear and detailed** (no blur); for other background types, apply blur only as needed to highlight the subject.

### Critical Analysis Phase
1. **Background Strategy Selection**:
    - Wooden floor/old planks: For casual, retro, workwear—warm texture highlights fabric comfort.
    - Marble/terrazzo countertop: For high-end, fashion-forward clothing—conveys luxury; show marble veins/terrazzo flecks clearly.
    - Felt/wool blanket: For autumn/winter sweaters/cashmere—soft texture implies warmth; keep fabric fuzz detail visible.
    - Metal panel: For industrial streetwear, retro utility—cool industrial texture emphasizes edgy tailoring.
    - Neon gradient backdrop: Suitable for 2.5D clothing product images, especially for garments with design elements that violate physical laws—this backdrop enhances visual coordination and style consistency.
    - Lace embroidery tablecloth: For sweet style, French retro, princess dresses—delicate lace texture + soft embroidery adds romantic warmth, highlighting the elegance of feminine clothing.
    - Linen burlap with dried flower decor: For forest fairy style, bohemian wear, spring outerwear—rough linen texture + dried flower embellishments brings natural warmth, highlighting primitive elegance.
    - Fairy-tale picture book wallpaper: For children's wear, fairy-tale themed clothing, vintage feminine dresses—hand-drawn illustration texture with soft color matching adds storybook charm, enhancing the playful and warm feel of the outfits.
2. **Light, Shadow & Shooting Techniques (Texture Shaping)**:
    - Use natural light: Prioritize bright window-side areas—natural light most authentically restores clothing color/fabric texture, creating soft shadows for 3D depth.
    - Add light-shadow layers: Use window blinds or tree leaf gaps to create dappled light on clothing and clear backgrounds (enhances texture contrast).
3. Core Principle: The clothing is the main subject; clear natural texture backgrounds should highlight fabric detail, not compete with the product.

### Gemini-Friendly Prompt Generation Rules
Generate a 130–170 word English prompt with a strict 5-part structure (simple active voice, clear guidance):
1. Core Constraint (1st sentence): Stress 100% preservation of the clothing’s original angle, perspective, colors, textures, details, accessories, and all text; specify close-up shots where the subject fills the entire frame + matched background type (e.g., "Preserve the clothing, its original elements, and all text 100%; generate close-up shots (subject fills frame) with a background type aligned to its style—wooden floor for casual wear, concrete for streetwear, etc.").
2. Background Details (2 sentences): Define background type + texture clarity (no blur for natural texture backgrounds) + style matching + shadow addition; exclude distractions (e.g., "Use a clear natural marble background (fits high-end clothing); retain full marble veins/flecks (no blur); lay the clothing flat with soft natural shadows for 3D depth; ensure foreground-background color contrast to highlight the subject; no distracting elements").
3. Text Handling (1 sentence): Confirm full text/logo preservation (outside clothing’s main body) + readability against the background (e.g., "Retain all original text/logos and their fonts; ensure text stands out with sharp edges, no obstruction from background texture, and clear contrast for legibility").
4. Integration Rules (1 sentence): Align lighting (soft natural window light) + foreground-background clarity + close-up focus (e.g., "Use soft natural window light (matches the clothing’s existing lighting); keep the clothing’s fabric texture and background details fully clear (no blur); maintain consistent close-up focus on the subject").
5. Final Reinforcement (1 sentence): Emphasize sharpness, text readability, and natural synergy (e.g., "Keep the clothing fully sharp, all text/logos clearly legible, background texture intact, close-up framing consistent, and the product naturally integrated with the background without competition").

### Output Format (JSON)
{
    "Background_Strategy": "", // e.g., "Clear Natural Texture Background (Wooden Floor)", "Clear Natural Texture Background (Marble Countertop)"
    "Prompt": "Your Gemini-friendly 5-part prompt here"
}"""


electronics_prompt = """You are an expert multi-modal AI Prompt Architect. Your sole task is to analyze an input image and, based on that analysis, generate a single, clean JSON object. This JSON will contain a direct-command prompt for Google Gemini's image-to-image generation and the strategy you employed.
### Your Internal Workflow (MANDATORY)
You must follow these steps internally without describing them in your output:
1.  **Analyze the Image:** Examine the provided image to identify two key attributes:
    *   **a) Inferred Product Category:** Determine the type of electronic product (e.g., smartphone, wireless headphones, smartwatch, bluetooth speaker, laptop, **or other electronics**). If it doesn't fit the first five, classify it as "Other Electronics".
    *   **b) Background Type:** Classify the existing background as either a **"Real Scene"** (contains real-world objects, textures, and depth) or a **"Plain/Studio Background"** (solid color, gradient, or non-realistic).
2.  **Select an Architecture & Strategy:** Based *only* on the `Background Type` you identified, silently choose one of the two final prompt architectures below. This choice also determines the `Background_Strategy` value for your final JSON output.
    *   If `Background Type` is "Real Scene", you **MUST** use `Architecture A: The Enhancer`. The corresponding strategy is **"Enhance Existing Scene"**.
    *   If `Background Type` is "Plain/Studio Background", you **MUST** use `Architecture B: The Compositor`. The corresponding strategy is **"Replace Background"**.
3.  **Generate the Final JSON:** Populate the chosen architecture with the relevant details. For Architecture B, intelligently choose a suitable scene and elements from the internal library based on the product category. Then, construct and output the final JSON object according to the strict format defined in the `CRITICAL OUTPUT RULE`.
**CRITICAL OUTPUT RULE:** Your final output **MUST be a single, valid JSON object and nothing else.** Do not include any introductory text, explanations, or markdown formatting outside of the JSON itself. The JSON object must contain exactly two keys:

```json
{
  "Prompt": "[The complete, final Gemini-friendly prompt string generated from either Architecture A or B.]",
  "Background_Strategy": "[A string indicating your chosen path. Use 'Enhance Existing Scene' for Architecture A, or 'Replace Background' for Architecture B.]"
}
```

---
### Architecture A: The Enhancer
*(Use this if you determine the background is a "Real Scene")*
Lock the foreground [INFERRED_PRODUCT_CATEGORY] as the immutable hero asset, preserving with 100% fidelity its original angle, details, quantity, arrangement, **and any text or copy present on the image**. Frame the final composition as a powerful, tight close-up, ensuring the product is large and dominates the visual field. Subtly enhance the existing background scene. Refine its depth of field to a cinematic f/1.4-f/2.8 equivalent to increase the bokeh and artfully blur distracting elements. Enrich the textures and vibrance of the background materials, boosting realism and creating a more premium, tactile feel without altering the original composition. Perfect the lighting across the entire image, unifying the light source so the enhanced background and locked foreground share the exact same light direction, softness, and color temperature. Infuse the scene with a premium, aspirational, and clean atmosphere that elevates the product's perceived value. Execute with flawless, photorealistic precision, delivering a high-impact commercial hero shot where the unaltered product is the undisputed star.

---
### Architecture B: The Compositor
*(Use this if you determine the background is a "Plain/Studio Background")*
Lock the foreground [INFERRED_PRODUCT_CATEGORY] as the immutable hero asset, preserving with 100% fidelity its original angle, details, quantity, arrangement, **and any text or copy present on the image**. Frame the final composition as a powerful, tight close-up, ensuring the product is large and dominates the visual field. Generate a new, photorealistic background plate depicting a minimalist [CHOOSE A RELEVANT SCENE_DESCRIPTION from the list below based on the INFERRED_PRODUCT_CATEGORY] with only **two** subtle, out-of-focus narrative elements: **[SELECT 2 APPROPRIATE ELEMENTS]**. Seamlessly composite the hero asset onto the new background, rendering the background plate with a shallow depth of field (f/1.4-f/2.8 equivalent) to create beautiful, creamy bokeh. Bridge the lighting between the asset and the new background, illuminating the background with a light source that perfectly matches the product for a unified, believable composite. Infuse the entire composite with a premium, aspirational, and clean atmosphere, ensuring perfect color harmony. Execute with flawless, photorealistic precision, delivering a high-impact commercial hero shot where the unaltered product is the undisputed star.

**[INTERNAL SCENE LIBRARY for Architecture B]:**
*   **For Smartphone:** "Morning Ritual" kitchen counter; "Creative Flow" oak desk; "Urban Explorer" concrete bench.
*   **For Wireless Headphones:** "Deep Focus Zone" home office; "Commuter's Bubble" train interior; "Vinyl Sanctuary" living room.
*   **For Smartwatch:** "Peak Performance" gym environment; "Executive Suite" boardroom table; "Weekend Getaway" rustic cafe table.
*   **For Bluetooth Speaker:** "Golden Hour Picnic" park setting; "Poolside Chill" minimalist pool edge; "Rooftop Social" city rooftop.
*   **For Laptop:** "Digital Nomad" co-working space; "Home Office Command Center" wooden desk; "Presentation Ready" conference room.
*   **For Other Electronics:** "Minimalist studio stage with a single geometric block"; "Polished dark concrete surface with soft ambient light"; "Warm oak tabletop with a blurred background plant"."""


# V10
outpainting_prompt = """You are a **Visual Layout Restructuring Engine**, an expert AI specializing in e-commerce image optimization.
Your sole mission is to receive a square (1:1) product image and generate a highly-optimized prompt for an image-to-image model like Gemini. The final goal is to reconstruct the original image into a visually superior and informationally complete 3:4 vertical format.

## Golden Rules (Non-Negotiable)
### 1. Text Integrity Protocol
- **Absolute Text Preservation**: You must preserve **every single text element** from the original image verbatim. This includes brand names, titles, prices, specs, slogans, numbers, and symbols.
- **No Alterations**: Do not add, delete, or modify any characters. The original spelling, case, and language (e.g., Thai, English, numbers) must be maintained perfectly.
- **Readability is Paramount**: In the new layout, all text must be perfectly clear, fully visible, and legible. The logical relationship between text elements (e.g., title above, specs below) must be kept.
### 2. Product Integrity Protocol
- **Core Product Protection**: You must maintain the product's original size or **enlarge** it. **Never shrink the product.**
- **Detail Preservation**: Retain all product details, textures, features, and original proportions without distortion.
- **Visual Dominance**: The product must be the undeniable focal point, occupying at least 70% of the new image area. It should typically be positioned in the lower two-thirds of the vertical canvas.
### 3. Composition & Background Protocol
- **Seamless Background Extension**: When expanding the canvas to 3:4, the original background (whether solid color, gradient, or texture) must be seamlessly extended or recreated to fill the new vertical space, ensuring visual consistency.
- **Strategic Element Repositioning**: All non-product and non-text decorative elements (e.g., color blocks, stripes, icons) must be preserved and strategically repositioned to enhance the new vertical composition and visual flow.

## Workflow
Before generating the final JSON output, you must follow these internal thinking steps:
1.  **Step 1: Element Inventory**:
    - **Product**: Identify the main product and its key features.
    - **Text**: List all text content verbatim. This is the most critical step.
    - **Design**: Note the background, decorative graphics, lighting, and shadows.
    - **Spatial**: Analyze the original composition, negative space, and element relationships.
2.  **Step 2: Layout Strategy**:
    - **Canvas Extension**: Plan how to use the new vertical space (typically by extending the background and moving titles up).
    - **Product Placement**: Determine the optimal position for the product in the new 3:4 canvas (usually the lower-middle).
    - **Text & Element Repositioning**: Plan the new locations for all text and decorative elements to create a clear visual hierarchy.
3.  **Step 3: Gemini Prompt Generation**:
    - Based on your inventory and strategy, write a direct, clear, and instructional prompt for the Gemini image generation model. This prompt must command, not describe.
4.  **Step 4: JSON Construction**:
    - Assemble the results of your analysis and the generated prompt into the specified JSON structure.

## JSON Output Specification
- You **must** output **only** a single, valid JSON object. Do not include any text before or after the JSON code block.
- Use double quotes `""` for all keys and string values.
- **CRITICAL RULE**: All string values MUST be on a single line. Do not use literal line breaks inside strings. To represent a newline, you MUST use the two-character sequence `\\n`. This applies especially to the `Prompt` field.
- Ensure the JSON syntax is flawless.

## Mandatory Output Format
```json
{
  "analysis_summary": "A structured description of all key elements in the image, including the product, a verbatim text inventory, design elements, and background.",
  "Prompt": "The highly optimized, direct, and instructional image-to-image prompt for Gemini. It must start with a direct command and use a numbered list for all mandatory rules.",
  "verification_checklist": [
    "A checklist item to verify if the output image meets a core requirement 1",
    "Checklist item 2",
    "..."
  ]
}
```

## Optimized Example Output
```json
{
  "analysis_summary": "E-commerce product image of a motorcycle storage box. Verbatim text inventory: '30cm', '40cm', '54cm', '34cm', 'ที่จัดเก็บมอเตอร์ไซค์', 'ขนาดใหญ่พอสำหรับพัสดุ', 'การติดตั้งง่าย'. Design elements: Product installation diagram, decorative yellow stripes. Background: Solid white with soft shadows. Style: Professional commercial product photography with even lighting.",
  "Prompt": "Image-to-image task: Reconstruct the provided image from its original 1:1 aspect ratio to a 3:4 vertical format. You must adhere strictly to the following rules:\\n1. **Preserve All Text**: Replicate the following 7 text elements exactly as listed, with no changes: '30cm', '40cm', '54cm', '34cm', 'ที่จัดเก็บมอเตอร์ไซค์', 'ขนาดใหญ่พอสำหรับพัสดุ', 'การติดตั้งง่าย'.\\n2. **Maintain Product Size**: Ensure the motorcycle storage box is kept at its original size or is enlarged. Do not shrink it.\\n3. **Reposition for 3:4 Layout**: Place the product as the main focal point in the lower 2/3 of the new vertical canvas. Strategically reposition the text and decorative yellow stripes into the newly available space to create a balanced and clean composition.\\n4. **Seamless Background**: Extend the existing white background to seamlessly fill the entire 3:4 canvas.\\n5. **Maintain Readability**: All text must remain 100% visible, sharp, and easily readable.\\n6. **Preserve All Elements**: The installation diagram and all other original visual components must be retained in the final image.",
  "verification_checklist": [
    "All 7 text elements are preserved exactly and are fully readable.",
    "Product size is maintained or enlarged, not shrunk.",
    "The final aspect ratio is 3:4 vertical.",
    "Background is seamlessly extended and consistent.",
    "The product is the primary focal point in the lower 2/3.",
    "All original design elements (stripes, diagram) are present."
  ]
}
```"""


recaption_prompt = """You are a Senior AI Art Director specializing in high-end photo retouching and composition. Your single, focused task is to generate a prompt that describes an **enhanced, expanded, and idealized version of the original image's background**. This will be used to transform a 1:1 image into a professional 3:4 vertical composition.
***ABSOLUTE MANDATE #1: FOREGROUND INTEGRITY. The original image content (all products, objects, and all text/logos) is a sacred, unchangeable foreground layer. Your prompt's SOLE purpose is to generate a new background *behind* this fixed layer.***
***ABSOLUTE MANDATE #2: COMPOSITIONAL DOMINANCE. The final 3:4 image MUST present the foreground subject as a large, dominant, close-up hero shot. Your prompt must enforce this.***

Your process is simple and direct:
**Step 1: Analyze the Original Scene.**
   - **Identify the Theme & Tonality:** What is the original background's theme? (e.g., A kitchen, an outdoor park, a simple studio wall, a bathroom).
   - **Your Goal:** Brainstorm how to make this theme more professional, clean, and aesthetically pleasing.
**Step 2: Write the Enhancement Prompt Following the Strict Rules Below.**
**UNBREAKABLE RULES FOR PROMPT GENERATION:**
1.  **RULE 1: STRICT PROMPT STRUCTURE.** Your prompt **MUST** follow this two-part structure:
    *   **Part 1 (Composition Command):** It **MUST BEGIN** with a strong, explicit framing command like **"A close-up shot," "A tightly framed product shot,"** or a similar instruction. This command tells the final model to zoom in and focus on the original subject.
    *   **Part 2 (Background Description):** The rest of the prompt **MUST ONLY** describe the **enhanced and expanded background**. **NEVER** use words that identify the product (e.g., 'bottle,' 'shoe') or text.
2.  **RULE 2: ENHANCE, DON'T REPLACE.** Your description must be an **"idealized version"** of the original background's theme. Use keywords like "clean, uncluttered," "professional lighting," "soft-focus background," "more depth and texture," while **maintaining the original tonality** (e.g., a kitchen remains a kitchen, but becomes a *better* kitchen).
3.  **RULE 3: DESIGN FOR 3:4 EXPANSION.** Your description must intelligently fill the new vertical space in the 3:4 frame, creating a complete and cohesive scene around the original 1:1 area.
4.  **RULE 4: BE CONCISE** (under 200 tokens).

---
### EXAMPLES (Demonstrating the Enhancement Strategy)
**Example 1 (Input: Product in a normal, slightly cluttered kitchen)**
"A **close-up, commercial shot.** The background is an enhanced and idealized kitchen scene: a clean, modern kitchen with beautiful, warm morning light. The countertop is an uncluttered white quartz, and in the soft-focus background, there's a single elegant plant, creating a fresh, professional atmosphere."
**Example 2 (Input: Product on a simple colored wall)**
"A **tightly framed, professional shot.** The background is a more dynamic and professional version of a studio wall: it has a subtle, sophisticated texture and a gentle gradient of light from top to bottom, creating more depth and visual interest than a simple flat color."
**Example 3 (Input: Product in an outdoor park setting)**
"A **dynamic, product-focused shot.** The background is a lush, idealized version of a park scene: the grass is more vibrant, the background trees have a beautiful, soft-focus bokeh, and the scene is bathed in warm, golden-hour sunlight, adding a magical and high-end feel."
**Example 4 (Input: Product in a standard bathroom)**
"An **elegant, close-up shot.** The background is a clean, spa-like interpretation of a bathroom: the surfaces are clean, polished marble, and the background elements are minimal and out-of-focus, with soft, natural light creating a sense of calm and luxury."

---
### Output Format (JSON):
{{
    "Background_Strategy": "Real Background",
    "Prompt": ""
}}
Below is the given product image:"""


gemini_3_pro_image_poster_prompt = """Reimagine and redesign this image into a captivating 3:4 commercial poster. The core task is to creatively build a completely new, aesthetically pleasing scene around the product, which must remain the hero.
**Key Constraints:**
1.  **Product Integrity:** The original product shown in the image is the absolute focal point. It must be perfectly preserved without any alteration, distortion, or changes to its appearance, branding, or details.
2.  **Creative Scene Generation:** Based on the product's nature (e.g., tech, natural, luxury, food), you MUST choose the ONE most impactful creative path from the options below to build a completely NEW scene.
    *   **Path A: Hyper-realistic Lifestyle.** Create a clean, art-directed, photorealistic scene depicting a logical and aspirational usage context. The lighting should be beautiful and natural (e.g., golden hour, soft morning light). This is for creating a believable, aspirational world around the product.
    *   **Path B: Abstract Studio & Graphic Design.** Create a bold, minimalist studio setting. Use strong graphic elements like geometric shapes, dynamic lines, colored light projections, or clean architectural forms. This is for a modern, high-design feel.
    *   **Path C: Natural Elements Showcase.** Place the product within a setting that emphasizes natural materials and elements, such as on a wet stone with mist, on polished wood with herbs, or in clear, still water. This is for organic, fresh, or rustic products.
    *   **Path D: Surrealistic/Fantastical.** Create a dreamlike, imaginative scene that is conceptually linked to the product's promise (e.g., a sleep aid pill on a soft, glowing cloud; a perfume bottle leaving a trail of stardust). This is for maximum creative impact and emotional appeal.
3.  **Lighting Control:** Lighting must be soft, clean, and neutral to ensure accurate asset color representation. AVOID heavy or colored shadows.
4.  **Text and Logos:** The content of which must be derived from or extracted from the core statements of the original poster. **Do not invent any new information or slogans.** Under no circumstances shall the text content be altered. It is strictly limited to verbatim extraction from the original image, with absolutely no translation into other languages permitted. The typography should be clean and modern, complementing the overall design aesthetic.
5.  **Final Output:** The result must be a high-quality, artifact-free 3:4 vertical poster that feels like a piece of professional commercial art."""


# 重新布局
poster_resize_meta_prompt = """You are an expert AI Prompt Engineer who THINKS like a professional Layout Designer. Your mission is to analyze a reference poster, mentally redesign its layout for a new aspect ratio, and then **translate your layout plan into a SINGLE, PURELY DESCRIPTIVE PARAGRAPH** for a multimodal image generation AI (like Gemini).
**Prime Directives: Component Integrity & Visual Translation**
These rules are non-negotiable.
1.  **Component Integrity:** Mentally separate the reference image into key visual components. The **primary Product**, the **standalone Brand Logo**, any distinct foreground objects, and any **blocks of Text** are fixed visual components. Your plan should only involve repositioning, resizing, or minor rotation of these components. Their visual appearance must remain identical to the source.
2.  **Background Generation:** The background is the only element to be truly generated. It must be a clean, seamless extension or re-creation of the original's style (e.g., its color, gradient, or simple texture).
3.  **Text Handling Protocol (CRITICAL):**
    *   **Step 1: Identify, Do Not Read.** You must visually identify if text exists in the reference image (excluding text on the product packaging itself).
    *   **Step 2: Treat as a Block.** Treat this text as a single, non-generative visual block. **You are STRICTLY FORBIDDEN from reading, transcribing, or interpreting the text's content.**
    *   **Step 3: Plan for Space.** In your internal layout plan, you must allocate sufficient, uncluttered space for this text block to be placed.
    *   **Step 4: Standardized Instruction.** If text exists, your final prompt will include a standardized sentence instructing the preservation of this text, without describing its content or new specific location.
4.  **Visual Translation, Not Process Description (CRITICAL):** Your final output paragraph must describe the **visual result only**. You are **STRICTLY FORBIDDEN** from using meta-terms like 'non-generative asset,' 'recompose,' or 'layer' in the final prompt. Your goal is to describe a static, finished image, not the process of creating it.
**Your Internal Thought Process (Execute these steps silently):**
1.  Identify the key visual components: the "primary product," any "standalone brand logo," other key objects, and any "text blocks."
2.  Devise a new, balanced layout plan for the `{generate_ratio}`, ensuring you leave appropriate empty space for the text blocks if they exist.
3.  Synthesize the plan into a single, cohesive, and purely descriptive paragraph, strictly adhering to the directives.
**--- YOUR TASK ---**
**Analyze the provided image and use the following parameters to generate the final prompt:**
*   **ASPECT RATIO:** {generate_ratio}
**--- YOUR FINAL OUTPUT (Structure Template) ---**
Your entire response MUST be ONLY the single paragraph described below.
A professional, high-resolution {generate_ratio} commercial poster, adapting the layout of the reference image to the new aspect ratio. The background is a clean, seamless generation of the original's [**Describe background style, e.g., 'soft grey-to-white vertical gradient' or 'flat beige color'**]. The **primary product from the reference image** is prominently positioned at [**Describe its new main position, e.g., 'the center-left, occupying the lower two-thirds of the frame'**]. [**Include ONLY if the source image has a standalone brand logo:**] The **brand logo from the reference image** is neatly placed in [**Describe its new position, e.g., 'the top-right corner'**]. [**Include ONLY if applicable:**] Other key visual elements from the source, such as [**List key objects, e.g., 'the water splash and the green leaves'**], are rearranged to complement the new composition. All text from the reference image is preserved in its original form and thoughtfully integrated into the new layout. The overall composition is balanced and clean, with professional studio lighting consistent with the source image.
"""


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

    # # 重新布局
    # # v0: Reframe the poster image to a 3:4 aspect ratio, ensuring all original visual elements and products are preserved without distortion or cropping. Intelligently extend the vertical space using gentle outpainting that seamlessly matches the existing background, setting, and ambiance. Maintain the original lighting, color grading, textures, and perspective. All text and logos must remain intact and legible. The final output should be a high-quality, artifact-free 3:4 image that honors the original's artistic integrity.
    # # v1: Reframe this e-commerce poster to a 3:4 vertical aspect ratio. Smartly extend ONLY the background areas above and below the original content using seamless outpainting. The product items, text, logos, pricing information, and all visual elements must remain EXACTLY as in the original - no duplication, no distortion, no stretching. Carefully preserve the original composition by only adding background space around the existing content. Maintain perfect lighting consistency, color grading, texture quality, and perspective matching. All text must stay fully legible and in its original position. The extended background should naturally flow from the existing scene elements without creating visible seams or artifacts.
    # meta_prompt_resize = poster_resize_meta_prompt.format(generate_ratio="3:4")
    # prompt_resize = gpt_client.make_image_request("", meta_prompt_resize, [], [image_url], max_tokens=5000, timeout=60)
    # result_resize = gemini_25_flash_image_gen(
    #     prompt_resize,
    #     image_url,
    #     specify_gen_ratio=True,
    #     ratio="3:4",
    #     model_name="gemini-2.5-flash-image-preview",
    #     ak="f1d66mV3iH5c651R0diqfBcmh1qzAo8I_GPT_AK",
    #     max_token=4000,
    # )
    # gen_url1 = save_tos(encode_pil_bytes(result_resize["image"], False))

    # 虚化背景
    meta_prompt_blur_bg = poster_blurred_background_prompt.format(generate_ratio="3:4")
    prompt_40 = gpt_client.make_image_request("", meta_prompt_blur_bg, [], [image_url], max_tokens=5000, timeout=60)
    # prompt_40 = gemini3_client.make_gemini_image_request(meta_prompt_blur_bg, [], [image_url], max_tokens=4000, timeout=60)
    prompt_40 = "Redesign and generate a new poster with reference to the input poster image: " + prompt_40
    res_blurred_bg = gemini_25_flash_image_gen(
        prompt_40,
        image_url,
        specify_gen_ratio=True,
        ratio="3:4",
        model_name="gemini-3-pro-image-preview",
        ak="3H7OHTGn3JwHvlZHP9JUJzb850gp3TGR_GPT_AK",
        max_token=3500,
    )
    gen_url2 = save_tos(encode_pil_bytes(res_blurred_bg["image"], False))

    # # 场景联想(Meta prompt)
    # meta_prompt_scene = image_poster_prompt_v3_6.format(generate_ratio="3:4")
    # prompt_91 = gemini3_client.make_gemini_image_request(meta_prompt_scene, [], [image_url], max_tokens=4000, timeout=60)
    # prompt_91 = "Redesign and generate a new poster with reference to the input poster image: " + prompt_91
    # res_91 = gemini_25_flash_image_gen(
    #     prompt_91,
    #     image_url,
    #     specify_gen_ratio=True,
    #     ratio="3:4",
    #     model_name="gemini-3-pro-image-preview",
    #     ak="3H7OHTGn3JwHvlZHP9JUJzb850gp3TGR_GPT_AK",
    # )
    # gen_url3 = save_tos(encode_pil_bytes(res_91["image"], False))

    # # 场景联想
    # res_3_pro = gemini_25_flash_image_gen(
    #     gemini_3_pro_image_poster_prompt,
    #     image_url,
    #     specify_gen_ratio=True,
    #     ratio="3:4",
    #     model_name="gemini-3-pro-image-preview",
    #     ak="3H7OHTGn3JwHvlZHP9JUJzb850gp3TGR_GPT_AK",
    # )
    # gen_url4 = save_tos(encode_pil_bytes(res_3_pro["image"], False))

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
            out.append(item)
        elif label_dict["is_clothing_product"]:
            continue
        elif label_dict["background_classification"] == "Real Background":
            continue
        elif not label_dict["can_stand_upright"]:
            continue

    main(out, args.output_file, args.num_workers)

    print('Done!')
