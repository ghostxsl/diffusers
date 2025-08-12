# Copyright (c) wilson.xu. All rights reserved.
import argparse
from tqdm import tqdm
from diffusers.data.utils import load_file, json_save
from diffusers.data.byted.clients.azure_mllm import MLLMClient


sys_prompt = "You are a world-class AI Art Director and a specialized Prompt Engineer for advanced text-to-image models like Flux Kontext. Your mission is to generate a high-quality poster description prompt (it needs to be in English) based on the given poster image and the following requirements, which will be used to guide the Flux Kontext model."
prompt = """First, you need to proofread the text based on the provided poster image. The given text may contain errors, so you should provide the corrected text answer according to the image. Then, describe the poster image based on the proofread text. Finally, the answer should be presented in JSON format.
Task Requirements:
1. Poster Style Description:
1) Provide a appropriate style description based on the given poster image. (e.g., realistic photography style, fashion photography style, abstract expressionist style, minimalist style, romantic style, futurist style, conceptual art style, expressionist style, vintage/retro style, flat style, hand-drawn style, surrealist style)
2) The poster style description should be one sentence and ** no more than 40 words **.
3) Examples:
- Photorealistic rugged outdoor product photography style with crisp detail, high contrast, natural sunlight and cinematic depth of field, emphasizing tactile rubber and metal textures.
- High-tech product advertising poster in sleek, futuristic digital art style.
- Bright, high-impact e-commerce poster in hyper-realistic product photography style.
- Modern fashion photography style with a clean, sophisticated editorial finish.
- Minimalist luxury jewelry poster design with a high-resolution, modern e-commerce style.
- Luxury fashion photography style, polished and cinematic with dramatic warm highlights and soft shadows.

2. Product Description:
The image you receive may include one or more objects, people, or abstract graphics/backgrounds, etc. Please specify the position of the product on the poster.
'product_position': center/left/right/bottom-center/bottom-left/bottom-right/upper-center/upper-left/upper-right
1) Based on the provided product information (image and text), provide descriptions of the product and the scenarios.
2) You only need to identify the category of the content in the image, such as: woman, perfume bottle, sofa. No additional excessive descriptions of the object are required. (e.g., the original text inherent to the object itself, the printed text on cans, the usage instructions on hand cream, the text around an object.)
3) The product description should be one sentence and ** no more than 80 words **.
4) Examples:
- Central focus is an elegant, intricately designed diamond ring in white gold, shown upright and prominently centered with a flawless gem in a prong setting. Lighting is soft and refined, emphasizing the brilliance and fine curves of the ring.
- A single bright safety-orange rectangular portable rugged speaker  sits centered in the frame, positioned horizontally with the square end-cap facing the viewer and the body extending back and slightly to the right, anchored by a soft grounded shadow and subtle floor reflection to convey weight and portability.
- The main subject is a sleek, high-resolution tablet device angled forward. Accompanying the tablet is a metallic stylus, positioned in the foreground, and a dark, matte-finished tablet back angled behind and slightly to the side.
- The scene features a single ribbed dark emerald-green perfume bottle with ornate gold metal ends and a separate gilded decorative cap, the bottle positioned diagonally center-left, angled slightly toward the viewer and lying on a low-reflective ivory-marble surface that creates a subtle mirror of the bottle and cap, delicate warm key light from the top-left produces golden gleams on the metalwork and gentle shadowing, with faint floating powdery particles and a soft halo to evoke fragrance and softness. 

3. Text layout description:
1) Describe the placement of each text in the poster image, and enclose the text in quotation marks and specify its position. (e.g., top-left corner, bottom-right corner) ** This text should remain unaltered and not translated. **
2) Describe the font style for each text. (e.g., serif, sans-serif, script (handwritten), bold/black, vintage/retro, and display (creative).)
3) Describe the font size (using small, medium, large to describe) and font color (e.g. red, orange, yellow, green, blue, purple, pink, white, black, gray, brown, beige, cream, light red, dark red, pale orange, deep orange, lemon yellow, golden yellow, light green, dark green, sky blue, navy blue, light purple, dark purple, light pink, hot pink, light gray, dark gray, cyan, magenta, turquoise, lavender, coral, salmon, olive, khaki, maroon, burgundy, teal, silver, gold, bronze, ivory) for each text.
4) For the text entered by users, proofreading should mainly focus on the **content and position of the text**, and the response shall be based on the content in the image. For example: If a word appears in **all uppercase form** in the image, the word entered by the user should be rewritten into all uppercase form (e.g., At the very top center, within a gold rectangular plaque, large uppercase serif text "PRADA SUNGLASSES" in white.).
5) Do not include descriptions of how the text is arranged into lines in the text layout description, such as "large serif brown text reads 'Hook On Chair' in two stacked lines". Instead, provide an overall description of the text layout, for example: "In the center position below the title, the brown bold text 'Hook On Chair'".
6) The text layout description should use simple sentence structures. It only needs to state the placement, font style, color and size of each text, ** and must not exceed 140 words **.
7) Examples:
- At the very top center, large, light brown, modern sans-serif text "Apple Watch Series 10". Below this, centered, medium-sized black sans-serif text "Ultra-thin, Largest Display Yet".
- At the top center, large uppercase serif text in a light metallic silver tone reads "MADE IN COOKWARE". On the left side above the pan, medium-sized black sans serif text "5-ply metal for flawless cooking". On the right side beside the pan, smaller serif black text "Induction-ready precision" above "Durable Italian craftsmanship" with a balanced vertical arrangement.

4. Background Description:
1) Describe the background based on the poster image, and briefly describe the elements in the background.
2) Background description should only focus on the background and exclude irrelevant content (e.g., theme, composition), with a limit of ** no more than 80 words **.
3) Examples:
- The background is a luminous gradient blending warm sandy beige into deep emerald near the bottom-right, lightly textured with soft-focus bokeh and powder-like specks to convey warmth, luxury, and soft, powdery elegance.
- The background is a dramatic outdoor scene with a rocky campsite foreground in soft focus, distant mountain ridgeline at golden hour, warm rim light and subtle lens flare, faint airborne dust and water-spray bokeh to imply durability and water resistance, palette of warm amber highlights and cool slate blues to contrast the charcoal product and reinforce a premium, adventure-ready mood.

Output Prompt Examples:
1. {{
'poster_style_description': 'Realistic commercial product photography style with crisp detail, controlled studio lighting, soft cinematic shadows and subtle specular highlights.',
'product_position': 'center',
'product_description': 'A single handheld shower head in matte gunmetal finish, presented upright and slightly angled toward the viewer's left, centered in the composition and occupying the dominant foreground; fine water micro-droplets on the nozzles and a soft natural reflection on the handle communicate usability and premium finish while a delicate shallow depth of field keeps focus on the head.',
'text_layout_description': 'At the top center, large silver text \"casataref EcoSpa Shower\" in Luxury Minimal Font; below center, medium dark teal text \"Transform Your Shower Experience\" in Bold Sans-serif; at the lower center, small light gray text \"Relax, Refresh, Rejuvenate\" in Elegant Serif.',
'background_description': 'The background is a soft aqua-to-ivory gradient—pale to warm—with faint steamed-glass bokeh orbs and gentle, sheeny water ripples, evoking a tranquil, cozy spa atmosphere.'
}}
2. {{
'poster_style_description': 'Modern product poster in high-resolution commercial style.',
'product_position': 'center',
'product_description': 'The central focus is a prominent bright red trigger spray bottle with a yellow nozzle, standing upright and facing forward, clearly displaying its label and details. The bottle is lit with soft shadows, accentuating its contours for a clean, professional look.',
'text_layout_description': 'At the very top left, large dark green bold sans serif text \"Sevin Insect Killer\". In the top right quadrant, inside a large yellow polygonal box, uppercase sans serif text appears in black: \"KILLS 700+ PESTS IN MINUTES\". On the left midsection, vertically stacked medium-sized dark green serif text \"Safe for Plants & Blooms\" and \"Simple Shake & Spray\".',
'background_description': 'The background features a soft, pastel gradient shifting from light cream at the top to pale green at the bottom, adorned with subtle illustrations of flowers on the left and ripe tomatoes and leafy plants on the right, all watercolor-like and gently faded for a fresh, garden-inspired atmosphere.'
}}
3. {{
'poster_style_description': 'Luxury product poster in high-end fashion photography style, featuring dramatic lighting and glossy visual effects. ',
'product_position': 'left',
'product_description': 'On the left is a sleek high-heeled stiletto-shaped perfume bottle, rendered in deep midnight blue and black with a metallic gold heel, standing upright and exuding an elegant, glamorous aura. ',
'text_layout_description': 'At the top right, inside a thin gold-edged square, large serif uppercase text \"XO KHLOÉ\" and below, medium serif text \"Perfume\", all in gold. On the top left side, mid-height, large serif uppercase gold text \"ROSE, PRALINE & MUSK ELEGANCE\". At the bottom right, medium-sized clean sans-serif uppercase white text \"LUXURIOUS, CELEBRITY-INSPIRED FRAGRANCE\". ',
'background_description': 'The background is a deep satin navy blue, subtly illuminated with golden highlights that echo the shimmer around the bottle and petals, imparting a sense of opulence and nighttime glamour. Rose petals float diagonally across the right side, accentuated by soft swirling lines of golden shimmer, adding a sense of sophistication and allure. '
}}

Below is the Text to be referenced or corrected. Please refer to the text in the image; the user's input is for reference only:
"""


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--output_file",
        default="output.json",
        type=str,
        help="Path to image list on vos.")

    parser.add_argument(
        "--rank",
        default=None,
        type=int)
    parser.add_argument(
        "--num_ranks",
        default=None,
        type=int)

    args = parser.parse_args()

    return args


def _format_prompt_string(input_str):
    try:
        input_str = eval(input_str)
    except:
        input_str = input_str

    if isinstance(input_str, str):
        return f"\"{input_str}\""
    elif isinstance(input_str, list):
        input_str = [f"\"{a}\"" for a in input_str]
        return ", ".join(input_str)
    else:
        return ""


def main(args):
    gpt_client = MLLMClient(model_name="gpt-5-mini-2025-08-07")

    info_dict = load_file(args.input_file)
    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        info_dict = info_dict[args.rank::args.num_ranks]

    out = []
    for item in tqdm(info_dict):
        try:
            input_text = [_format_prompt_string(item['product_name']),
                          _format_prompt_string(item['primary_selling_points']),
                          _format_prompt_string(item['secondary_selling_points'])]
            if 'text_ctas' in item:
                input_text.append(_format_prompt_string(item['text_ctas']))
            input_text = "; ".join(input_text)
            gpt_result = gpt_client.make_image_json_request(
                sys_prompt,
                prompt + input_text,
                image_urls=[item['poster_url']],
                max_tokens=5000,
            )
            item['gpt_prompt_result'] = gpt_result
            item['prompt'] = gpt_result['poster_style_description'] + gpt_result['product_description'] + \
                         gpt_result['text_layout_description'] + gpt_result['background_description']

            out.append(item)
        except Exception as e:
            print(item, e)

    json_save(out, args.output_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
