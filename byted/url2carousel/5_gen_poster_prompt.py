# Copyright (c) wilson.xu. All rights reserved.
import argparse
import json
from tqdm import tqdm

from diffusers.data.utils import json_save, load_file
from diffusers.data.byted.clients.azure_mllm import MLLMClient


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        help="File to input.")
    parser.add_argument(
        "--output_file",
        default="output.json",
        type=str,
        help="File to save.")

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


prompt = """You are a world-class AI Art Director and also a professional Prompt Engineer specializing in advanced text-to-image models such as Flux Kontext. Your task is to generate a high-quality poster description prompt (which must be in English). This prompt will be used to guide the creation process of advanced text-to-image models like Flux Kontext.

First, you need to design a themed poster with a distinct style based on the given product image, product name, and selling points. Then, you need to convert the design language into prompts that can guide the text-to-image model to generation. Finally, the answer should be presented in JSON format.
The resolution of the poster you need to design is 928×1664 (width × height), the aspect ratio is 9:16.
Task Requirements:
1. Poster Style Description:
1) Provide a appropriate style description based on the given product image. (e.g., realistic photography style, fashion photography style, abstract expressionist style, minimalist style, romantic style, futurist style, conceptual art style, expressionist style, vintage/retro style, flat style, hand-drawn style, surrealist style)
2) The poster style description should be one sentence and ** no more than 40 words **.
3) Examples:
- Photorealistic rugged outdoor product photography style with crisp detail, high contrast, natural sunlight and cinematic depth of field, emphasizing tactile rubber and metal textures.
- High-tech product advertising poster in sleek, futuristic digital art style.
- Bright, high-impact e-commerce poster in hyper-realistic product photography style.
- Modern fashion photography style with a clean, sophisticated editorial finish.
- Minimalist luxury jewelry poster design with a high-resolution, modern e-commerce style.
- Luxury fashion photography style, polished and cinematic with dramatic warm highlights and soft shadows.

2. Product Description:
The product image you receive may include one or more objects, people, or abstract graphics/backgrounds, etc. Please specify the position of the product on the poster.
'product_position': center/left/right/bottom-center/bottom-left/bottom-right
1) Based on the provided product information (image and text), clarify the product's usage scenarios, and finally provide descriptions of the product and the scenarios.
2) If only the upper/lower body of the person is shown in the product image, you can design to add a photo frame around the person. (e.g., The person on the right lies in an arched area, as though presented from a specific window or frame.)
3) If a model appears in the image, please do not blur their face. Instead, briefly describe their attire and design a scene to showcase them. (e.g., Centered, a European model wears a floor-length gold and silver sequin prom gown with a deep V neckline and high thigh slit, posed full-length to showcase fit and movement for prom, red-carpet and gala occasions.)
4) If the product image features an object, select an appropriate placement position based on the object's proportions. For example: A handheld vacuum cleaner is located at the bottom-right; Central composition features a single, ultra-soft, white pillow.
5) You only need to identify the category of the content in the product image, such as: woman, perfume bottle, sofa. No additional excessive descriptions of the object are required. (e.g., the original text inherent to the object itself, the printed text on cans, the usage instructions on hand cream, the text around an object.)
6) The product description should be one sentence and ** no more than 80 words **.
7) Examples:
- A single bright safety-orange rectangular portable rugged speaker  sits centered in the frame, positioned horizontally with the square end-cap facing the viewer and the body extending back and slightly to the right, anchored by a soft grounded shadow and subtle floor reflection to convey weight and portability.
- The main subject is a sleek, high-resolution tablet device angled forward. Accompanying the tablet is a metallic stylus, positioned in the foreground, and a dark, matte-finished tablet back angled behind and slightly to the side.
- Two elegant bronze-orange skincare bottles are at the bottom right, one squeeze tube and one pump bottle, both displayed upright as the central focus on a glossy, luminous surface. A radiant golden liquid flows beneath and around the bottles, enhancing their premium feel and reflecting highlights.
- The main subject is a woman standing upright in the center foreground, dressed in a relaxed-fit, slate gray lounge set with sleeveless top and flowing pants, positioned beside a modern, low-profile gray sectional sofa with matching pillow. The overall scene evokes comfort and casual elegance.

3. Text layout description:
1) Determine the placement of each text in the poster, and enclose the text in quotation marks and specify its position. (e.g., top-left corner, bottom-right corner) ** This text should remain unaltered and not translated. **
2) Provide the appropriate font style for each text. (e.g., serif, sans-serif, script (handwritten), bold/black, vintage/retro, and display (creative).)
3) Provide appropriate font size (using small, medium, large to describe) and font color (e.g. red, orange, yellow, green, blue, purple, pink, white, black, gray, brown, beige, cream, light red, dark red, pale orange, deep orange, lemon yellow, golden yellow, light green, dark green, sky blue, navy blue, light purple, dark purple, light pink, hot pink, light gray, dark gray, cyan, magenta, turquoise, lavender, coral, salmon, olive, khaki, maroon, burgundy, teal, silver, gold, bronze, ivory) for each text.
4) The text layout description should use simple sentence structures. It only needs to state the placement, font style, color and size of each text, ** and must not exceed 140 words **.
5) Examples:
- At the very top center, large, light brown, modern sans-serif text "Apple Watch Series 10". Below this, centered, medium-sized black sans-serif text "Ultra-thin, Largest Display Yet".
- At the top center, large uppercase serif text in a light metallic silver tone reads "MADE IN COOKWARE". On the left side above the pan, medium-sized black sans serif text "5-ply metal for flawless cooking". On the right side beside the pan, smaller serif black text "Induction-ready precision" above "Durable Italian craftsmanship" with a balanced vertical arrangement.

4. Background Description:
1) Design a relevant background based on the product and its selling points, ensuring a certain level of aesthetic design, and briefly describe the elements in the background.
2) Background description should only focus on the background and exclude irrelevant content (e.g., theme, composition), with a limit of ** no more than 80 words **.
3) Examples:
- The background features a luminous light blue gradient with large, soft-focus water droplets suspended throughout, softly blurred to convey freshness, hydration, and airiness.
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
'poster_style_description': 'Luxury product poster in high-end fashion photography style, featuring dramatic lighting and glossy visual effects.',
'product_position': 'left',
'product_description': 'On the left is a sleek high-heeled stiletto-shaped perfume bottle, rendered in deep midnight blue and black with a metallic gold heel, standing upright and exuding an elegant, glamorous aura.',
'text_layout_description': 'At the top right, inside a thin gold-edged square, large serif uppercase text \"XO KHLOÉ\" and below, medium serif text \"Perfume\", all in gold. On the top left side, mid-height, large serif uppercase gold text \"ROSE, PRALINE & MUSK ELEGANCE\". At the bottom right, medium-sized clean sans-serif uppercase white text \"LUXURIOUS, CELEBRITY-INSPIRED FRAGRANCE\".',
'background_description': 'The background is a deep satin navy blue, subtly illuminated with golden highlights that echo the shimmer around the bottle and petals, imparting a sense of opulence and nighttime glamour. Rose petals float diagonally across the right side, accentuated by soft swirling lines of golden shimmer, adding a sense of sophistication and allure.'
}}

Below is the product_name{input_label}, ** keep the text exactly unchanged (including the same capitalization), do not add any extra text **:
"""


def _format_prompt_string(input_str):
    if isinstance(input_str, str):
        try:
            input_str = eval(input_str)
        except:
            try:
                input_str = json.loads(input_str)
            except:
                input_str = input_str

        return input_str
    elif isinstance(input_str, list):
        return ", ".join(input_str)
    else:
        raise Exception(f"Error `input_str` type({type(input_str)})")


def remove_special_mark(input_str, special_chars = ('™', '®')):
    for char in special_chars:
        input_str = input_str.replace(char, '')
    return input_str


def main(args):
    gpt_client = MLLMClient(model_name="gpt-5-mini-2025-08-07")

    data = load_file(args.input_file)
    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        data = data[args.rank::args.num_ranks]

    out = []
    for line in tqdm(data):
        try:
            for image_url, text_carousel, image_tag in zip(line['image_urls'], line['text_carousels'], line['image_tags']):
                try:
                    input_text = {"product_name": remove_special_mark(text_carousel['product_name'])}
                    input_label = ""

                    if len(text_carousel['primary_selling_points']) > 0:
                        input_text["primary_selling_points"] = remove_special_mark(text_carousel['primary_selling_points'][0])
                        input_label += ", primary_selling_points"

                    if len(text_carousel['secondary_selling_points']) > 0:
                        input_text["secondary_selling_points"] = remove_special_mark(text_carousel['secondary_selling_points'][0])
                        input_label += ", secondary_selling_points"

                    user_prompt = prompt.format(input_label=input_label)
                    user_prompt += json.dumps(input_text)

                    product_url = image_tag['truncated_res']['subject_image_url']
                    gpt_res = gpt_client.make_image_json_request(
                        "", user_prompt, image_urls=[image_url], max_tokens=6000)

                    gpt_prompt = gpt_res['poster_style_description'] + gpt_res['product_description'] + \
                        gpt_res['text_layout_description'] + gpt_res['background_description']

                    out.append({
                        "ad_id": line['ad_id'],
                        "ad_country": line["ad_country"],
                        "advertiser_id": line['advertiser_id'],
                        "l3_vertical_tag": line['l3_vertical_tag'],
                        "external_website_url": line['external_website_url'],
                        "image_url": image_url,
                        "product_name": input_text.get('product_name', ''),
                        "primary_selling_points": input_text.get('primary_selling_points', ''),
                        "secondary_selling_points": input_text.get('secondary_selling_points', ''),
                        "product_url": product_url,
                        "gpt_prompt_result": gpt_res,
                        "prompt": gpt_prompt,
                    })

                    if len(out) % 10 == 0:
                        json_save(out, args.output_file)
                except Exception:
                    continue
        except Exception as e:
            print(line, e)

    # Final Save
    json_save(out, args.output_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
