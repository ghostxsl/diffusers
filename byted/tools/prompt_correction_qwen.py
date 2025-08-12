# Copyright (c) wilson.xu. All rights reserved.
import argparse
from tqdm import tqdm
from diffusers.data.utils import load_csv_or_xlsx_to_dict, xlsx_save, load_file
from diffusers.data.byted.clients.azure_mllm import MLLMClient


sys_prompt = "You are a world-class AI Art Director and a specialized Prompt Engineer for advanced text-to-image models like Flux Kontext. Your mission is to generate a high-quality poster description prompt (it needs to be in English) based on the given poster image and the following requirements, which will be used to guide the Flux Kontext model."
prompt = """
Task Requirements:
1. Poster Style Description: Provide a appropriate style description based on the given poster image. (e.g., realistic photography style, fashion photography style, abstract expressionist style, minimalist style, romantic style, futurist style, conceptual art style, expressionist style)
2. Main Subject Description: Briefly describe the content in the image, which should include the main subject and scene. If there is text on the object, please do not describe the text on the object; only describe the object's appearance. (e.g., The central focal point is a rugged portable speaker facing directly forward.)
3. Text layout description: Describe all the text appearing in the poster (does not include the original text inherent to the object itself, such as the printed text on cans, the usage instructions on hand cream, the label text on the perfume bottle, and so on.), which should include the text content, the placement of the text, the font size, the font style, and the font color. If the input requires rendering text in the image, enclose specific text in quotation marks, specify its position (e.g., top-left corner, bottom-right corner) and style. This text should remain unaltered and not translated.
4. Background Description: Briefly describe the background in the image. (e.g., The background features a luminous light blue gradient with large, soft-focus water droplets suspended throughout, softly blurred to convey freshness, hydration, and airiness.)
5. For the text entered by users, proofreading should mainly focus on the **content and position of the text**, and the response shall be based on the content in the image. For example: If a word appears in **all uppercase form** in the image, the word entered by the user should be rewritten into all uppercase form (e.g., At the very top center, within a gold rectangular plaque, large uppercase serif text "PRADA SUNGLASSES" in white.).
6. Do not include descriptions of how the text is arranged into lines in the text layout description, such as "large serif brown text reads 'Hook On Chair' in two stacked lines". Instead, provide an overall description of the text layout, for example: "In the center position below the title, the brown bold text 'Hook On Chair'".
7. The output prompt should ensure the expression of natural language fluency, and expressions such as 'Prompt:', 'Main subject:', 'Background:' and 'Text layout:' should not appear.
8. Except for the text layout, no additional description of the copy entered by the user is required.
9. Please ensure that the Prompt is less than 300 words and does not contain line breaks.

Prompt examples:
1. Dunhuang mural art style: Chinese animated illustration, masterwork. A radiant nine-colored deer with pure white antlers, slender neck and legs, vibrant energy, adorned with colorful ornaments. Divine flying apsaras aura, ethereal grace, elegant form. Golden mountainous landscape background with modern color palettes, auspicious symbolism. Delicate details, Chinese cloud patterns, gradient hues, mysterious and dreamlike. Highlight the nine-colored deer as the focal point.
2. Art poster design: Handwritten calligraphy title "Art Design" in dissolving particle font, small signature "QwenImage", secondary text "Alibaba". Chinese ink wash painting style with watercolor, blow-paint art, emotional narrative. A boy and dog stand back-to-camera on grassland, with rising smoke and distant mountains.
3. Minimalist product poster in high-resolution modern e-commerce style. Central focus: a sleek, upright skincare sunscreen bottle, pure white with an elegant glossy black cap, positioned on a pristine reflective surface. Subtle shadows and gentle rim lighting enhance its contours, highlighting bottle details without distraction. At the top center, place bold, large black text: "Invisible Comfort, Hydration All Day". Directly beneath, centered in smaller clean black font, add: "7 Hyaluronic Acids". Lighting is diffused and cool, emphasizing the hydration theme and product purity. Product label details are visible but understated, ensuring the bottle remains the focal point. The background features a luminous light blue gradient with large, soft-focus water droplets suspended throughout, softly blurred to convey freshness, hydration, and airiness. Overall composition is spacious and uncluttered, with a premium, comfortable atmosphere.
4. Luxury fashion e-commerce poster in high-end fashion photography style. The handbag is placed centrally as the hero subject, resting on a pristine, light cream surface with soft, realistic shadowing for depth. At the top center, large bold uppercase text “MICHAEL KORS ROMEe HANDBAG” appears in dark brown, modern sans serif font, followed immediately underneath by a slim horizontal gold line, and then a smaller dark brown tagline: “Elegant and versatile handbag perfect for summer outings.” All text is center-aligned, layered cleanly over the top half of the background to maintain crisp legibility. The backdrop is a gradient blend of warm beige shifting into olive green, adorned with faint, geometric gold linework for an understated, modern elegance. The composition exudes luxury, sophistication, and seasonal appeal, spotlighting the handbag with tasteful lighting, meticulous detail, and an uncluttered, fashionable look.

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
        default="output.xlsx",
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
    gpt_client = MLLMClient(model_name="gpt-4.1-2025-04-14")
    try:
        info_dict = load_csv_or_xlsx_to_dict(args.input_file)
    except:
        info_dict = load_file(args.input_file)

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks

        info_dict = info_dict[args.rank::args.num_ranks]

    for item in tqdm(info_dict):
        try:
            input_text = ",".join([
                _format_prompt_string(item['text_product_names']),
                _format_prompt_string(item['text_primary_selling_points']),
                _format_prompt_string(item['text_secondary_selling_points'])
            ])
            result_prompt = gpt_client.make_image_request(
                sys_prompt,
                prompt + input_text,
                image_urls=[item['gen_url']],
                max_tokens=800,
            )
            item['gpt_correction_prompt'] = result_prompt
        except Exception as e:
            print(item, e)

    xlsx_save(info_dict, args.output_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
