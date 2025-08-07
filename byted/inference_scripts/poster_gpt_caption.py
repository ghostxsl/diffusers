import argparse
from tqdm import tqdm
import pandas as pd

from diffusers.data.openai.azure_mllm import MLLMClient


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        help="File to input.")
    parser.add_argument(
        "--save_file",
        default="output.xlsx",
        type=str,
        help="File to save.")

    args = parser.parse_args()
    return args


sys_prompt = "You are a world-class AI Graphic Designer and a specialized Prompt Engineer for advanced text-to-image models like Flux Kontext. Your mission is to generate a high-quality graphic design prompt (it needs to be in English) based on the given reference image and the following requirements, which will be used to guide the Flux Kontext model."


prompt = """This is a poster material image. The product name is {product_name}. The primary selling point of the product is {primary_selling_point}. The secondary selling points are {secondary_selling_points}. You need to design an e-commerce-style product sales poster based on the product name, its primary and secondary selling points, and in combination with this poster material image.

Please complete the following tasks:
1.Overall image description: Briefly describe the content of the image, which should include the main subject, scene, style, and background.
2.Text layout design: Provide a creative layout design based on the given product name, primary selling points, and secondary selling points. It should include the text content, the placement of the text, font size, font color, and font style.
3.Image prompt: Generate a high-quality image prompt based on the above descriptions.

Note: Please treat tasks 1 and 2 as the thinking process, and only output the answer to the third task "Image Prompt".

Example of image prompt:
1.Create a high - end fashion - themed Black Friday promotional poster. Feature a stylish woman wearing a camel coat, denim inner layer, gray beanie, and sunglasses, leaning against a gray building wall. On the right side, there's a black graphic area with a splash - ink texture. Place “Black Friday” in large, bold, yellow, sans - serif font at the top of the black area. Below it, put “MEJATO LOPPINENS” in medium - sized, white, simple modern sans - serif font. Then add “BUY ANY ITEM IN OUR STORES OR ONLINE AND GET A DISCOUNT” in smaller, light gray, clear - readable sans - serif font. At the bottom, inside a white rectangle, display “NOVEMBER 29” in medium - sized, bold, white sans - serif font. The overall style is urban and fashionable, with a simple gray building wall as the background.
2.Create an e - commerce - style product sales poster for “Super Farm Japanese Wagyu Ribeye A4 (5 pcs)”. Feature a piece of marbled Japanese Wagyu ribeye with wisps of smoke rising, placed on a wooden surface. At the top center, use large, bold, dark red serif font for “Super Farm Japanese Wagyu Ribeye A4 (5 pcs)”. Below it, in medium - sized, black sans - serif font, show the main selling point “Premium Japanese Wagyu, melt - in - your - mouth tenderness”. Then, in smaller, gray sans - serif font, list the secondary selling points “350 - 400g per piece, perfect for grilling”, “A4 - grade marbling for rich, juicy flavor”, “Elevate your BBQ with gourmet quality beef”. The overall style is high - end and appetizing, with a warm, gradient background that highlights the beef's allure.
3.Create an e-commerce - style product sales poster for “Noble Panacea Chronobiology Sleep Mask”. Feature the white “Noble Panacea” product bottle placed on a white cube, with a soft beige background casting gentle shadows. At the top center, use large, elegant, black serif font for “Noble Panacea Chronobiology Sleep Mask”. Below it, in medium - sized, black sans - serif font, list the main selling point “Wake up to firmer, radiant skin”. Then, in smaller, gray sans - serif font, add the secondary selling points “Timed ingredient release for overnight repair”, “Detoxifies and nourishes while you sleep”, “Clinically proven to plump and energize skin”. The overall style is minimalist and high - end, with the soft beige background enhancing the product's elegance.
"""


gpt_4_1_client = MLLMClient(model_name="gpt-4.1-2025-04-14")


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
        raise Exception(f"Error `input_str` type({type(input_str)})")


def get_gpt_caption(image_url, product_name, primary_selling_point, secondary_selling_points):
    user_prompt = prompt.format(
        product_name=_format_prompt_string(product_name),
        primary_selling_point=_format_prompt_string(primary_selling_point),
        secondary_selling_points=_format_prompt_string(secondary_selling_points)
    )
    return gpt_4_1_client.make_image_request(sys_prompt, user_prompt, image_urls=[image_url], max_tokens=512)


def main(args):
    if args.input_file.endswith('.xlsx'):
        df = pd.read_excel(args.input_file)
    elif args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file, encoding='utf-8')
    else:
        raise Exception(f"Error `input_file` type:{args.input_file}")

    row_dict = df.to_dict('records')

    for i, line in tqdm(enumerate(row_dict)):
        img_with_bg_url = line['bg_res_url']
        try:
            # 1. Using GPT to get prompt
            product_name = line['result_product_name']
            primary_selling_point = line['result_primary_selling_points']
            secondary_selling_points = line['result_secondary_selling_points']
            prompt = get_gpt_caption(img_with_bg_url, product_name, primary_selling_point, secondary_selling_points)
            line['flux_kontext_prompt'] = prompt
        except Exception as e:
            print(i, img_with_bg_url, e)

    df = pd.DataFrame(row_dict)
    df.to_excel(args.save_file, index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
