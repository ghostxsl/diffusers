import os
from tqdm import tqdm
from os.path import splitext, basename
from diffusers.data.utils import load_file, json_save
from diffusers.data.clients.azure_mllm import MLLMClient


info_dict = load_file("poster_train_data_1/infos.json")
image_paths = [line["poster"] for line in info_dict.values()]

sys_prompt = "You are a world-class AI Art Director and a specialized Prompt Engineer for advanced text-to-image models like Flux Kontext. Your mission is to generate a high-quality poster design prompt (it needs to be in English) based on the given reference image and the following requirements, which will be used to guide the Flux Kontext model."
prompt = """
This is a poster image. Please complete the following tasks:

1. Overall image description: Briefly describe the content in the image, which should include the main subject, scene, style, background, and the promotional theme of the poster.
2. Text layout description: Describe all the text appearing in the poster, which should include the text content, the placement of the text, the font size, the font color, and the font style.
3. Image prompt: Generate a high-quality image prompt based on the above description.

Note: Please treat tasks 1 and 2 as the thinking process, and only output the answer to the third task "Image Prompt".

Image prompt examples:
1.Create a poster - style image. On a dark, solid - colored background, arrange a variety of fresh organic vegetables (tomatoes, garlic, peppers, leafy greens, etc.) in a natural, scattered way on the left side. On the right side, within a white rectangular border, place the text “Advantages of Organic Food” in an elegant, serif font, with “Advantages” and “Organic Food” in larger sizes and “of” in a smaller, delicately placed size, to promote the benefits of organic food, with an overall natural and healthy style.
2.Create an image for an Easter - themed decoration idea collection. Feature a close - up of a nest filled with speckled eggs and decorative feathers placed on a dark, textured surface. The overall style should be warm, natural, and festive. Add text at the top center: “A collection of the best ideas” in a small, clean font; below it, “Decorating for Easter” in a larger, prominent font. At the bottom center, include “Edited by” in a small font and “Renata Hewitt” in a medium - sized font. The background should be a dark, understated surface to highlight the Easter elements.
3.Create a poster with a white background. The main element is a bear cub silhouette, inside which is a misty forest scene with dense green and yellowish trees. At the top center, there are two horizontal lines on both sides of the large font 'BEAR CUBS', and below it is the slightly smaller font 'PLAYING IN THE FOREST' in a simple and modern font style, presenting a natural and fresh theme of bear cubs playing in the forest.
4.Create a high-resolution poster featuring a stack of three donuts with different vibrant toppings: the top donut covered in white icing and colorful sprinkles, the middle donut with pink icing and white flakes, and the bottom donut with glossy chocolate icing, all centrally stacked on a simple solid pale pink background. The lower quarter of the poster should have a wavy, hot pink overlay, scattered with small pastel sprinkle illustrations. In the center of this section, add large, bold white text in a rounded sans-serif font: “3 new donut flavor”. Below this, place a small, centered upward arrow icon, and directly underneath it, the phrase “Swipe up” in a smaller, pale yellow sans-serif font. The overall style should be playful, modern, and enticing, designed to highlight exciting new donut flavors.

"""
gpt_4_1_client = MLLMClient(model_name="gpt-4.1-2025-04-14")


for img_path in tqdm(image_paths):
    try:
        result_prompt = gpt_4_1_client.make_image_request(sys_prompt, prompt, image_paths=[img_path], max_tokens=512)
        key = basename(img_path).split('_')[0]
        info_dict[key]['gpt_caption'] = result_prompt
    except Exception as e:
        print(img_path, e)

json_save(info_dict, "infos_gpt.json")
print('done')
