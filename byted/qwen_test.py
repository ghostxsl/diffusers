
import os
from tqdm import tqdm
import torch
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from diffusers.data.utils import load_file
from diffusers.utils import load_image


csv_list = load_file("/mlx_devbox/users/xushangliang/playground/creative_image_core_solution/whx_workspace/carousel/v7_folder/v7_results.csv")

os.makedirs("qwenimage", exist_ok=True)


device = torch.device('cuda')
pipe = QwenImagePipeline.from_pretrained("/mnt/bn/ttcc-algo-bytenas/xsl/Qwen-Image", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": "超清，4K，电影级构图", # for chinese prompt
}

prompt = 'Bookstore window display. A sign displays "New Arrivals This Week". Below, a shelf tag with the text "Best-Selling Novels Here". To the side, a colorful poster advertises "Author Meet And Greet on Saturday" with a central portrait of the author. There are four books on the bookshelf, namely "The light between worlds" "When stars are scattered" "The slient patient" "The night circus"'
negative_prompt = "Vague text, small font size"
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472)
}
width, height = aspect_ratios["9:16"]

for i, line in enumerate(tqdm(csv_list)):
    if line[0] == "Apparel & Accessories":
        prompt = line[-1]
        image = pipe(
            prompt=prompt + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            ).images[0]
        image.save(f"qwenimage/{i}.png")

print('done')
