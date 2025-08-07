import os
from os.path import join, splitext, basename
from tqdm import tqdm
import torch
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
import pandas as pd


positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": "超清，4K，电影级构图", # for chinese prompt
}

device = torch.device('cuda')
pipe = QwenImagePipeline.from_pretrained("/mnt/bn/ttcc-algo-bytenas/xsl/Qwen-Image", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

negative_prompt = "Vague text, small font size"
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472)
}
width, height = aspect_ratios["9:16"]

os.makedirs("output_qwen", exist_ok=True)
df = pd.read_excel("/mnt/bn/ttcc-algo-bytenas/xsl/out1.xlsx")
row_dict = df.to_dict('records')

for i, line in tqdm(enumerate(row_dict)):
    img_with_bg_url = line['bg_res_url']
    prompt = line['flux_kontext_prompt']
    out_img = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cuda").manual_seed(42),
        ).images[0]
    name = splitext(basename(img_with_bg_url))[0] + '_qwen_t2i.jpg'
    out_img.save(join('output_qwen', name))

print('done')
