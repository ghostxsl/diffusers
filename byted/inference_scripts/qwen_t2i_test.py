import os
from os.path import join, splitext, basename, exists
from tqdm import tqdm
import torch
import pandas as pd

from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from diffusers.data.utils import load_file


positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": "，超清，4K，电影级构图", # for chinese prompt
}
negative_prompt = "Vague text, smaller font size, the details are vague and unclear, overexposure, low quality."

device = torch.device('cuda')
pipe = QwenImagePipeline.from_pretrained("/mnt/bn/ttcc-algo-bytenas/xsl/Qwen-Image", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472)
}
width, height = aspect_ratios["9:16"]

out_dir = "output_qwen"
os.makedirs(out_dir, exist_ok=True)
# df = pd.read_excel("/mnt/bn/ttcc-algo-bytenas/xsl/out1.xlsx")
# row_dict = df.to_dict('records')
row_dict = load_file("crello_gpt_caption.json")

for k, v in tqdm(row_dict.items()):
    name = splitext(basename(k))[0] + '_qwen_t2i.jpg'
    if exists(join(out_dir, name)):
        continue

    prompt = v
    out_img = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        ).images[0]

    out_img.save(join(out_dir, name))

print('done')
