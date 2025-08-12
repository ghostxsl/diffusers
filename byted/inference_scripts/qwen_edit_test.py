import os
from os.path import join, splitext, basename, exists
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline
from diffusers.data.utils import load_file
from diffusers.data.outer_vos_tools import load_or_download_image


positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": "，超清，4K，电影级构图", # for chinese prompt
}
negative_prompt = "Vague text, smaller font size, the details are vague and unclear, overexposure, low quality."

device = torch.device('cuda')
pipe = QwenImageEditPipeline.from_pretrained("/mnt/bn/ttcc-algo-bytenas/xsl/Qwen-Image-Edit", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472)
}
width, height = aspect_ratios["9:16"]

out_dir = "output_qwen_edit"
os.makedirs(out_dir, exist_ok=True)

row_dict = load_file("/mnt/bn/ttcc-algo-bytenas/xsl/data/gen_ai_dataset/test.json")

for line in tqdm(row_dict):
    name = basename(line['poster_url']) + '_qwen_lora.jpg'

    product_img = load_or_download_image(line['product_img'])
    product_img = np.array(product_img)
    product_img = np.pad(product_img, ((249, 249), (0,0), (0,0)), constant_values=255)
    product_img = Image.fromarray(product_img).resize((width, height), 1)

    gt_img = load_or_download_image(line['poster_img'])
    gt_img = gt_img.resize((width, height), 1)

    prompt = line['prompt']
    out_img = pipe(
        image=product_img,
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        ).images[0]

    out_img = np.concatenate([product_img, gt_img, out_img], axis=1)
    Image.fromarray(out_img).save(join(out_dir, name))

print('done')
