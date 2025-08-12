# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, splitext, basename
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

from diffusers.pipelines.byted.pipeline_qwenimage_edit import QwenImageEditPipeline
from diffusers.data.utils import load_file
from diffusers.data.outer_vos_tools import load_or_download_image


aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472)
}
width, height = aspect_ratios["9:16"]

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": "，超清，4K，电影级构图", # for chinese prompt
}
negative_prompt = "Vague text, smaller font size, the details are vague and unclear, overexposure, low quality."

pretrained_model_name_or_path = "/mnt/bn/creative-algo/xsl/models/Qwen-Image-Edit"
device = torch.device('cuda')
pipe = QwenImageEditPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

pipe.load_lora_weights("/mnt/bn/creative-algo/xsl/exp_weights/xsl-lora/pytorch_lora_weights.safetensors")
pipe.fuse_lora(lora_scale=1.0)
pipe.unload_lora_weights()

out_dir = "gen_lora_test"
os.makedirs(out_dir, exist_ok=True)

row_dict = load_file("/mnt/bn/creative-algo/xsl/data/gen_ai_dataset/test_rewrite.json")

for line in tqdm(row_dict):
    product_img = load_or_download_image(line['product_img'])
    prompt = line['prompt']
    out_img = pipe(
        image_reference=product_img,
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cuda").manual_seed(42),
        prompt_image_size=(280, 280),
        reference_image_size=(768, 768),
    ).images[0]

    name = basename(line['poster_url']) + '_lora.jpg'
    out_img.save(join(out_dir, name))

print('done')
