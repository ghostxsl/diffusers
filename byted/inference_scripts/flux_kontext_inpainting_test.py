import os
from os.path import splitext, basename
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import pandas as pd

from diffusers.pipelines.flux.pipeline_flux_kontext_inpaint import FluxKontextInpaintPipeline
from diffusers.data.outer_vos_tools import download_pil_image
from diffusers.utils.vip_utils import load_image


def read_image(img):
    if img.startswith('http'):
        img = download_pil_image(img)
    return load_image(img)


device = torch.device('cuda')
pipe = FluxKontextInpaintPipeline.from_pretrained("/mnt/bn/ttcc-algo-bytenas/zjn/models/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

out_dir = "output_flux_kontext_inpainting"
os.makedirs(out_dir, exist_ok=True)
df = pd.read_excel("/mnt/bn/ttcc-algo-bytenas/xsl/out1.xlsx")
row_dict = df.to_dict('records')

for line in tqdm(row_dict):
    img = read_image(line['bg_res_url'])
    mask = read_image(line['bg_res_mask_url'])
    mask = Image.fromarray(255 - np.array(mask))
    prompt = line['flux_kontext_prompt']
    image = pipe(
        image=img,
        mask_image=mask,
        prompt=prompt,
        strength=1.0,
        guidance_scale=2.5,
        height=1280,
        width=720,
        num_inference_steps=50,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]
    img = read_image(line['bg_res_url']).resize(image.size, 1)
    out_img = np.concatenate([img, image], axis=1)
    name = splitext(basename(line['bg_res_url']))[0] + '_kontext_inpainting.jpg'
    Image.fromarray(out_img).save(f"{out_dir}/{name}")

print('done')
