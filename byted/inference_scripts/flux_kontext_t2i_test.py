import os
from os.path import join, splitext, basename
from tqdm import tqdm
import torch
import pandas as pd

from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline


device = torch.device('cuda')
pipe = FluxKontextPipeline.from_pretrained("/mnt/bn/ttcc-algo-bytenas/zjn/models/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

out_dir = "output_flux_kontext_t2i"
os.makedirs(out_dir, exist_ok=True)
df = pd.read_excel("/mnt/bn/ttcc-algo-bytenas/xsl/out1.xlsx")
row_dict = df.to_dict('records')

for i, line in enumerate(tqdm(row_dict)):
    img_with_bg_url = line['bg_res_url']
    prompt = line['flux_kontext_prompt']
    out_img = pipe(
        prompt=prompt,
        guidance_scale=2.5,
        height=1280,
        width=720,
        num_inference_steps=50,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]
    name = splitext(basename(img_with_bg_url))[0] + '_kontext_t2i.jpg'
    out_img.save(join(out_dir, name))

print('done')
