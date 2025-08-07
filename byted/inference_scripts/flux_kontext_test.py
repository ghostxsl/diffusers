import os
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
from diffusers.data.utils import load_file


device = torch.device('cuda')
pipe = FluxKontextPipeline.from_pretrained("/mnt/bn/ttcc-algo-bytenas/zjn/models/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

pipe.load_lora_weights("/mnt/bn/ttcc-algo-bytenas/xsl/flux-kontext-lora-xsl/checkpoint-16000/pytorch_lora_weights.safetensors")
pipe.fuse_lora(lora_scale=1.0)
pipe.unload_lora_weights()

out_dir = "output_flux_kontext_lora"
os.makedirs(out_dir, exist_ok=True)
row_dict = load_file("/mnt/bn/ttcc-algo-bytenas/zjn/data/poster_train_data_1/infos_gpt.json")

for k, line in tqdm(row_dict.items()):
    img_with_bg_url = line['bg']
    prompt = line['gpt_caption']
    size = line['canvas_size']
    img = load_image(img_with_bg_url)
    image = pipe(
        image=img,
        prompt=prompt,
        guidance_scale=2.5,
        height=size[1],
        width=size[0],
        num_inference_steps=50,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]
    img = load_image(line['poster']).resize(image.size, 1)
    out_img = np.concatenate([img, image], axis=1)
    name = k + '_kontext_lora.jpg'
    Image.fromarray(out_img).save(f"{out_dir}/{name}")

print('done')
