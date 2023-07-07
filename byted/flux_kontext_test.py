
import os
from tqdm import tqdm
import torch
from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
from diffusers.data.utils import load_file
from diffusers.utils import load_image


os.makedirs("flux_kontext", exist_ok=True)
csv_list = load_file("/mlx_devbox/users/xushangliang/playground/creative_image_core_solution/whx_workspace/carousel/v7_folder/v7_results.csv")

device = torch.device('cuda')
pipe = FluxKontextPipeline.from_pretrained("/mnt/bn/ttcc-algo-bytenas/zjn/models/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

for i, line in enumerate(tqdm(csv_list)):
    if line[0] == "Apparel & Accessories":
        img = load_image(line[21])
        image = pipe(
            image=img,
            prompt=line[-1],
            guidance_scale=3.5,
            height=16,
            width=9,
            num_inference_steps=50,
        ).images[0]
        image.save(f"flux_kontext/{i}.jpg")

print('done')
