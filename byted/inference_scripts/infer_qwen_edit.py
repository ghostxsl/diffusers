import os
from os.path import join, splitext, basename
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

from diffusers.models.byted.transformer_qwenimage import QwenImageTransformer2DModel
from diffusers.pipelines.byted.pipeline_qwenimage_edit import QwenImageEditPipeline
from diffusers.data.utils import load_file
from diffusers.data.outer_vos_tools import load_or_download_image

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": "，超清，4K，电影级构图", # for chinese prompt
}
negative_prompt = "Vague text, smaller font size, the details are vague and unclear, overexposure, low quality."

pretrained_model_name_or_path = "/mnt/bn/ttcc-algo-bytenas/xsl/Qwen-Image"
device = torch.device('cuda')
transformer = QwenImageTransformer2DModel.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
pipe = QwenImageEditPipeline.from_pretrained(
    pretrained_model_name_or_path,
    transformer=transformer,
    torch_dtype=torch.bfloat16
)
pipe = pipe.to(device)

pipe.load_lora_weights("/mnt/bn/ttcc-algo-bytenas/qwen-lora-xsl/checkpoint-5000/pytorch_lora_weights.safetensors")
pipe.fuse_lora(lora_scale=1.0)
pipe.unload_lora_weights()

out_dir = "output_qwen_lora"
os.makedirs(out_dir, exist_ok=True)

row_dict = load_file("/mnt/bn/ttcc-algo-bytenas/xsl/data/gen_ai_dataset/test.json")

for line in tqdm(row_dict):
    product_img = load_or_download_image(line['src_url'])
    gt_img = load_or_download_image(line['poster_url'])
    prompt = line['prompt']
    out_img = pipe(
        image_reference=product_img,
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=720,
        height=1280,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]
    out_img = np.concatenate([out_img, gt_img], axis=1)
    name = basename(line['poster_url']) + '_qwen_lora.jpg'
    Image.fromarray(out_img).save(join(out_dir, name))

print('done')
