import os
from os.path import join, splitext, basename
from tqdm import tqdm
import torch

from diffusers.pipelines.byted.pipeline_qwenimage_t2i_inpaint import QwenImageInpaintPipeline
from diffusers.data.outer_vos_tools import load_or_download_image
from diffusers.data.utils import load_csv_or_xlsx_to_dict


positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": "，超清，4K，电影级构图", # for chinese prompt
}
negative_prompt = "Vague text, smaller font size, the details are vague and unclear, overexposure, low quality."

device = torch.device('cuda')
pipe = QwenImageInpaintPipeline.from_pretrained("/mnt/bn/ttcc-algo-bytenas/xsl/Qwen-Image", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472)
}
width, height = aspect_ratios["9:16"]

out_dir = "output_qwen_inpaint"
os.makedirs(out_dir, exist_ok=True)
row_dict = load_csv_or_xlsx_to_dict("/mnt/bn/ttcc-algo-bytenas/xsl/out1.xlsx")

for line in tqdm(row_dict):
    img = load_or_download_image(line['bg_res_url'])
    mask = load_or_download_image(line['bg_res_mask_url'])
    prompt = line['flux_kontext_prompt']
    out_img = pipe(
        image=img,
        mask_image=mask,
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        mask_blur=2,
        invert_mask=True,
        blend_src_image=True,
        ).images[0]
    name = splitext(basename(line['bg_res_url']))[0] + '_qwen_inpaint.jpg'
    out_img.save(join(out_dir, name))

print('done')
