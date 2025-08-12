# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, splitext, basename, exists
from tqdm import tqdm
import torch
from PIL import Image

from diffusers.pipelines.flux.pipeline_flux_fill import FluxFillPipeline
from diffusers.data.utils import load_file, load_csv_or_xlsx_to_dict
from diffusers.data.outer_vos_tools import load_or_download_image


def pad_image(image, padding, fill_color=(0, 0, 0)):
    """
    为图像添加填充

    参数:
        image: PIL Image对象
        padding: 填充大小，可以是一个整数(四周填充相同大小)或元组(左,右,上,下)
        fill_color: 填充区域的颜色，默认为白色(255,255,255)

    返回:
        填充后的PIL Image对象
    """
    # 处理填充参数
    if isinstance(padding, int):
        left = right = top = bottom = padding
    else:
        left, right, top, bottom = padding

    # 获取原图尺寸
    width, height = image.size

    # 计算新图像尺寸
    new_width = width + left + right
    new_height = height + top + bottom

    # 创建新图像，使用指定颜色填充
    new_image = Image.new(image.mode, (new_width, new_height), fill_color)

    # 将原图粘贴到新图像的指定位置
    new_image.paste(image, (left, top))

    return new_image


device = torch.device('cuda')
pipe = FluxFillPipeline.from_pretrained("/mnt/bn/creative-algo/xsl/models/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

out_dir = "output_flux_fill"
os.makedirs(out_dir, exist_ok=True)
row_dict = load_csv_or_xlsx_to_dict("/mnt/bn/creative-algo/xsl/results_all_done_4567_v2.xlsx")

width, height = 928, 1664
for line in tqdm(row_dict):
    img = load_or_download_image(line['gen_url'])
    w, h = img.size
    oh = int(h * width / w)
    img = img.resize((width, oh), 1)
    pad_ = (height - oh) // 2
    img = pad_image(img, (0, 0, pad_, pad_), fill_color=(0, 0, 0))

    mask_image = Image.new("L", (width, oh), 0)
    mask_image = pad_image(mask_image, (0, 0, pad_, pad_), fill_color=255)

    prompt = line['gen_bg_prompt']
    out_img = pipe(
        prompt=prompt,
        image=img,
        mask_image=mask_image,
        width=width,
        height=height,
        num_inference_steps=40,
        guidance_scale=30,
        ).images[0]

    name = basename(line['gen_url']) + '_poster.jpg'
    out_img.save(join(out_dir, name))

print('done')
