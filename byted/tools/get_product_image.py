# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, basename
import time
from tqdm import tqdm
import numpy as np
from PIL import Image

from diffusers.data.byted.tos import save_tos
from diffusers.data.outer_vos_tools import load_or_download_image, encode_pil_bytes
from diffusers.data.utils import load_csv_or_xlsx_to_dict, xlsx_save, get_bbox_from_mask


def pad_image(image, padding, fill_color=(255, 255, 255)):
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


def crop_image(image, bbox, padding=10):
    x1, y1, x2, y2 = bbox
    item_image = image[y1: y2, x1: x2]
    item_image = Image.fromarray(item_image)
    item_image = pad_image(item_image, padding)

    return item_image


input_file = "/mnt/bn/creative-algo/xsl/temp/gpt_0824_v1.xlsx"
data_list = load_csv_or_xlsx_to_dict(input_file)

save_image_dir = "/mnt/bn/creative-algo/xsl/data/gpt_dataset/images"
os.makedirs(save_image_dir, exist_ok=True)
for line in tqdm(data_list):
    name = basename(line['gen_url'])

    poster_image = load_or_download_image(line['gen_url'])
    mask = load_or_download_image(line['gen_mask_url'])

    mask = np.array(mask.convert('L'))
    bbox = get_bbox_from_mask(mask)

    mask = np.float32(mask[..., None]) / 255
    product_image = mask * np.array(poster_image, dtype=np.float32) + (1 - mask) * 255
    product_image = np.uint8(np.clip(product_image, 0, 255))

    product_image = crop_image(product_image, bbox)

    for i in range(3):
        product_url = save_tos(
            encode_pil_bytes(product_image, to_string=False),
            object_name=f"xsl/{name}_product.jpg",
        )
        if product_url is None:
            time.sleep(1)
            continue
        else:
            line['product_url'] = product_url
            break

    poster_image.save(join(save_image_dir, name + '_poster.jpg'))
    product_image.save(join(save_image_dir, name + '_product.jpg'))

xlsx_save(data_list, "gpt_0824_v2.xlsx")

print('done')
