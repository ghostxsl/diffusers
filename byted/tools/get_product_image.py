# Copyright (c) wilson.xu. All rights reserved.
import os
from tqdm import tqdm
import numpy as np
from PIL import Image

from diffusers.data.tos import save_tos
from diffusers.data.outer_vos_tools import load_or_download_image, encode_pil_bytes
from diffusers.data.utils import get_bbox_from_mask
from diffusers.data.utils import load_csv_or_xlsx_to_dict, csv_save


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


def pad_image_to_square(img, pad_values=255):
    h, w = img.shape[:2]
    if w > h:
        pad_ = w - h
        img = np.pad(
            img,
            ((pad_ // 2, pad_ - pad_ // 2), (0, 0), (0, 0)),
            constant_values=pad_values
        )
    elif h > w:
        pad_ = h - w
        img = np.pad(
            img,
            ((0, 0), (pad_ // 2, pad_ - pad_ // 2), (0, 0)),
            constant_values=pad_values
        )
    return Image.fromarray(img)


def crop_image(image, bbox, out_size=640, padding=64):
    x1, y1, x2, y2 = bbox
    item_image = image[y1: y2, x1: x2]
    item_image = pad_image_to_square(item_image)
    size = out_size - 2 * padding
    item_image = item_image.resize((size, size), 1)
    item_image = pad_image(item_image, padding)

    return item_image


input_file = "out1.csv"
data_list = load_csv_or_xlsx_to_dict(input_file)

for line in tqdm(data_list):
    image = load_or_download_image(line['image_url'])
    mask = load_or_download_image(line['src_mask_url'])

    mask = np.array(mask.convert('L'))
    bbox = get_bbox_from_mask(mask)

    mask = np.float32(mask[..., None]) / 255
    product_image = mask * np.array(image, dtype=np.float32) + (1 - mask) * 255
    product_image = np.uint8(np.clip(product_image, 0, 255))

    product_image = crop_image(product_image, bbox)

    product_url = save_tos(
        encode_pil_bytes(product_image, to_string=False),
        folder_name="xsl"
    )
    line['product_url'] = product_url

csv_save(data_list, "out_data.csv")

print('done')
