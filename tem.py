import os
from os.path import split, splitext, join, basename, exists
from tqdm import tqdm
import copy
import numpy as np
import torch
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import shutil
import random
import pandas
import requests
from diffusers.data.utils import *
from diffusers.data.vos_client import VOSClient
from diffusers.utils.vip_utils import load_image


# data_list = load_file("/Users/wilson.xu/Downloads/20240612_jimmy02.jiang.csv")
#
# out_list = []
# for line in tqdm(data_list):
#     if line[5] in ['运动裙', '连衣裙', '旗袍', '睡裙']:
#         if int(line[-4]) <= 5:
#             img_url = line[-3]
#             if img_url.endswith('_420_531.jpg'):
#                 img_url = img_url.replace('_420_531.jpg', '.jpg')
#             out_list.append(img_url)
#
# pkl_save(out_list, "/Users/wilson.xu/Downloads/dress_0613_jianbin.list")



HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
}


def down_one(url):
    content = requests.get(url, timeout=2, headers=HEADERS).content
    image = Image.open(BytesIO(content))
    # image = np.frombuffer(resp, dtype="uint8")
    # image = cv2.imdecode(image, 1)
    return image


font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 80, encoding='utf-8')

data_list = pandas.read_excel("/Users/wilson.xu/Downloads/test_0617.xlsx").values.tolist()
num_right = 0
num_right_1 = 0
num_right_2 = 0
num_other = 0

out_dir = "/Users/wilson.xu/Downloads/test_0617_res"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(join(out_dir, "可用"), exist_ok=True)
os.makedirs(join(out_dir, "背景瑕疵"), exist_ok=True)
os.makedirs(join(out_dir, "不可用"), exist_ok=True)

num_len = len(data_list)
for line in tqdm(data_list):
    name = line[0]
    img = down_one(name)
    text1 = line[1].replace("生成图1", "")
    text2 = line[2].replace("生成图2", "")
    draw = ImageDraw.Draw(img)
    draw.text(
        (1024, 0),
        text1,
        (255, 0, 0),
        font=font
    )
    draw.text(
        (2048, 0),
        text2,
        (255, 0, 0),
        font=font
    )


    if text1 == '可用':
        num_right_1 += 1
    elif text2 == '可用':
        num_right_2 += 1

    if text1 == '可用' or text2 == '可用':
        num_right += 1
        img.save(join(out_dir, "可用", basename(name)))
    else:
        if text1 == '背景瑕疵' or text2 == '背景瑕疵':
            num_other += 1
            img.save(join(out_dir, "背景瑕疵", basename(name)))
        else:
            img.save(join(out_dir, "不可用", basename(name)))


print(num_right / num_len)
print(num_right_1 / num_len)
print(num_right_2 / num_len)
print(num_other / num_len)
print('done')
