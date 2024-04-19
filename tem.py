import os
from os.path import split, splitext, join, basename, exists
from tqdm import tqdm
import random
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import pandas
import csv
from diffusers.data.utils import *


HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
}


def download_pil_image(url):
    content = requests.get(url, timeout=3, headers=HEADERS).content
    image = Image.open(BytesIO(content))
    return image


root_dir = "/Users/wilson.xu/Downloads/xhs_infer_pt_redraw_label"
input_file = "xhs4_0625-3_58125_0.57.xlsx"
output_file = "xhs_4_label.list"
data_list = pandas.read_excel(join(root_dir, input_file)).values.tolist()

out_list = []
if exists(join(root_dir, output_file)):
    out_list = load_file(join(root_dir, output_file))

for line in tqdm(data_list):
    out_list.append([line[5], line[6]])
pkl_save(out_list, join(root_dir, output_file))

print('done')
