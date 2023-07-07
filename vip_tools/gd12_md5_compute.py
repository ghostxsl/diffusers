# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, splitext, exists, basename
import concurrent
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import hashlib
import numpy as np

from diffusers.data.vos_client import VOSClient
from diffusers.data.utils import load_file, pkl_save


num_workers = 1
save_path = "gd12_md5.dict"
info_dict = load_file("/home/llms/0321-1w.dict")
vos = VOSClient('xhs')

url_list = []
for k, v in info_dict.items():
    for url in v:
        url_list.append([k, url])


def single_process(key, url):
    url = url.replace('http://gd12-inner-storegw.api.vip.com/xhs-raw-data/', '')
    img = vos.download_vos_pil(url)
    content = np.array(img).tobytes()
    file_hash = hashlib.md5(content).hexdigest()
    return key, file_hash, url


md5_info = {}
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    future_list = [executor.submit(single_process, k, url) for k, url in url_list]
    with tqdm(total=len(url_list)) as pbar:
        for future in concurrent.futures.as_completed(future_list):
            key, file_hash, url = future.result()
            pbar.update(1)  # Update progress bar

            if file_hash not in md5_info:
                md5_info[file_hash] = [[url, key]]
            else:
                md5_info[file_hash].append([url, key])

            if len(md5_info) > 0 and len(md5_info) % 50 == 0:
                pkl_save(md5_info, save_path)

pkl_save(md5_info, save_path)
print('done')
