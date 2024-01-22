import os
from os.path import join, split
from tqdm import tqdm
from diffusers.data.vos_uploader import VOSUploader
from diffusers.data import json_load

vos_obj = VOSUploader()

pidm = json_load("/home/llms/s1_pidm.json")

save_dir = "/home/llms/s1_pidm"

for k, v in tqdm(pidm.items()):
    os.makedirs(join(save_dir, k), exist_ok=True)

    for name in v:
        img = vos_obj.display_vos(name)
        img.save(join(save_dir, k, split(name)[1]))

print("Done!")
