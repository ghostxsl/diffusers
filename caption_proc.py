# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, splitext
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import csv
import pandas
import torch
from lavis.models import load_model_and_preprocess


def load_image(image):
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for `image`. "
            "Should be a local path to image or a PIL image."
        )
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA":
        # returning an RGB mode image with no transparency
        img = np.array(image)[..., :3]
        image = Image.fromarray(img)
    elif image.mode != "RGB":
        image = image.convert("RGB")

    return image


device = torch.device("cuda")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
)

img_dir = "/xsl/wilson.xu/dataset/tmp_imgs/pro_img"
img_list = sorted(os.listdir(img_dir))

caption_list = pandas.read_csv("/xsl/wilson.xu/dataset/new_mote_caption.csv").values.tolist()
caption_dict = {a[0]: a[1] for a in caption_list}


out_file = "/xsl/wilson.xu/dataset/mote_caption_split1.csv"
f = open(out_file, "a", encoding="utf-8", newline="")
writer = csv.writer(f)
writer.writerow(['file_name', 'text'])

count = 0
for name in tqdm(img_list):
    if caption_dict.get(name, None) is None:
        img = load_image(join(img_dir, name))
        image = vis_processors["eval"](img).unsqueeze(0).to(device)
        caption = model.generate({"image": image})
        writer.writerow([name, caption[0]])
        f.flush()
    else:
        count += 1

f.close()
print('Done!')
print(count)
