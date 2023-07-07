# Copyright (c) wilson.xu. All rights reserved.
from lavis.models import load_model_and_preprocess
from PIL import Image, ImageOps
import torch
import numpy as np
import os
from os.path import join, splitext
import csv
from tqdm import tqdm


def load_image(image_path):
    if isinstance(image_path, str):
        image = Image.open(image_path)
    elif isinstance(image_path, Image.Image):
        image = image_path
    else:
        raise ValueError(
            "Incorrect format used for image. Should be a local path to an image, or a PIL image."
        )
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA":
        # returning an RGB mode image with no transparency
        image = Image.fromarray(np.array(image)[..., :3])
    elif image.mode != "RGB":
        image = image.convert("RGB")

    return image


img_dir = "/xsl/wilson.xu/dataset/train_data/train_3"
out_file = "/xsl/wilson.xu/dataset/train_data/train_3.csv"
device = torch.device("cuda")

model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
)

f = open(out_file, "a", encoding="utf-8", newline="")
writer = csv.writer(f)
writer.writerow(['file_name', 'text'])

num_no_caption = 0
img_list = sorted(os.listdir(img_dir))
for name in tqdm(img_list):
    try:
        img = load_image(join(img_dir, name))
        image = vis_processors["eval"](img).unsqueeze(0).to(device)
        caption = model.generate({"image": image})
        if caption[0].strip() == '':
            num_no_caption += 1
            print(f"No caption: {name}/{num_no_caption}")
        else:
            writer.writerow([name, caption[0]])
            f.flush()
    except:
        print(f"Wrong type: {name}")
        num_no_caption += 1

f.close()
print("Done")
