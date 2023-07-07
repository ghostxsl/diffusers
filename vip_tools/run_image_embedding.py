# Copyright (c) wilson.xu. All rights reserved.
import os
import sys
from os.path import join, split, splitext
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.data.utils import *


device = torch.device("cuda")
dtype = torch.float16

root_dir = "/xsl/wilson.xu/dataset/train_data"
img_dir = join(root_dir, sys.argv[1])
out_dir = join(root_dir, "embeddings_" + sys.argv[1][-1])
os.makedirs(out_dir, exist_ok=True)
image_encoder_model_path = "/xsl/wilson.xu/weights/IP-Adapter/image_encoder"

feature_extractor = CLIPImageProcessor()
image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_model_path)
image_encoder.requires_grad_(False)
image_encoder.to(device, dtype=dtype)


for name in tqdm(os.listdir(img_dir)):
    img = Image.open(join(img_dir, name))
    img = feature_extractor(img, return_tensors="pt").pixel_values
    img = img.to(device=device, dtype=dtype)
    with torch.no_grad():
        image_embeds = image_encoder(img).image_embeds

    image_embeds = image_embeds.cpu().numpy()
    np.save(join(out_dir, splitext(name)[0]), image_embeds)

print("Done!")
