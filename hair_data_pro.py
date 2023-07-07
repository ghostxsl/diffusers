import os
from os.path import join, splitext, basename
import cv2
from PIL import Image
import numpy as np
import torch

from aistudio.data.utils import dilate_mask, get_bbox_from_mask, pad_bbox_to_square, pad_bbox
from aistudio.utils import (
    parse_path,
    load_config,
    build_extensions,
)


def get_head_bbox(label_parsing, use_head=True):
    H, W = label_parsing.shape[:2]
    # 1. 计算人脸bbox
    head_mask = np.zeros_like(label_parsing)
    head_mask[label_parsing == 13] = 255
    # head_mask = dilate_mask(head_mask, dilate_factor=5)
    face_bbox = get_bbox_from_mask(head_mask)
    if face_bbox is None:
        return None
    x1, y1, x2, y2 = face_bbox
    h, w = y2 - y1, x2 - x1
    max_hw = max(h, w)
    # 按比例padding人脸bbox
    pad_top = int(0.7 * max_hw)
    y1 = 0 if y1 - pad_top < 0 else y1 - pad_top
    pad_bottom = int(1.3 * max_hw)
    y2 = H if y2 + pad_bottom > H else y2 + pad_bottom
    head_bbox = pad_bbox_to_square([x1, y1, x2, y2], [H, W])

    if use_head:
        # 2. 计算人脸+头发bbox
        head_mask[label_parsing == 1] = 255
        head_mask[label_parsing == 2] = 255
        head_mask[label_parsing == 4] = 255
        # 开运算去噪点
        kernel = np.ones([5, 5], np.uint8)
        head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_OPEN, kernel)
        head_mask = dilate_mask(head_mask, dilate_factor=5)
        head_bbox2 = get_bbox_from_mask(head_mask)
        if head_bbox2 is None:
            return None
        head_bbox2 = pad_bbox_to_square(head_bbox2, [H, W])
        # 与人脸bbox取并集
        x1 = min(head_bbox[0], head_bbox2[0])
        y1 = min(head_bbox[1], head_bbox2[1])
        x2 = max(head_bbox[2], head_bbox2[2])
        y2 = max(head_bbox[3], head_bbox2[3])
        head_bbox = pad_bbox_to_square([x1, y1, x2, y2], [H, W])
    return head_bbox


config = load_config("experiments/exp_change_face.yaml")
extensions = build_extensions(
            config['extensions'],
            weight_dir=parse_path("weights"),
            device=torch.device("cuda"))

img_dir = '/xsl/wilson.xu/dataset/hair_man_data'
out_dir = '/xsl/wilson.xu/dataset/hair_man_processed'

os.makedirs(out_dir, exist_ok=True)

for root, dirs, files in os.walk(img_dir):
    for name in files:
        if splitext(name)[-1].lower() in ['.jpg', '.jpeg', '.png']:
            img = Image.open(join(root, name))
            label_parsing = extensions['HumanParsing'](img)
            hat_mask = np.zeros_like(label_parsing)
            hat_mask[label_parsing==1] = 1
            if np.sum(hat_mask) > 10:
                print(f"with hat: {name}")
                continue
            head_bbox = get_head_bbox(label_parsing)
            crop_img = img.crop(head_bbox)
            crop_img = crop_img.resize([512, 512], 1)

            os.makedirs(join(out_dir, root.split('/')[-1]), exist_ok=True)
            crop_img.save(join(out_dir, root.split('/')[-1], splitext(name)[0] + '.png'))
            print(f"Save {join(out_dir, root.split('/')[-1], splitext(name)[0] + '.png')}")


print("Done!")
