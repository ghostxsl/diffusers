# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import join, exists, split, splitext
import re
import hashlib
import cv2
import math
import torch
import json
import pickle
import pandas
import regex
import string
import matplotlib
import numpy as np
from PIL import Image


__all__ = [
    't2i_collate_fn', 'controlnet_collate_fn', 'animate_collate_fn',
    'i2i_collate_fn', 'kolors_collate_fn', 'flux_collate_fn',
    'flux_fill_collate_fn', 'flux_kontext_collate_fn',
    'pkl_save', 'pkl_load', 'json_save', 'json_load', 'load_file',
    'csv_save', 'xlsx_save', 'load_csv_or_xlsx_to_dict',
    'draw_bodypose', 'draw_handpose', 'draw_facepose',
    'get_file_md5', 'get_str_md5', 'crop_human_bbox',
    'compute_OKS',
    'get_bbox_from_mask',
]


def t2i_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.cat([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


def controlnet_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat([example["input_ids"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }


def animate_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    reference_pixel_values = torch.stack([example["reference_pixel_values"] for example in examples])
    reference_pixel_values = reference_pixel_values.to(memory_format=torch.contiguous_format).float()

    if 'input_ids' in examples[0]:
        input_ids = torch.cat([example["input_ids"] for example in examples])
    else:
        input_ids = torch.zeros([0])

    if 'reference_image' in examples[0]:
        reference_image = torch.cat([example["reference_image"] for example in examples])
    else:
        reference_image = torch.zeros([0])

    if 'uncond' in examples[0]:
        uncond = torch.stack([example["uncond"] for example in examples])
        uncond = uncond.to(memory_format=torch.contiguous_format).float()
    else:
        uncond = torch.zeros([0])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "reference_pixel_values": reference_pixel_values,
        "input_ids": input_ids,
        "reference_image": reference_image,
        "uncond": uncond,
    }


def i2i_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    if 'input_ids' in examples[0]:
        input_ids = torch.cat([example["input_ids"] for example in examples])
    else:
        input_ids = torch.zeros([0])

    if 'reference_image' in examples[0]:
        reference_image = torch.cat([example["reference_image"] for example in examples])
        reference_image = reference_image.to(memory_format=torch.contiguous_format).float()
    else:
        reference_image = torch.zeros([0])

    if 'uncond' in examples[0]:
        uncond = torch.stack([example["uncond"] for example in examples])
        uncond = uncond.to(memory_format=torch.contiguous_format).float()
    else:
        uncond = torch.zeros([0])

    if 'conditioning_pixel_values' in examples[0]:
        conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
        conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        conditioning_pixel_values = torch.zeros([0])

    if 'class_label' in examples[0]:
        class_label = torch.cat([example["class_label"] for example in examples])
    else:
        class_label = torch.zeros([0])

    return {
        "pixel_values": pixel_values,
        "reference_image": reference_image,
        "input_ids": input_ids,
        "uncond": uncond,
        "conditioning_pixel_values": conditioning_pixel_values,
        "class_label": class_label,
    }


def kolors_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    if 'input_ids' in examples[0]:
        input_ids = torch.cat([example["input_ids"] for example in examples])
    else:
        input_ids = torch.zeros([0])

    if 'reference_image' in examples[0]:
        reference_image = torch.cat([example["reference_image"] for example in examples])
        reference_image = reference_image.to(memory_format=torch.contiguous_format).float()
    else:
        reference_image = torch.zeros([0])

    if 'uncond' in examples[0]:
        uncond = torch.stack([example["uncond"] for example in examples])
        uncond = uncond.to(memory_format=torch.contiguous_format).float()
    else:
        uncond = torch.zeros([0])

    if 'conditioning_pixel_values' in examples[0]:
        conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
        conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        conditioning_pixel_values = torch.zeros([0])

    if 'add_time_ids' in examples[0]:
        add_time_ids = torch.cat([example["add_time_ids"] for example in examples])
    else:
        add_time_ids = torch.zeros([0])

    return {
        "pixel_values": pixel_values,
        "reference_image": reference_image,
        "input_ids": input_ids,
        "uncond": uncond,
        "conditioning_pixel_values": conditioning_pixel_values,
        "add_time_ids": add_time_ids,
    }


def flux_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    if 'conditioning_pixel_values' in examples[0]:
        conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
        conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        conditioning_pixel_values = torch.zeros([0])

    if 'input_ids' in examples[0]:
        input_ids = torch.cat([example["input_ids"] for example in examples])
    else:
        input_ids = torch.zeros([0])

    if 'pooled_input_ids' in examples[0]:
        pooled_input_ids = torch.cat([example["pooled_input_ids"] for example in examples])
    else:
        pooled_input_ids = torch.zeros([0])

    if 'reference_image' in examples[0]:
        reference_image = torch.cat([example["reference_image"] for example in examples])
        reference_image = reference_image.to(memory_format=torch.contiguous_format).float()
    else:
        reference_image = torch.zeros([0])

    if 'uncond' in examples[0]:
        uncond = torch.stack([example["uncond"] for example in examples])
        uncond = uncond.to(memory_format=torch.contiguous_format).float()
    else:
        uncond = torch.zeros([0])

    if 'reference_pixel_values' in examples[0]:
        reference_pixel_values = torch.stack([example["reference_pixel_values"] for example in examples])
        reference_pixel_values = reference_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        reference_pixel_values = torch.zeros([0])

    if 'pooled_prompt_embeds' in examples[0]:
        pooled_prompt_embeds = torch.cat([example["pooled_prompt_embeds"] for example in examples])
    else:
        pooled_prompt_embeds = torch.zeros([0])

    if 'prompt_embeds' in examples[0]:
        prompt_embeds = torch.cat([example["prompt_embeds"] for example in examples])
    else:
        prompt_embeds = torch.zeros([0])

    if 'img_ids' in examples[0]:
        img_ids = torch.stack([example["img_ids"] for example in examples])
        img_ids = img_ids.to(memory_format=torch.contiguous_format).float()
    else:
        img_ids = torch.zeros([0])

    if 'txt_ids' in examples[0]:
        txt_ids = torch.stack([example["txt_ids"] for example in examples])
        txt_ids = txt_ids.to(memory_format=torch.contiguous_format).float()
    else:
        txt_ids = torch.zeros([0])

    if 'latents_mask' in examples[0]:
        latents_mask = torch.stack([example["latents_mask"] for example in examples])
        latents_mask = latents_mask.to(memory_format=torch.contiguous_format).float()
    else:
        latents_mask = torch.zeros([0])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
        "pooled_input_ids": pooled_input_ids,
        "reference_image": reference_image,
        "uncond": uncond,
        "reference_pixel_values": reference_pixel_values,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "prompt_embeds": prompt_embeds,
        "img_ids": img_ids,
        "txt_ids": txt_ids,
        "latents_mask": latents_mask,
    }


def flux_fill_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    mask = torch.stack([example["mask"] for example in examples])
    mask = mask.to(memory_format=torch.contiguous_format).float()

    latent_image_ids = torch.stack([example["latent_image_ids"] for example in examples])
    latent_image_ids = latent_image_ids.to(memory_format=torch.contiguous_format).float()

    if 'masked_image_pixel_values' in examples[0]:
        masked_image_pixel_values = torch.stack([example["masked_image_pixel_values"] for example in examples])
        masked_image_pixel_values = masked_image_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        masked_image_pixel_values = torch.zeros([0])

    if 'reference_pixel_values' in examples[0]:
        reference_pixel_values = torch.stack([example["reference_pixel_values"] for example in examples])
        reference_pixel_values = reference_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        reference_pixel_values = torch.zeros([0])

    if 'input_ids' in examples[0]:
        input_ids = torch.cat([example["input_ids"] for example in examples])
    else:
        input_ids = torch.zeros([0])

    if 'pooled_input_ids' in examples[0]:
        pooled_input_ids = torch.cat([example["pooled_input_ids"] for example in examples])
    else:
        pooled_input_ids = torch.zeros([0])

    return {
        "pixel_values": pixel_values,
        "masked_image_pixel_values": masked_image_pixel_values,
        "mask": mask,
        "latent_image_ids": latent_image_ids,
        "reference_pixel_values": reference_pixel_values,
        "input_ids": input_ids,
        "pooled_input_ids": pooled_input_ids,
    }


def flux_kontext_collate_fn(examples):
    image_latents = torch.stack([example["image_latents"] for example in examples])
    image_latents = image_latents.to(memory_format=torch.contiguous_format)

    cond_image_latents = torch.stack([example["cond_image_latents"] for example in examples])
    cond_image_latents = cond_image_latents.to(memory_format=torch.contiguous_format)

    if 'pooled_prompt_embeds' in examples[0]:
        pooled_prompt_embeds = torch.cat([example["pooled_prompt_embeds"] for example in examples])
    else:
        pooled_prompt_embeds = torch.zeros([0])

    if 'prompt_embeds' in examples[0]:
        prompt_embeds = torch.cat([example["prompt_embeds"] for example in examples])
    else:
        prompt_embeds = torch.zeros([0])

    if 'prompt_embeds_mask' in examples[0]:
        prompt_embeds_mask = torch.cat([example["prompt_embeds_mask"] for example in examples])
    else:
        prompt_embeds_mask = torch.zeros([0])

    if 'latent_image_ids' in examples[0]:
        latent_image_ids = torch.cat([example["latent_image_ids"] for example in examples])
        latent_image_ids = latent_image_ids.to(memory_format=torch.contiguous_format)
    else:
        latent_image_ids = torch.zeros([0])

    if 'cond_image_ids' in examples[0]:
        cond_image_ids = torch.cat([example["cond_image_ids"] for example in examples])
        cond_image_ids = cond_image_ids.to(memory_format=torch.contiguous_format)
    else:
        cond_image_ids = torch.zeros([0])

    return {
        "image_latents": image_latents,
        "cond_image_latents": cond_image_latents,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "prompt_embeds": prompt_embeds,
        "latent_image_ids": latent_image_ids,
        "cond_image_ids": cond_image_ids,
        "prompt_embeds_mask": prompt_embeds_mask,
    }


def pkl_save(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def pkl_load(file):
    with open(file, 'rb') as f:
        out = pickle.load(f)
    return out


def json_save(obj, file):
    with open(file, 'w') as f:
        json.dump(obj, f)


def json_load(file):
    with open(file, 'r') as f:
        out = json.load(f)
    return out


def load_file(file_path):
    assert exists(file_path), f"File {file_path} does not exist."

    if splitext(file_path)[1] == ".json":
        return json_load(file_path)
    elif splitext(file_path)[1] == ".csv":
        return pandas.read_csv(file_path).values.tolist()
    elif splitext(file_path)[1] == ".xlsx":
        return pandas.read_excel(file_path).values.tolist()
    else:
        return pkl_load(file_path)


def csv_save(obj, file, mode="w"):
    df = pandas.DataFrame(obj)
    header = True
    if mode == "a":
        header = not exists(file)
    df.to_csv(file, mode=mode, index=False, header=header, encoding='utf-8')


def xlsx_save(obj, file):
    df = pandas.DataFrame(obj)
    df.to_excel(file, index=False)


def load_csv_or_xlsx_to_dict(file_path):
    if file_path.endswith('.xlsx'):
        df = pandas.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pandas.read_csv(file_path, encoding='utf-8')
    else:
        raise Exception(f"Error `file_path` type:{file_path}")
    return df.to_dict('records')


def draw_bodypose(canvas, kpts, kpt_valid, stickwidth=4,
                  radius=4, alpha=0.6, show_number=False):
    """
    Args:
        canvas (ndarray): shape[H, W, 3]
        kpts (ndarray): shape[n, 18, 2]
        kpt_valid (ndarray: bool): shape[n, 18]
        stickwidth (int): Default 4.
        radius (int): Default 4.
        alpha (float): Default 0.6.

    Returns: `canvas`: shape[H, W, 3]
    """
    links = [[1, 2], [1, 5],
             [2, 3], [3, 4],
             [5, 6], [6, 7],
             [1, 8], [8, 9], [9, 10],
             [1, 11], [11, 12], [12, 13],
             [1, 0], [0, 14], [14, 16],
             [0, 15], [15, 17]]

    colors = [[255, 0, 0], [255, 85, 0],
              [255, 170, 0], [255, 255, 0], [170, 255, 0],
              [85, 255, 0], [0, 255, 0], [0, 255, 85],
              [0, 255, 170], [0, 255, 255], [0, 170, 255],
              [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    # draw links
    for link, color in zip(links, colors):
        for kpt, is_valid in zip(kpts, kpt_valid):
            if np.sum(is_valid[link]) == 2:
                kpt_XY = kpt[link]
                mean_XY = np.mean(kpt_XY, axis=0)
                diff_XY = kpt_XY[0] - kpt_XY[1]
                length = np.sqrt(np.square(diff_XY).sum())
                angle = math.degrees(math.atan2(diff_XY[1], diff_XY[0]))
                polygon = cv2.ellipse2Poly((int(mean_XY[0]), int(mean_XY[1])),
                                           (int(length / 2), stickwidth),
                                           int(angle), 0, 360, 1)
                cv2.fillConvexPoly(canvas, polygon, [int(float(c) * alpha) for c in color])

    # draw points
    for i, color in enumerate(colors):
        for kpt, is_valid in zip(kpts, kpt_valid):
            if is_valid[i]:
                cv2.circle(
                    canvas, (int(kpt[i, 0]), int(kpt[i, 1])),
                    radius, color, thickness=-1)
                if show_number:
                    cv2.putText(
                        canvas,
                        str(i), (int(kpt[i, 0]), int(kpt[i, 1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 0),
                        lineType=cv2.LINE_AA)
    return canvas


def draw_handpose(canvas, kpts, kpt_valid, radius=4, show_number=False):
    links = [[0, 1], [1, 2], [2, 3], [3, 4],
             [0, 5], [5, 6], [6, 7], [7, 8],
             [0, 9], [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15], [15, 16],
             [0, 17], [17, 18], [18, 19], [19, 20]]
    H, W, _ = canvas.shape
    kpts[..., 0] = np.clip(kpts[..., 0], 0, W - 1)
    kpts[..., 1] = np.clip(kpts[..., 1], 0, H - 1)
    for kpt, is_valid in zip(kpts, kpt_valid):
        kpt = kpt.astype(np.int32)
        # draw links
        for i, link in enumerate(links):
            if np.sum(is_valid[link]) == 2:
                x1, y1 = kpt[link[0]]
                x2, y2 = kpt[link[1]]
                cv2.line(
                    canvas, (x1, y1), (x2, y2),
                    matplotlib.colors.hsv_to_rgb(
                        [i / len(links), 1.0, 1.0]) * 255,
                    thickness=2)
        # draw points
        for i, (x, y) in enumerate(kpt):
            if is_valid[i]:
                cv2.circle(canvas, (x, y), radius, (0, 0, 255), thickness=-1)
                if show_number:
                    cv2.putText(
                        canvas,
                        str(i), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 0),
                        lineType=cv2.LINE_AA)
    return canvas


def draw_facepose(canvas, kpts, kpt_valid):
    H, W, _ = canvas.shape
    kpts[..., 0] = np.clip(kpts[..., 0], 0, W - 1)
    kpts[..., 1] = np.clip(kpts[..., 1], 0, H - 1)
    for face, face_valid in zip(kpts, kpt_valid):
        for kpt, is_valid in zip(face, face_valid):
            if is_valid:
                cv2.circle(
                    canvas, (int(kpt[0]), int(kpt[1])),
                    3, (255, 255, 255), thickness=-1)
    return canvas


def get_file_md5(file_path, nbytes=-1):
    assert exists(file_path)

    with open(file_path, 'rb') as f:
        content = f.read(nbytes) if nbytes > 0 else f.read()
        file_hash = hashlib.md5(content).hexdigest()

    return file_hash


def get_str_md5(string):
    return hashlib.md5(string.encode("utf-8")).hexdigest()


def crop_human_bbox(det_bbox, img_size, crop_size=(1024, 768), pad_bbox=5):
    x1, y1, x2, y2 = det_bbox
    h, w = img_size
    ch, cw = crop_size

    x1 = 0 if x1 - pad_bbox < 0 else x1 - pad_bbox
    y1 = 0 if y1 - pad_bbox < 0 else y1 - pad_bbox
    x2 = w if x2 + pad_bbox > w else x2 + pad_bbox
    y2 = h if y2 + pad_bbox > h else y2 + pad_bbox

    bh, bw = y2 - y1, x2 - x1
    ratio_h, ratio_w = ch / bh, cw / bw

    pad_ = [[0, 0], [0, 0], [0, 0]]
    # 长边resize
    if ratio_h < ratio_w:
        # 按高 resize
        ow = int(bh / ch * cw)
        expand_w = ow - bw

        x1 -= int(expand_w / 2)
        if x1 < 0:
            pad_[1][0] = abs(x1)
            x1 = 0

        x2 += (expand_w - int(expand_w / 2))
        if x2 > w:
            pad_[1][1] = x2 - w
            x2 = w

        if sum(pad_[1]) == 0:
            pad_ = None

        return [x1, y1, x2, y2], pad_
    elif ratio_h > ratio_w:
        # 按宽 resize
        oh = int(bw / cw * ch)
        expand_h = oh - bh

        y1 -= int(expand_h / 2)
        if y1 < 0:
            pad_[0][0] = abs(y1)
            y1 = 0

        y2 += (expand_h - int(expand_h / 2))
        if y2 > h:
            pad_[0][1] = y2 - h
            y2 = h

        if sum(pad_[0]) == 0:
            pad_ = None

        return [x1, y1, x2, y2], pad_
    else:
        return [x1, y1, x2, y2], None


def compute_OKS(gt_kpts, gt_bboxes, dt_kpts, kpt_thr=0.3, perfect_match=True):
    """
    Args:
        gt_kpts: shape [N, 18, 3]
        gt_bboxes: shape [N, 4]
        dt_kpts: shape [M, 18, 3]

    Returns: ious(N, M)
    """
    assert len(gt_kpts) == len(gt_bboxes)
    if len(gt_kpts) == 0 or len(dt_kpts) == 0:
        return None

    N, M = len(gt_kpts), len(dt_kpts)
    ious = np.zeros([N, M])
    sigmas = np.array([0.026, 0.079,
                       0.079, 0.072, 0.062,
                       0.079, 0.072, 0.062,
                       0.107, 0.087, 0.089,
                       0.107, 0.087, 0.089,
                       0.025, 0.025, 0.035, 0.035])
    vars = (sigmas * 2) ** 2

    def oks_kernel(gts, dts, bbox_area, gt_valid_ind):
        xg, yg, _ = np.split(gts[:, gt_valid_ind], 3, axis=-1)
        xd, yd, _ = np.split(dts[:, gt_valid_ind], 3, axis=-1)
        dx = xd - xg
        dy = yd - yg
        e = (dx[..., 0] ** 2 + dy[..., 0] ** 2) / vars[gt_valid_ind] / (bbox_area + np.spacing(1)) / 2
        out = np.sum(np.exp(-e), axis=-1) / len(gt_valid_ind)
        return out

    # 0. 计算检出的关键点
    gts_valid = gt_kpts[..., 2] > kpt_thr
    num_gts_valid = np.sum(gts_valid, axis=1)

    dts_valid = dt_kpts[..., 2] > kpt_thr
    num_dts_valid = np.sum(dts_valid, axis=1)

    for i in range(N):
        if num_gts_valid[i] == 0:
            continue

        gt_valid_ind = np.nonzero(gts_valid[i])[0]
        # 1. 计算gt与dt的similarity
        gt_bbox = gt_bboxes[i]
        bw, bh = gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]
        bbox_area = bw * bh
        ious[i] = oks_kernel(gt_kpts[i: i + 1], dt_kpts, bbox_area, gt_valid_ind)

        if perfect_match:
            # 2. 保留gt与dt完全匹配的关键点similarity, 降低不完全匹配的关键点similarity
            contain_dts_ind = np.nonzero(
                np.sum(dts_valid[:, gt_valid_ind], axis=1) == len(gt_valid_ind))[0]
            same_dts_ind = np.intersect1d(
                contain_dts_ind, np.nonzero(num_dts_valid == len(gt_valid_ind))[0])
            not_same_dts_ind = np.setdiff1d(np.arange(M), same_dts_ind)
            ious[i, not_same_dts_ind] -= 1.

    return ious


def get_bbox_from_mask(mask):
    assert isinstance(mask, np.ndarray) and mask.ndim == 2

    x = np.where(np.sum(mask, axis=0))[0]
    y = np.where(np.sum(mask, axis=1))[0]
    if len(x) > 1 and len(y) > 1:
        x1, y1, x2, y2 = int(x[0]), int(y[0]), int(x[-1]) + 1, int(y[-1]) + 1
        return [x1, y1, x2, y2]
    else:
        return None


def remove_punctuation(input_str):
    # 1. \p{P} 匹配所有Unicode标点符号
    temp = regex.sub(r'\p{P}', ' ', input_str)

    # 2. 再使用string.punctuation中的标点进行二次替换
    # 确保覆盖当前Python版本定义的所有标点（包括可能未被\p{P}覆盖的）
    punct_pattern = regex.compile(f"[{regex.escape(string.punctuation + '™°一º©®★')}]")
    return punct_pattern.sub(' ', temp)


def full_to_half(text):
    # 先处理全角空格
    text = text.replace('　', ' ')
    # 再处理其他全角字符
    return re.sub(r'[Ａ-Ｚａ-ｚ０-９！-～]', lambda x: chr(ord(x.group(0)) - 65248), text)


def is_english(input_text):
    """
    判断输入的文本是否为英文（忽略所有标点符号、数字及特定特殊字符）

    参数:
        input_text: 待检查的文本字符串

    返回:
        True: 如果文本是英文
        False: 如果文本包含非英文字符
    """
    # 定义需要忽略的特殊字符
    ignored_special_chars = {'°', '®', '™'}

    # 检查字符串中的每个字符
    for char in input_text:
        # 检查是否是英文字母（大小写）- 允许
        if 'A' <= char <= 'Z' or 'a' <= char <= 'z':
            continue
        # 检查是否是数字（0-9）- 忽略
        if '0' <= char <= '9':
            continue
        # 检查是否是需要忽略的特殊字符
        if char in ignored_special_chars:
            continue
        # 所有其他字符都视为需要检查的字符
        # 如果字符不是英文字母、数字或忽略的特殊字符
        # 且不在基本拉丁符号范围内（即可能是非英文字符）
        if not (0x0020 <= ord(char) <= 0x007E):
            return False
    # 所有字符都检查通过
    return True


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


def get_product_and_mask_image(image, mask_image, padding=0, fill_color=(255, 255, 255)):
    mask_image = np.array(mask_image.convert('L'))
    bbox = get_bbox_from_mask(mask_image)

    mask = np.float32(mask_image[..., None]) / 255
    product_image = mask * np.array(image, dtype=np.float32) + (1 - mask) * 255
    product_image = np.uint8(np.clip(product_image, 0, 255))

    x1, y1, x2, y2 = bbox
    product_image = Image.fromarray(product_image[y1: y2, x1: x2])
    if padding > 0:
        product_image = pad_image(product_image, padding, fill_color)

    mask_image = Image.fromarray(mask_image[y1: y2, x1: x2])
    if padding > 0:
        mask_image = pad_image(mask_image, padding, 0)
    return product_image, mask_image


def rm_image_border(image, upper=240, lower=15):
    image = np.array(image)
    # 转换成灰度图
    mean_img = np.mean(image, axis=-1)
    # 裁剪白边
    x = np.where(np.mean(mean_img, axis=0) < upper)[0]
    y = np.where(np.mean(mean_img, axis=1) < upper)[0]
    if len(x) > 1 and len(y) > 1:
        x1, y1, x2, y2 = int(x[0]), int(y[0]), int(x[-1]) + 1, int(y[-1]) + 1
    else:
        raise Exception("The whole picture is white, check the input image.")
    mean_img = mean_img[y1:y2, x1:x2]
    image = image[y1:y2, x1:x2]

    # 裁剪黑边
    x = np.where(np.mean(mean_img, axis=0) > lower)[0]
    y = np.where(np.mean(mean_img, axis=1) > lower)[0]
    if len(x) > 1 and len(y) > 1:
        x1, y1, x2, y2 = int(x[0]), int(y[0]), int(x[-1]) + 1, int(y[-1]) + 1
    else:
        raise Exception("The whole picture is black, check the input image.")
    image = image[y1:y2, x1:x2]

    return Image.fromarray(image)


def resize_image_by_short_side(img, size=1024):
    width, height = img.size
    short_side = min(width, height)

    if short_side <= size:
        return img

    # 计算缩放比例
    scale = size / short_side
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized_img = img.resize((new_width, new_height), 1)
    return resized_img
