# Copyright 2023 The VIP AIGC Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from PIL import Image, ImageOps, ImageDraw
import safetensors
from collections import defaultdict
import numpy as np
import cv2
import torch

from .torch_utils import randn_tensor


Image.MAX_IMAGE_PIXELS = None


__all__ = [
    'get_fixed_seed', 'load_image', 'mask_process', 'create_random_tensors',
    'mediapipe_face_detection', 'get_crop_region', 'expand_crop_region',
    'alpha_composite', 'load_lora_weights', 'get_torch_generator'
]


def get_fixed_seed(seed):
    if seed is None or seed == '' or seed == -1:
        return int(random.randrange(4294967294))
    return seed


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
        # Fix UserWarning for palette images with transparency
        if "transparency" in image.info:
            image = image.convert("RGBA")
            image = Image.fromarray(np.array(image)[..., :3])
        image = image.convert("RGB")

    return image


def mask_process(mask, invert_mask=True, blur=4):
    mask = mask.convert("L")
    if invert_mask:
        mask = ImageOps.invert(mask)
    if blur > 0:
        np_mask = np.array(mask)
        kernel_size = 2 * int(4 * blur + 0.5) + 1
        np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), blur)
        np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), blur)
        mask = Image.fromarray(np_mask)
    return mask


def create_mask_from_bbox(bboxes, shape):
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks


def create_mask_from_bbox_to_one_img(bboxes, shape):
    mask = Image.new("L", shape, 0)
    mask_draw = ImageDraw.Draw(mask)
    for bbox in bboxes:
        mask_draw.rectangle(bbox, fill=255)
    return mask


def mediapipe_face_detection(image, model_type=1, confidence=0.3,
                             dilate=4, erode=0):
    """
    model_selection: 0 or 1. 0 to select a short-range model that works
        best for faces within 2 meters from the camera, and 1 for a full-range
        model best for faces within 5 meters. See details in
        https://solutions.mediapipe.dev/face_detection#model_selection.
    """
    import mediapipe as mp

    img_width, img_height = image.size

    mp_face_detection = mp.solutions.face_detection
    draw_util = mp.solutions.drawing_utils

    img_array = np.array(image)

    with mp_face_detection.FaceDetection(
        model_selection=model_type, min_detection_confidence=confidence
    ) as face_detector:
        pred = face_detector.process(img_array)

    if pred.detections is None:
        return [None, None, None]

    preview_array = img_array.copy()

    bboxes = []
    for detection in pred.detections:
        draw_util.draw_detection(preview_array, detection)

        bbox = detection.location_data.relative_bounding_box
        x1 = bbox.xmin * img_width
        y1 = bbox.ymin * img_height
        w = bbox.width * img_width
        h = bbox.height * img_height
        x2 = x1 + w
        y2 = y1 + h

        bboxes.append([x1, y1, x2, y2])

    masks = create_mask_from_bbox(bboxes, image.size)
    preview = Image.fromarray(preview_array)

    def _dilate(img, value):
        img = np.array(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
        return Image.fromarray(cv2.dilate(img, kernel, iterations=1))

    def _erode(img, value):
        img = np.array(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
        return Image.fromarray(cv2.erode(img, kernel, iterations=1))

    if dilate > 0:
        masks = [_dilate(a, dilate) for a in masks]

    if erode > 0:
        masks = [_erode(a, erode) for a in masks]

    return [bboxes, masks, preview]


def get_crop_region(mask, pad=0):
    """finds a rectangular region that contains all masked ares in an image. Returns (x1, y1, x2, y2) coordinates of the rectangle.
    For example, if a user has painted the top-right part of a 512x512 image", the result may be (256, 0, 512, 256)"""

    h, w = mask.shape

    crop_left = 0
    for i in range(w):
        if not (mask[:, i] == 0).all():
            break
        crop_left += 1

    crop_right = 0
    for i in reversed(range(w)):
        if not (mask[:, i] == 0).all():
            break
        crop_right += 1

    crop_top = 0
    for i in range(h):
        if not (mask[i] == 0).all():
            break
        crop_top += 1

    crop_bottom = 0
    for i in reversed(range(h)):
        if not (mask[i] == 0).all():
            break
        crop_bottom += 1

    return (
        int(max(crop_left-pad, 0)),
        int(max(crop_top-pad, 0)),
        int(min(w - crop_right + pad, w)),
        int(min(h - crop_bottom + pad, h))
    )


def expand_crop_region(crop_region, processing_width, processing_height, image_width, image_height):
    """expands crop region get_crop_region() to match the ratio of the image the region will processed in; returns expanded region
    for example, if user drew mask in a 128x32 region, and the dimensions for processing are 512x512, the region will be expanded to 128x128."""

    x1, y1, x2, y2 = crop_region

    ratio_crop_region = (x2 - x1) / (y2 - y1)
    ratio_processing = processing_width / processing_height

    if ratio_crop_region > ratio_processing:
        desired_height = (x2 - x1) / ratio_processing
        desired_height_diff = int(desired_height - (y2-y1))
        y1 -= desired_height_diff//2
        y2 += desired_height_diff - desired_height_diff//2
        if y2 >= image_height:
            diff = y2 - image_height
            y2 -= diff
            y1 -= diff
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        if y2 >= image_height:
            y2 = image_height
    else:
        desired_width = (y2 - y1) * ratio_processing
        desired_width_diff = int(desired_width - (x2-x1))
        x1 -= desired_width_diff//2
        x2 += desired_width_diff - desired_width_diff//2
        if x2 >= image_width:
            diff = x2 - image_width
            x2 -= diff
            x1 -= diff
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        if x2 >= image_width:
            x2 = image_width

    return x1, y1, x2, y2


# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(low, high, val=0.):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1) * low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def create_random_tensors(noise_shape, generator, subseeds=-1, subseed_strength=0.0,
                          device=torch.device("cpu"), dtype=torch.float32):
    if subseeds == -1:
        if isinstance(generator, list):
            sub_generator = [
                torch.Generator(device).manual_seed(
                    get_fixed_seed(-1)) for _ in generator]
        else:
            sub_generator = torch.Generator(device).manual_seed(get_fixed_seed(-1))
    elif isinstance(subseeds, list) and isinstance(generator, list):
        assert len(subseeds) == len(generator)
        sub_generator = [
            torch.Generator(device).manual_seed(
                get_fixed_seed(s)) for s in subseeds]
    elif isinstance(subseeds, (int, float)) and isinstance(generator, torch.Generator):
        sub_generator = torch.Generator(device).manual_seed(subseeds)
    else:
        raise Exception(f"Unknown parameter `subseeds`: {subseeds}")

    noise = randn_tensor(noise_shape, generator=generator, device=device, dtype=dtype)
    subnoise = randn_tensor(noise_shape, generator=sub_generator, device=device, dtype=dtype)

    return slerp(noise, subnoise, subseed_strength)


def alpha_composite(image_list, image_overlay):
    # `alpha_composite` postprocess
    for i, img in enumerate(image_list):
        img = img.convert('RGBA')
        img.alpha_composite(image_overlay)
        image_list[i] = img.convert('RGB')
    return image_list


def load_lora_weights(pipeline, checkpoint_path, multiplier, device="cuda", dtype=torch.float32):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    state_dict = safetensors.torch.load_file(checkpoint_path, device=device)
    # state_dict = load_file(checkpoint_path)
    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value
    # directly update weight in diffusers model
    for layer, elems in updates.items():
        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet
        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)
        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        alpha = elems['alpha']
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0
        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)
    return pipeline


def get_torch_generator(seed, batch_size=1, device=torch.device("cpu")):
    if batch_size == 1:
        generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = [torch.Generator(device).manual_seed(seed + s) for s in range(batch_size)]
    return generator
