import os
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw
import torch
import safetensors
from collections import defaultdict
from diffusers import ControlNetModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel
from diffusers import WebUIStableDiffusionControlNetInpaintPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler
import extensions.ESRGAN.test as esrgan
from extensions.CodeFormer.inference_codeformer_re import apply_codeformer
from mmpose.blendpose.inferencer import VIPPoseInferencer

from diffusers.utils.promt_parser import get_promt_embedding, load_webui_textual_inversion
from diffusers.utils.webui_utils import *


def prompt_preprocess(pos_prompt, neg_prompt, pipe, device=torch.device("cuda")):
    pos_embeds = get_promt_embedding(pos_prompt, pipe.tokenizer, pipe.text_encoder, device)
    neg_embeds = get_promt_embedding(neg_prompt, pipe.tokenizer, pipe.text_encoder, device)
    # pad embedding
    len_pos, len_neg = pos_embeds.shape[1], neg_embeds.shape[1]
    if len_pos > len_neg:
        pad_embed = neg_embeds[:, -1].unsqueeze(1).repeat(1, len_pos - len_neg, 1)
        neg_embeds = torch.cat([neg_embeds, pad_embed], dim=1)
    elif len_neg > len_pos:
        pad_embed = pos_embeds[:, -1].unsqueeze(1).repeat(1, len_neg - len_pos, 1)
        pos_embeds = torch.cat([pos_embeds, pad_embed], dim=1)
    return pos_embeds, neg_embeds


def load_lora_weights(pipeline, checkpoint_path, multiplier, device, dtype):
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


### init pipeline ###
root_dir = "/xsl/wilson.xu/weights"
base_path = f"{root_dir}/braBeautifulRealistic_brav3"
canny_path = f"{root_dir}/control_v11p_sd15_canny"
pose_path = f"{root_dir}/control_v11p_sd15_openpose"
face_restored_path = f"{root_dir}/codeformer-v0.1.0.pth"
lora_model_path = f"{root_dir}/lora_10ep_lr1e-4_vip_2023_07_05_as_woman_braBeautifulRealistic_brav3.safetensors"
lora_ratio = 1.0

device = torch.device("cuda")
dtype = torch.float16

# canny_model = ControlNetModel.from_pretrained(canny_path, torch_dtype=dtype).to(device)
pose_model = ControlNetModel.from_pretrained(pose_path, torch_dtype=dtype).to(device)
controlnet = MultiControlNetModel([pose_model])

pipe_control = WebUIStableDiffusionControlNetInpaintPipeline.from_pretrained(
    base_path, controlnet=controlnet, torch_dtype=dtype).to(device)
pipe_head = load_lora_weights(pipe_control, lora_model_path, lora_ratio, device="cuda", dtype=dtype)
pipe_control.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_control.scheduler.config)
load_webui_textual_inversion("embedding", pipe_control)

# load pose model
det_config = "/xsl/wilson.xu/mmpose/demo/mmdetection_cfg/rtmdet_l_8xb32-300e_coco.py"
det_checkpoint = "/xsl/wilson.xu/weights/rtmpose_model/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
body_config = "/xsl/wilson.xu/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-420e_body8-256x192.py"
body_checkpoint = "/xsl/wilson.xu/weights/rtmpose_model/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"
pose_inferencer = VIPPoseInferencer(det_config, det_checkpoint,
            body_config, body_checkpoint, kpt_thr=0.3, device=device)


# prompt_embedding
prompt = "1girl,smile,solo,(best quality),(masterpiece:1.1), upper body, long black hair, cute, clear facial skin"
negative_prompt = "(cross-eye:2), EasyNegative, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans,extra fingers,fewer fingers,, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, (only body)"
pos_prompt_embs, neg_prompt_embs = prompt_preprocess(prompt, negative_prompt, pipe_control, device)


input_size = (1024, 1024)
seed = get_fixed_seed(179174)
generator = torch.Generator("cuda").manual_seed(seed)
print(f'processing seed = {seed}')

img_file = "/xsl/wilson.xu/case/test_0.jpg"
mask_file = "/xsl/wilson.xu/case/test_0_mask.jpg"
save_dir = f'./output'
os.makedirs(save_dir, exist_ok=True)

### processing input ###
init_image = load_image(img_file)
# TODO: scale any size input
upscale = 1 if init_image.size[1] < 1024 else 0
print(f"image_size = {init_image.size}, upscale = {upscale}")
if upscale:
    w, h = init_image.size
    init_image = Image.fromarray(esrgan.upscale(
        np.asarray(init_image))).resize((w * 2, h * 2))  # 放大四倍再resize回2倍
if init_image.size[0] < input_size[0] or init_image.size[1] < input_size[1]:
    bgImg: Image.Image = Image.new("RGB", input_size, (255, 255, 255))
    bgImg.paste(init_image, (round((input_size[0] - init_image.size[0]) / 2), round((input_size[1] - init_image.size[1]) / 2)))
    init_image = bgImg
else:
    init_image = init_image.resize(input_size)
# inpaint mask
ori_mask = load_image(mask_file)
mask_image = mask_process(ori_mask, inpainting_mask_invert=False)

# # clothing canny
# canny_image = cv2.Canny(np.array(ori_mask), 100, 200)[..., None]
# canny_image = np.tile(canny_image, [1, 1, 3])
# canny_image = Image.fromarray(canny_image)

# pose image
pose_image = Image.fromarray(pose_inferencer(np.array(init_image)))

### start pipeline ###
pipe1_list, has_nsfw_concept_1 = pipe_control(
        image=init_image,
        mask_image=mask_image,
        control_image=[pose_image],
        height=input_size[0],
        width=input_size[1],
        strength=0.75,
        num_inference_steps=20,
        guidance_scale=6,  # CFGscale
        num_images_per_prompt=1,
        generator=generator,
        prompt_embeds=pos_prompt_embs,
        negative_prompt_embeds=neg_prompt_embs,
        return_dict=False,
        controlnet_conditioning_scale=1.0,)

# np_mask = np.array(mask_image, dtype=np.float32) / 255
# np_mask = np.around(np_mask)[..., None]
# for i, out_img in enumerate(pil_img_list):
#     pil_img_list[i] = Image.fromarray(
#         np.uint8(np.array(out_img) * np_mask + np.array(init_image) * (1 - np_mask)))


# 2. Adetailer
for i, is_valid in enumerate(has_nsfw_concept_1):
    if not is_valid:
        img = pipe1_list[i]
        bboxes, masks, preview = mediapipe_face_detection(img)
        for mask in masks:
            mask = mask_process(mask, inpainting_mask_invert=False)
            crop_region = get_crop_region(np.array(mask), pad=32)
            crop_region = expand_crop_region(crop_region,
                                             input_size[1], input_size[0],
                                             mask.width, mask.height)
            mask = mask.crop(crop_region)
            img2 = img.crop(crop_region)
            pipe2_list, has_nsfw_concept = pipe_control(
                image=img2,
                mask_image=mask,
                control_image=[pose_image],
                height=input_size[0],
                width=input_size[1],
                strength=0.2,
                num_inference_steps=20,
                guidance_scale=6,  # CFGscale
                num_images_per_prompt=1,
                generator=generator,
                prompt_embeds=pos_prompt_embs,
                negative_prompt_embeds=neg_prompt_embs,
                return_dict=False,
                controlnet_conditioning_scale=0.)
            pipe2_list = apply_codeformer(face_restored_path, pipe2_list)
            x1, y1, x2, y2 = crop_region
            img.paste(
                pipe2_list[0].resize(
                    (int(x2 - x1), int(y2 - y1)), resample=Image.LANCZOS),
                (x1, y1))
        pipe1_list[i] = img

# save images
final = np.concatenate([init_image, ori_mask, pose_image], axis=1)
for i, out_img in enumerate(pipe1_list):
    Image.fromarray(
        np.uint8(np.concatenate((final, out_img), axis=1))).save(
            f"{save_dir}/test_webui_{i}.png")
