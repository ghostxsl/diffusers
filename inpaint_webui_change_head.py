import os
import numpy as np
from PIL import Image
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers import VIPStableDiffusionControlNetInpaintPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler, KDPMPP2MDiscreteScheduler
from extensions.CodeFormer.inference_codeformer_re import apply_codeformer
from mmpose.blendpose.inferencer import VIPPoseInferencer

from diffusers.utils.prompt_parser import load_webui_textual_inversion
from diffusers.utils.vip_utils import *


# set args
root_dir = "/xsl/wilson.xu/weights"
base_path = f"{root_dir}/braBeautifulRealistic_brav3"
canny_path = f"{root_dir}/control_v11p_sd15_canny"
pose_path = f"{root_dir}/control_v11p_sd15_openpose"
face_restored_path = f"{root_dir}/codeformer-v0.1.0.pth"
lora_model_path = f"{root_dir}/lora_10ep_lr1e-4_vip_2023_07_05_as_woman_braBeautifulRealistic_brav3.safetensors"
lora_ratio = 1.0

device = torch.device("cuda")
dtype = torch.float16

input_size = (1024, 1024)
seed = get_fixed_seed(179174)
batch_size = 1
generator = get_torch_generator(seed, batch_size, device=device)
print(f'processing seed = {seed}')

img_file = "/xsl/wilson.xu/case/test_0.jpg"
mask_file = "/xsl/wilson.xu/case/test_0_mask.jpg"
save_dir = f'./output'
os.makedirs(save_dir, exist_ok=True)

# prompt
prompt = "1girl,smile,solo,(best quality),(masterpiece:1.1), upper body, long black hair, cute, clear facial skin"
negative_prompt = "(cross-eye:2), EasyNegative, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans,extra fingers,fewer fingers,, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, (only body)"


# init inpaint pipeline
pose_model = ControlNetModel.from_pretrained(pose_path, torch_dtype=dtype).to(device)
controlnet = MultiControlNetModel([pose_model])
pipe_control = VIPStableDiffusionControlNetInpaintPipeline.from_pretrained(
    base_path, controlnet=controlnet, torch_dtype=dtype).to(device)
# load lora weight
load_lora_weights(pipe_control, lora_model_path, lora_ratio, device="cuda", dtype=dtype)
# load DPM++ 2M Karras scheduler
pipe_control.scheduler = KDPMPP2MDiscreteScheduler.from_config(pipe_control.scheduler.config)
# load textual inversion embeddings
load_webui_textual_inversion("embedding", pipe_control)


# load pose model
det_config = "/xsl/wilson.xu/mmpose/demo/mmdetection_cfg/rtmdet_l_8xb32-300e_coco.py"
det_checkpoint = "/xsl/wilson.xu/weights/rtmpose_model/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
body_config = "/xsl/wilson.xu/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-420e_body8-256x192.py"
body_checkpoint = "/xsl/wilson.xu/weights/rtmpose_model/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"
pose_inferencer = VIPPoseInferencer(det_config, det_checkpoint,
            body_config, body_checkpoint, kpt_thr=0.3, device=device)

# load image and inpaint mask
init_image = load_image(img_file)
mask_image = mask_process(load_image(mask_file), invert_mask=False)

# get pose image
pose_image = Image.fromarray(pose_inferencer(np.array(init_image)))

### start pipeline ###
pipe1_list, image1_overlay = pipe_control(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    mask_image=mask_image,
    control_image=[pose_image],
    height=input_size[0],
    width=input_size[1],
    strength=0.75,
    num_inference_steps=20,
    guidance_scale=6,  # CFGscale
    num_images_per_prompt=batch_size,
    generator=generator,
    controlnet_conditioning_scale=1.0)

# 2. Adetailer
for i, img in enumerate(pipe1_list):
    bboxes, masks, preview = mediapipe_face_detection(img)
    for mask in masks:
        mask = mask_process(mask, invert_mask=False)
        crop_region = get_crop_region(np.array(mask), pad=32)
        crop_region = expand_crop_region(crop_region,
                                         input_size[1], input_size[0],
                                         mask.width, mask.height)
        mask = mask.crop(crop_region)
        img2 = img.crop(crop_region)
        pipe2_list, image2_overlay = pipe_control(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img2,
            mask_image=mask,
            control_image=[pose_image],
            height=input_size[0],
            width=input_size[1],
            strength=0.2,
            num_inference_steps=20,
            guidance_scale=6,  # CFGscale
            num_images_per_prompt=1,
            generator=generator[i] if isinstance(generator, list) else generator,
            controlnet_conditioning_scale=0.)
        pipe2_list = apply_codeformer(face_restored_path, pipe2_list)
        # Paste back according to the mask
        pipe2_list = alpha_composite(pipe2_list, image2_overlay)
        x1, y1, x2, y2 = crop_region
        img.paste(
            pipe2_list[0].resize(
                (int(x2 - x1), int(y2 - y1)), resample=Image.LANCZOS),
            (x1, y1))
    pipe1_list[i] = img

# Paste back according to the mask
pipe1_list = alpha_composite(pipe1_list, image1_overlay)

# save images
final = np.concatenate([init_image, mask_image.convert("RGB"), pose_image], axis=1)
for i, out_img in enumerate(pipe1_list):
    Image.fromarray(
        np.uint8(np.concatenate((final, out_img), axis=1))).save(
            f"{save_dir}/test_webui_{i}.png")
