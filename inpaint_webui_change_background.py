import os
import numpy as np
from PIL import Image
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers import VIPStableDiffusionControlNetInpaintPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler, KDPMPP2MDiscreteScheduler

from diffusers.utils.prompt_parser import load_webui_textual_inversion
from diffusers.utils.vip_utils import *


# set args
root_dir = "/xsl/wilson.xu/weights"
base_path = f"{root_dir}/CyberRealistic_V3.0-FP32"
canny_path = f"{root_dir}/control_v11p_sd15_canny"
pose_path = f"{root_dir}/control_v11p_sd15_openpose"
face_restored_path = f"{root_dir}/codeformer-v0.1.0.pth"

device = torch.device("cuda")
dtype = torch.float16

input_size = (1024, 1024)
seed = get_fixed_seed(3365673641)
batch_size = 1
generator = get_torch_generator(seed, batch_size, device=device)
print(f'processing seed = {seed}')

img_file = "/xsl/wilson.xu/case/4.png"
mask_file = "/xsl/wilson.xu/case/3.png"
save_dir = f'./output'
os.makedirs(save_dir, exist_ok=True)

# prompt
prompt = "bokeh,outdoors background,clear background, beautiful light, best quality, masterpiece, ultra highres, very detailed skin, photorealistic, masterpiece, very high detailed,4k, 8k, 64k,photo detail"
negative_prompt = "floor reflection, tilt, people, ((messy hair:2)), tree, ((light at background)), human, ((no peoples background)), ((sun)),((backlight)), ((light above the head)), ng_deepnegative_v1_75t , (badhandv4), verybadimagenegative_v1.3, easynegative, EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), 2 body, 2 pussy, 2 upper, 2 lower, ((2 head)), ((3 hands)), 3 feet, 3 ear, 4 ear, bad anatomy,((cross-eyed)), bad hands, bad feet, ((watermark:2)),((logo:2)), (moles:2), illustration ,3d, sepia ,painting ,cartoon, sketch, light above the head, stand on the table, high light on body"


# init inpaint pipeline
canny_model = ControlNetModel.from_pretrained(canny_path, torch_dtype=dtype).to(device)
controlnet = MultiControlNetModel([canny_model])
pipe_control = VIPStableDiffusionControlNetInpaintPipeline.from_pretrained(
    base_path, controlnet=controlnet, torch_dtype=dtype).to(device)
# load Euler a scheduler
pipe_control.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_control.scheduler.config)
# load textual inversion embeddings
load_webui_textual_inversion("embedding", pipe_control)

# load image and inpaint mask
init_image = load_image(img_file)
ori_mask = load_image(mask_file)
mask_image = mask_process(ori_mask, invert_mask=True)

# clothing canny
canny_image = cv2.Canny(np.array(ori_mask), 100, 200)[..., None]
canny_image = np.tile(canny_image, [1, 1, 3])
canny_image = Image.fromarray(canny_image)

### start pipeline ###
pipe_list, image_overlay = pipe_control(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    mask_image=mask_image,
    control_image=[canny_image],
    height=input_size[0],
    width=input_size[1],
    strength=0.75,
    num_inference_steps=35,
    guidance_scale=9,  # CFGscale
    num_images_per_prompt=batch_size,
    generator=generator,
    controlnet_conditioning_scale=1.0)

# Paste back according to the mask
pipe_list = alpha_composite(pipe_list, image_overlay)

# save images
final = np.concatenate([init_image, ori_mask, canny_image], axis=1)
for i, out_img in enumerate(pipe_list):
    Image.fromarray(
        np.uint8(np.concatenate((final, out_img), axis=1))).save(
            f"{save_dir}/test_webui_{i}.png")
