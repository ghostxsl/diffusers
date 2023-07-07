# !pip install opencv-python transformers accelerate
import argparse

import numpy as np
import torch
from PIL import Image
from pipeline_controlnet_xs import StableDiffusionControlNetXSPipeline

from diffusers.models.controlnetxs import ControlNetXSModel
from diffusers.utils import load_image
from diffusers.schedulers import EulerAncestralDiscreteScheduler


parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt", type=str, default="woman posing for a photo, 4k, high-res, masterpiece, (best quality), head:1.3,((Hasselblad photography)), finely detailed skin, sharp focus, dynamic angle, (cinematic lighting), [:(detailed face:1.2):0.2]"
)
parser.add_argument("--negative_prompt", type=str, default="(nsfw:2), (greyscale:1.2), paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), low-res, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans")
parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
parser.add_argument(
    "--image_path",
    type=str,
    default="/xsl/wilson.xu/pose.png",
)
parser.add_argument("--num_inference_steps", type=int, default=25)

args = parser.parse_args()

prompt = args.prompt
negative_prompt = args.negative_prompt
# load an image
pose_image = load_image(args.image_path)

# initialize the models and pipeline
controlnet_conditioning_scale = args.controlnet_conditioning_scale
controlnet = ControlNetXSModel.from_pretrained("/xsl/wilson.xu/controlnetxs_pose_100k", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetXSPipeline.from_pretrained(
    "/xsl/wilson.xu/weights/film", controlnet=controlnet, torch_dtype=torch.float16
).to(torch.device("cuda"))
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
num_inference_steps = args.num_inference_steps

# generate image
image = pipe(prompt, pose_image, 512, 512,
    negative_prompt=negative_prompt,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    num_inference_steps=num_inference_steps,
).images[0]
image.save("cnxs_pose.png")
