import os
from os.path import join, splitext
from collections import OrderedDict
import random
import numpy as np
from PIL import Image, ImageOps
import pandas
import torch

from transformers import AutoProcessor, CLIPVisionModel
from diffusers.pipelines.vip.vip_animate_image import VIPAnimateImagePipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.models import ControlNetModel
from diffusers.models.referencenet import ReferenceNetModel
from diffusers.models.pose_guider import PoseGuider
from diffusers.utils.vip_utils import *

device = torch.device("cuda")
dtype = torch.float16
stride = 4
num_frames = 24

root_dir = "/xsl/wilson.xu/fashion_video"
out_dir = "output"
csv_file = join(root_dir, "train_png.csv")
img_dir = join(root_dir, "train_png")
pose_dir = join(root_dir, "train_png_pose")
base_path = "/xsl/wilson.xu/weights/stable-diffusion-v1-5"
pose_guider_path = "/xsl/wilson.xu/animate_sd1.5_ctrl_10/pose_guider"
referencenet_model_path = "/xsl/wilson.xu/animate_sd1.5_ctrl_10/referencenet"
controlnet_path = "/xsl/wilson.xu/animate_sd1.5_ctrl_10/controlnet"
os.makedirs(out_dir, exist_ok=True)


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


def pad_image(img, pad_values=255):
    w, h = img.size
    img = np.array(img)
    if w > h:
        pad_ = w - h
        img = np.pad(
            img,
            ((pad_ // 2, pad_ - pad_ // 2), (0, 0), (0, 0)),
            constant_values=pad_values
        )
    elif h > w:
        pad_ = h - w
        img = np.pad(
            img,
            ((0, 0), (pad_ // 2, pad_ - pad_ // 2), (0, 0)),
            constant_values=pad_values
        )
    return Image.fromarray(img)


def get_reference_pose_frame(video_list):
    st_idx = random.randint(0, stride - 1)
    samples_idx = list(range(st_idx, len(video_list), stride))
    if len(samples_idx) >= num_frames:
        st_idx = random.randint(0, len(samples_idx) - num_frames)
        samples_idx = samples_idx[st_idx: st_idx + num_frames]
    else:
        samples_idx = samples_idx + [samples_idx[-1], ] * (num_frames - len(samples_idx))

    name = video_list[random.choice(samples_idx)]
    img = load_image(join(img_dir, name))
    img = pad_image(img)
    pose = load_image(join(pose_dir, name))
    pose = pad_image(pose, 0)

    ref_name = video_list[random.choice(samples_idx)]
    reference_image = load_image(join(img_dir, ref_name))
    reference_image = pad_image(reference_image)

    return reference_image, pose, img


if __name__ == '__main__':
    metadata = pandas.read_csv(csv_file).values.tolist()
    video_to_image = OrderedDict()
    for name, captions in metadata:
        video_name = name.split('_')[0]
        if video_name not in video_to_image:
            video_to_image[video_name] = [name]
        else:
            video_to_image[video_name].append(name)

    feature_extractor = AutoProcessor.from_pretrained(join(base_path, "image_encoder"))
    image_encoder = CLIPVisionModel.from_pretrained(base_path, subfolder="image_encoder").to(device, dtype=dtype)
    controlnet = ControlNetModel.from_pretrained(controlnet_path).to(device, dtype=dtype)
    # pose_guider = PoseGuider.from_pretrained(pose_guider_path).to(device, dtype=dtype)
    referencenet = ReferenceNetModel.from_pretrained(referencenet_model_path).to(device, dtype=dtype)
    pipe = VIPAnimateImagePipeline.from_pretrained(
        base_path,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        controlnet=controlnet,
        # pose_guider=pose_guider,
        referencenet=referencenet,
        torch_dtype=dtype).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    seed = get_fixed_seed(-1)
    generator = get_torch_generator(seed, device=device)

    for k, v in video_to_image.items():
        print(k)

        reference, pose, gt_img = get_reference_pose_frame(v)
        out = pipe(
            prompt="",
            reference_image=reference,
            control_image=pose,
            height=512,
            width=512,
            num_inference_steps=25,
            guidance_scale=1.0,
            negative_prompt="",
            num_images_per_prompt=1,
            generator=generator,
        )
        out_img = np.concatenate(
            [reference.resize((512, 512), 1), pose.resize((512, 512), 1),
             gt_img.resize((512, 512), 1), out[0]], axis=1)
        Image.fromarray(out_img).save(join(out_dir, k + '.png'))

    print('Done!')
