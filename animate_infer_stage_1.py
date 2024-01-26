import os
from os.path import join, splitext
from collections import OrderedDict
import random
import numpy as np
from PIL import Image, ImageOps
import pandas
import torch

from diffusers.pipelines.vip.vip_animate_image import VIPAnimateImagePipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.models.controlnetxs import ControlNetXSModel
from diffusers.models.referencenet import ReferenceNetModel
from diffusers.utils.vip_utils import *
from diffusers.data import DrawPose, pkl_load


device = torch.device("cuda")
dtype = torch.float16
stride = 4
num_frames = 24
H, W = 1024, 768

root_dir = "/xsl/wilson.xu/pidm_data_mini"

pkl_file = join(root_dir, "mini_img.pkl")
img_dir = join(root_dir, "train_img")
pose_dir = join(root_dir, "train_pose")

base_path = "/xsl/wilson.xu/weights/film"
referencenet_model_path = "/xsl/wilson.xu/animate_xs_768/referencenet"
controlnet_path = "/xsl/wilson.xu/animate_xs_768/controlnet"

out_dir = "output"
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


_draw = DrawPose(prob_hand=1.0, prob_face=0.0)
def get_reference_pose_frame(video_list):
    item = random.choice(video_list)

    img = load_image(join(img_dir, item['image']))
    pose = pkl_load(join(pose_dir, item['pose']))
    pose = _draw.draw_pose(img, pose)

    # img = pad_image(img)
    # pose = pad_image(pose, 0)

    video_list.remove(item)
    ref_item = random.choice(video_list)
    reference_image = load_image(join(img_dir, ref_item['image']))
    # reference_image = pad_image(reference_image)

    return reference_image, pose, img


if __name__ == '__main__':
    metadata = pkl_load(pkl_file)

    controlnet = ControlNetXSModel.from_pretrained(controlnet_path).to(device, dtype=dtype)
    referencenet = ReferenceNetModel.from_pretrained(referencenet_model_path).to(device, dtype=dtype)
    pipe = VIPAnimateImagePipeline.from_pretrained(
        base_path,
        controlnet=controlnet,
        referencenet=referencenet,
        torch_dtype=dtype).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    ind = 1
    for k, v in metadata.items():
        print(f"{ind}: {k}")
        seed = get_fixed_seed(-1)
        generator = get_torch_generator(seed, device=device)

        reference, pose, gt_img = get_reference_pose_frame(v)
        out = pipe(
            prompt="",
            reference_image=reference,
            control_image=pose,
            height=H,
            width=W,
            num_inference_steps=25,
            guidance_scale=2.0,
            negative_prompt="",
            num_images_per_prompt=1,
            generator=generator,
        )
        out_img = np.concatenate(
            [reference.resize((W, H), 1), pose.resize((W, H), 1),
             gt_img.resize((W, H), 1), out[0]], axis=1)
        Image.fromarray(out_img).save(join(out_dir, k + '.png'))
        ind += 1

    print('Done!')
