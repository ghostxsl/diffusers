import os
from os.path import join, splitext
from collections import OrderedDict
import random
import numpy as np
from PIL import Image, ImageOps
import pandas
import torch

from diffusers.pipelines.vip.vip_animate_video import VIPAnimateVideoPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.models import ControlNetModel, MotionAdapter
from diffusers.models.referencenet import ReferenceNetModel
from diffusers.models.controlnetxs import ControlNetXSModel
from diffusers.models.controlnetxs_motion_model import ControlNetXSMotionModel
from diffusers.utils.vip_utils import *
from diffusers.utils import export_to_gif, export_to_video
from diffusers.data import DrawPose, pkl_load


device = torch.device("cuda")
dtype = torch.float16
total_frames = 25
stride = 4
num_frames = 12
overlap_frame = 4
H, W = 768, 768

root_dir = "/xsl/wilson.xu/fashion_video"

csv_file = join(root_dir, "mini_train_png.csv")
img_dir = join(root_dir, "train_png")
pose_dir = join(root_dir, "train_pose")

base_path = "/xsl/wilson.xu/weights/film"
referencenet_model_path = "/xsl/wilson.xu/animate_motion_768_0116/referencenet"
controlnet_path = "/xsl/wilson.xu/animate_s2_768_0119/controlnet_motion/"
motion_adapter_model_path = "/xsl/wilson.xu/animate_s2_768_0119/motion_adapter/"

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


_draw = DrawPose(prob_hand=1.0, prob_face=1.0)
def get_reference_pose_frames(video_list, num_frames=25, stride=4):
    st_idx = random.randint(0, stride - 1)
    samples_idx = list(range(st_idx, len(video_list), stride))
    if len(samples_idx) >= num_frames:
        st_idx = random.randint(0, len(samples_idx) - num_frames)
        samples_idx = samples_idx[st_idx: st_idx + num_frames]
    else:
        samples_idx = samples_idx + [samples_idx[-1], ] * (num_frames - len(samples_idx))

    pose_list, imgs = [], []
    for i in samples_idx:
        name = video_list[i]
        img = load_image(join(img_dir, name))

        pose = pkl_load(join(pose_dir, splitext(name)[0] + '.pose'))
        pose = _draw.draw_pose(img.size[::-1], pose)

        img = pad_image(img)
        pose = pad_image(pose, 0)
        imgs.append(img)
        pose_list.append(pose)

    ref_name = video_list[random.choice(samples_idx)]
    reference_image = load_image(join(img_dir, ref_name))
    reference_image = pad_image(reference_image)

    return reference_image, pose_list, imgs


if __name__ == '__main__':
    metadata = pandas.read_csv(csv_file).values.tolist()
    video_to_image = OrderedDict()
    for name, captions in metadata:
        video_name = name.split('_')[0]
        if video_name not in video_to_image:
            video_to_image[video_name] = [name]
        else:
            video_to_image[video_name].append(name)

    motion_adapter = MotionAdapter.from_pretrained(motion_adapter_model_path)
    controlnet = ControlNetXSModel.from_pretrained(controlnet_path).to(device, dtype=dtype)
    referencenet = ReferenceNetModel.from_pretrained(referencenet_model_path).to(device, dtype=dtype)
    pipe = VIPAnimateVideoPipeline.from_pretrained(
        base_path,
        motion_adapter=motion_adapter,
        controlnet=controlnet,
        referencenet=referencenet,
        num_frames=num_frames,
        overlap_frame=overlap_frame,
        torch_dtype=dtype).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="linear",
    )

    ind = 1
    for k, v in video_to_image.items():
        print(f"{ind}: {k}")
        seed = get_fixed_seed(-1)
        generator = get_torch_generator(seed, device=device)

        reference, pose, gt_img = get_reference_pose_frames(v, total_frames, stride)
        out = pipe(
            prompt="woman posing for a photo, simple white background",
            reference_image=reference,
            control_images=pose,
            height=H,
            width=W,
            num_inference_steps=25,
            guidance_scale=1.0,
            negative_prompt="",
            num_videos_per_prompt=1,
            generator=generator,
        )
        out_gifs = []
        reference = reference.resize((H, W), 1)
        for out_img, gt, pose_img in zip(out[0], gt_img, pose):
            out_gif = np.concatenate(
                [reference, pose_img.resize((H, W), 1),
                 gt.resize((H, W), 1), out_img], axis=1
            )
            out_gifs.append(Image.fromarray(out_gif))
        export_to_gif(out_gifs, join(out_dir, k + '.gif'))
        ind += 1

    print('Done!')
