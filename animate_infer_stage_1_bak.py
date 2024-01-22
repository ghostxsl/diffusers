import os
from os.path import join, splitext
from collections import OrderedDict
import random
import numpy as np
from PIL import Image, ImageOps
import pandas
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
)
from diffusers.models.embeddings import IPAdapterPlusImageProjection
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
H, W = 768, 768

root_dir = "/xsl/wilson.xu/fashion_video"

csv_file = join(root_dir, "mini_train_png.csv")
img_dir = join(root_dir, "train_png")
pose_dir = join(root_dir, "train_pose")

base_path = "/xsl/wilson.xu/weights/film"
referencenet_model_path = "/xsl/wilson.xu/animate_xs_768/referencenet"
controlnet_path = "/xsl/wilson.xu/animate_xs_768/controlnet"
image_encoder_model_path = "/xsl/wilson.xu/weights/IP-Adapter/image_encoder"

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

    pose = pkl_load(join(pose_dir, splitext(name)[0] + '.pose'))
    pose = _draw.draw_pose(img, pose)

    img = pad_image(img)
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

    controlnet = ControlNetXSModel.from_pretrained(controlnet_path).to(device, dtype=dtype)
    referencenet = ReferenceNetModel.from_pretrained(referencenet_model_path).to(device, dtype=dtype)
    pipe = VIPAnimateImagePipeline.from_pretrained(
        base_path,
        controlnet=controlnet,
        referencenet=referencenet,
        torch_dtype=dtype).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # pipe.feature_extractor = CLIPImageProcessor()
    # pipe.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_model_path).to(device, dtype=dtype)
    #
    # unet = pipe.unet
    # num_image_text_embeds = 16
    # unet.encoder_hid_proj = None
    #
    # # set ip-adapter cross-attention processors
    # attn_procs = {}
    # for name in unet.attn_processors.keys():
    #     cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    #     if name.startswith("mid_block"):
    #         hidden_size = unet.config.block_out_channels[-1]
    #     elif name.startswith("up_blocks"):
    #         block_id = int(name[len("up_blocks.")])
    #         hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    #     elif name.startswith("down_blocks"):
    #         block_id = int(name[len("down_blocks.")])
    #         hidden_size = unet.config.block_out_channels[block_id]
    #     if cross_attention_dim is None or "motion_modules" in name:
    #         attn_processor_class = (
    #             AttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else AttnProcessor
    #         )
    #         attn_procs[name] = attn_processor_class()
    #     else:
    #         attn_processor_class = (
    #             IPAdapterAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else IPAdapterAttnProcessor
    #         )
    #         attn_procs[name] = attn_processor_class(
    #             hidden_size=hidden_size,
    #             cross_attention_dim=cross_attention_dim,
    #             scale=1.0,
    #             num_tokens=num_image_text_embeds,
    #         ).to(dtype=unet.dtype, device=unet.device)
    #
    # unet.set_attn_processor(attn_procs)
    #
    # # convert IP-Adapter Image Projection layers to diffusers
    # image_projection = IPAdapterPlusImageProjection(
    #     embed_dims=1280,
    #     output_dims=768,
    #     hidden_dims=768,
    #     heads=12,
    #     num_queries=num_image_text_embeds,
    # )
    #
    # unet.encoder_hid_proj = image_projection.to(device=unet.device, dtype=unet.dtype)
    # unet.config.encoder_hid_dim_type = "ip_image_proj"
    # ip_adapter_state_dict = load_file("/xsl/wilson.xu/animate_film_ip_150/pytorch_ip_adapter_weights.safetensors")
    # unet.load_state_dict(ip_adapter_state_dict, strict=False)

    ind = 1
    for k, v in video_to_image.items():
        print(f"{ind}: {k}")
        seed = get_fixed_seed(-1)
        generator = get_torch_generator(seed, device=device)

        reference, pose, gt_img = get_reference_pose_frame(v)
        out = pipe(
            prompt="woman posing for a photo, simple white background",
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
            [reference.resize((H, W), 1), pose.resize((H, W), 1),
             gt_img.resize((H, W), 1), out[0]], axis=1)
        Image.fromarray(out_img).save(join(out_dir, k + '.png'))
        ind += 1

    print('Done!')
