# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, split, exists
import random
import numpy as np
from PIL import Image
import torch

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.vip.vip_pose_transfer import VIPPoseTransferPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.models.controlnetxs import ControlNetXSModel
from diffusers.models.referencenet import ReferenceNetModel
from diffusers.utils.vip_utils import *
from diffusers.data.utils import *
from diffusers.data import DrawPose


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Path to dataset file.")
    parser.add_argument(
        "--img_dir",
        default=None,
        type=str,
        help="Directory to train image.")
    parser.add_argument(
        "--pose_dir",
        default=None,
        type=str,
        help="Directory to pose image.")
    parser.add_argument(
        "--out_dir",
        default="output",
        type=str,
        help="Directory to save.")

    parser.add_argument(
        "--crop",
        default=False,
        type=bool,
        help="")

    parser.add_argument(
        "--resolution",
        type=str,
        default="768x576",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/xsl/wilson.xu/pt_7k_0229",
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="/xsl/wilson.xu/weights/vae_ft_mse",
    )
    parser.add_argument(
        "--image_encoder_model_path",
        type=str,
        default="/xsl/wilson.xu/weights/sd-image-variations/image_encoder",
    )
    parser.add_argument(
        "--weight_dir",
        default=None,
        type=str,
        help="Directory to weights.")

    parser.add_argument(
        "--device",
        default='cuda',
        type=str,
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    parser.add_argument(
        "--dtype",
        default='fp16',
        type=str,
        help="Data type to use (e.g. fp16, fp32, etc.)")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if len(args.resolution.split("x")) == 1:
        args.resolution = int(args.resolution)
    elif len(args.resolution.split("x")) == 2:
        args.resolution = [int(r) for r in args.resolution.split("x")]
    else:
        raise Exception(f"Error `resolution` type({type(args.resolution)}): {args.resolution}.")

    args.device = torch.device(args.device)
    if args.dtype == 'fp16':
        args.dtype = torch.float16
    else:
        args.dtype = torch.float32

    return args


_draw = DrawPose(prob_hand=1.0, prob_face=0.0)
def get_reference_pose_frame(video_list, img_dir, pose_dir):
    item = random.choice(video_list)

    img = load_image(join(img_dir, item['image']))
    pose = pkl_load(join(pose_dir, item['pose']))
    pose = _draw.draw_pose(img, pose)

    video_list.remove(item)
    ref_item = random.choice(video_list)
    reference_image = load_image(join(img_dir, ref_item['image']))

    return reference_image, pose, img


if __name__ == '__main__':
    args = parse_args()
    metadata = load_file(args.dataset_file)

    device = args.device
    dtype = args.dtype
    vae = AutoencoderKL.from_pretrained(args.vae_model_path).to(device, dtype=dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_model_path).to(device, dtype=dtype)

    unet = UNet2DConditionModel.from_pretrained(args.base_model_path, subfolder="unet").to(device, dtype=dtype)
    controlnet = ControlNetXSModel.from_pretrained(args.base_model_path, subfolder="controlnet").to(device, dtype=dtype)
    referencenet = ReferenceNetModel.from_pretrained(args.base_model_path, subfolder="referencenet").to(device, dtype=dtype)
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    pipe = VIPPoseTransferPipeline(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=CLIPImageProcessor(),
        image_encoder=image_encoder,
        controlnet=controlnet,
        referencenet=referencenet).to(device)

    for i, (k, v) in enumerate(metadata.items()):
        print(f"{i}: {k}")
        seed = get_fixed_seed(-1)
        generator = get_torch_generator(seed, device=device)

        ref_img, pose, gt_img = get_reference_pose_frame(v, args.img_dir, args.pose_dir)
        out = pipe(
            reference_image=ref_img,
            control_image=pose,
            height=args.resolution[0],
            width=args.resolution[1],
            num_inference_steps=25,
            guidance_scale=2.0,
            num_images_per_prompt=1,
            generator=generator,
        )[0]
        out_img = np.concatenate(
            [ref_img.resize(args.resolution[::-1], 1),
             pose.resize(args.resolution[::-1], 1),
             gt_img.resize(args.resolution[::-1], 1),
             out], axis=1)
        Image.fromarray(out_img).save(join(args.out_dir, k + '.jpg'))

    print('Done!')
