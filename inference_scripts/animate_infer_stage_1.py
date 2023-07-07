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
from diffusers.data import DrawPose, BoxCrop
from diffusers.data.vos_client import VOSClient


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Path to dataset file.")
    parser.add_argument(
        "--use_vos",
        default=False,
        action="store_true")
    parser.add_argument(
        "--out_dir",
        default="output",
        type=str,
        help="Directory to save.")

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
        "--matting_dir",
        default=None,
        type=str,
        help="Directory to matting image.")

    parser.add_argument(
        "--use_pad",
        default=False,
        action="store_true",
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
        default="/apps/dat/cv/xsl/exp_animate/checkpoint-130000",
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/sd-image-variations/vae",
    )
    parser.add_argument(
        "--image_encoder_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/sd-image-variations/image_encoder",
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
        args.resolution = [int(args.resolution),] * 2
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
_crop = BoxCrop(crop_size=(768, 576))
def get_reference_pose_frame(args, video_list):
    item = random.choice(video_list)

    if not args.use_vos:
        gt_img = load_image(join(args.img_dir, item['image']))
        pose = pkl_load(join(args.pose_dir, item['pose']))
        pose_img = _draw.draw_pose(gt_img.size[::-1], pose)

        if len(video_list) > 1:
            video_list.remove(item)
        ref_item = random.choice(video_list)
        ref_img = load_image(join(args.img_dir, ref_item['image']))
        ref_pose = pkl_load(join(args.pose_dir, ref_item['pose']))
        if args.matting_dir:
            # matting
            label_matting = np.array(load_image(join(args.matting_dir, ref_item['matting'])))
            label_matting = label_matting[..., None].astype('float32') / 255.0
            ref_img = np.array(ref_img).astype('float32')
            bg = np.ones_like(ref_img) * 255.0
            ref_img = ref_img * label_matting + bg * (1 - label_matting)
            ref_img = Image.fromarray(np.clip(ref_img, 0, 255).astype('uint8'))
    else:
        gt_img = load_image(args.vos.download_vos_pil(item['image']))
        pose = args.vos.download_vos_pkl(item['pose'])
        pose_img = _draw.draw_pose(gt_img.size[::-1], pose)

        if len(video_list) > 1:
            video_list.remove(item)
        ref_item = random.choice(video_list)
        ref_img = args.vos.download_vos_pil(ref_item['image'])
        ref_pose = args.vos.download_vos_pkl(ref_item['pose'])
        try:
            # matting
            label_matting = np.array(args.vos.download_vos_pil(ref_item['matting']))
            label_matting = label_matting[..., None].astype('float32') / 255.0
            ref_img = np.array(ref_img).astype('float32')
            bg = np.ones_like(ref_img) * 255.0
            ref_img = ref_img * label_matting + bg * (1 - label_matting)
            ref_img = Image.fromarray(np.clip(ref_img, 0, 255).astype('uint8'))
        except:
            pass

    if args.use_pad:
        ref_img = pad_image(ref_img)
        pose_img = pad_image(pose_img, 0)
        gt_img = pad_image(gt_img)

    gt_img, pose_img = _crop.crop_and_resize(
        gt_img,
        pose['body']['bboxes'],
        cond_img=pose_img,
    )
    ref_img, _ = _crop.crop_and_resize(ref_img, ref_pose['body']['bboxes'])
    return ref_img, pose_img, gt_img


if __name__ == '__main__':
    args = parse_args()
    if args.use_vos:
        args.vos = VOSClient()

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

        ref_img, pose_img, gt_img = get_reference_pose_frame(args, v)
        out = pipe(
            reference_image=ref_img,
            control_image=pose_img,
            height=args.resolution[0],
            width=args.resolution[1],
            num_inference_steps=25,
            guidance_scale=2.0,
            num_images_per_prompt=1,
            generator=generator,
        )[0]
        out_img = np.concatenate(
            [ref_img.resize(args.resolution[::-1], 1),
             pose_img.resize(args.resolution[::-1], 1),
             gt_img.resize(args.resolution[::-1], 1),
             out], axis=1)
        Image.fromarray(out_img).save(join(args.out_dir, k + '.jpg'))

    print('Done!')
