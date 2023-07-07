# Copyright (c) wilson.xu. All rights reserved.
import copy
import os
import argparse
from os.path import join, splitext, exists, basename
import random
import numpy as np
from PIL import Image
import torch

from transformers import Dinov2Model, AutoImageProcessor
from diffusers import AutoencoderKL
from diffusers.models.vip.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.vip.flux_repose_image_variation_img2img import FluxImg2ImgPipeline
from diffusers.models.vip.pose_encoder import PoseEncoder
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.vip_utils import *
from diffusers.data.vos_client import VOSClient
from diffusers.data.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--vos_pkl",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--out_dir",
        default="output",
        type=str,
        help="Directory to save.")

    parser.add_argument(
        "--resolution",
        type=str,
        default="512",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )

    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/apps/dat/cv/wty/models/flux/FLUX.1-dev",
    )
    parser.add_argument(
        "--pt_model_path",
        type=str,
        default="/apps/dat/cv/xsl/exp_animate/checkpoint-5000",
    )
    parser.add_argument(
        "--image_encoder_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/dinov2-large",
    )
    parser.add_argument(
        "--dinov2_size",
        type=int,
        default=224,
    )

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


def draw_pose_(pose, size, kpt_thr=0.3, canvas=None, draw_hand=True):
    w, h = pose['width'], pose['height']
    w = int(size / h * w) if size != h else w
    h = size
    if canvas is None:
        canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    else:
        canvas = np.array(canvas)

    size_ = np.array([w, h])

    kpts = pose['body']['keypoints'][..., :2] * size_
    kpt_valid = pose['body']['keypoints'][..., 2] > kpt_thr
    canvas = draw_bodypose(canvas, kpts, kpt_valid)

    if draw_hand:
        kpts = pose['hand']['keypoints'][..., :2] * size_
        kpt_valid = pose['hand']['keypoints'][..., 2] > kpt_thr
        canvas = draw_handpose(canvas, kpts, kpt_valid)

    return Image.fromarray(canvas)


def get_group_image(args, group_list, size):
    item = random.choice(group_list)
    gt_img = args.vos_client.download_vos_pil(item["image"])
    gt_img = load_image(gt_img)

    gt_pose = args.vos_client.download_vos_pkl(item["pose"])
    pose_img = draw_pose_(gt_pose, size=size)

    if len(group_list) > 1:
        group_list.remove(item)
        item = random.choice(group_list)

        ref_img = args.vos_client.download_vos_pil(item["image"])
        ref_img = load_image(ref_img)
    else:
        ref_img = gt_img

    ref_img = pad_image(ref_img).resize((size, size), 1)
    pose_img = pad_image(pose_img).resize((size, size), 1)
    gt_img = pad_image(gt_img).resize((size, size), 1)

    return ref_img, pose_img, gt_img


def main(args):
    device = args.device
    dtype = args.dtype

    img_dict = load_file(args.vos_pkl)
    args.vos_client = VOSClient()

    # Load vae and image_encoder
    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae").to(device, dtype=dtype)
    image_encoder = Dinov2Model.from_pretrained(
        args.image_encoder_model_path).to(device, dtype=dtype)
    feature_extractor = AutoImageProcessor.from_pretrained(args.image_encoder_model_path)
    feature_extractor.size['shortest_edge'] = args.dinov2_size
    feature_extractor.do_center_crop = False

    # Load transformer and scheduler
    transformer = FluxTransformer2DModel.from_pretrained(
        args.base_model_path, subfolder="transformer")
    transformer._init_image_variation(
        joint_attention_dim=image_encoder.config.hidden_size,
        pooled_projection_dim=image_encoder.config.hidden_size,
        state_dict=torch.load(
            join(args.pt_model_path, "image_variation.bin"),
            map_location=torch.device("cpu")),
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    transformer = transformer.to(device, dtype=dtype)

    pose_encoder = PoseEncoder.from_pretrained(join(args.pt_model_path, "pose_encoder"))

    pipe = FluxImg2ImgPipeline(
        scheduler=scheduler,
        vae=vae,
        transformer=transformer,
        pose_encoder=pose_encoder,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        x_embedder=False,
    ).to(device)

    pipe.load_lora_weights(join(args.pt_model_path, "pytorch_lora_weights.safetensors"))
    pipe.fuse_lora(lora_scale=1.0)
    pipe.unload_lora_weights()

    for i, (k, v) in enumerate(img_dict.items()):
        print(f"{i + 1}/{len(img_dict)}: {k}")

        ref_img, pose_img, gt_img = get_group_image(args, copy.deepcopy(v), size=args.resolution[0])

        seed = get_fixed_seed(-1)
        generator = get_torch_generator(seed)

        out = pipe(
            image=ref_img,
            height=args.resolution[0],
            width=args.resolution[1],
            strength=1.0,
            num_inference_steps=30,
            guidance_vector=1.0,
            guidance_scale=1.0,
            num_images_per_prompt=1,
            generator=generator,
            control_image=pose_img,
            controlnet_conditioning_scale=1.0,
        )[0]
        out_img = np.concatenate(
            [ref_img, pose_img, out], axis=1)
        Image.fromarray(out_img).save(join(args.out_dir, k + '.jpg'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
