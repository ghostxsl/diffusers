# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, basename
import random
import numpy as np
from PIL import Image
import torch

from transformers import Dinov2Model, AutoImageProcessor
from diffusers import AutoencoderKL
from diffusers.models.vip.controlnetxs_flux import FluxControlNetXSModel
from diffusers.pipelines.vip.flux_image_variation_img2img import FluxImageVariationPipeline

from diffusers.models.vip.transformer_flux import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.vip_utils import *
from diffusers.data.vos_client import VOSClient
from diffusers.data.utils import *
from aistudio.data.outer_vos_tools import download_pil_image
from aistudio.utils.loader import ROOT_DIR
from aistudio.extensions.HumanPose import HumanPose
from aistudio.extensions.HumanPose.utils import POSE_CONFIG_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--img_dir",
        default=None,
        type=str,
        help="Directory to reference image.")
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
        "--controlnet_model_path",
        type=str,
        default="/apps/dat/cv/xsl/exp_animate/flux_iv_repose_512_17k/controlnet",
    )
    parser.add_argument(
        "--image_variation_model_path",
        type=str,
        default="/apps/dat/cv/xsl/exp_animate/flux_iv_repose_512_17k",
    )
    parser.add_argument(
        "--image_encoder_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/dinov2-large",
    )
    parser.add_argument(
        "--dinov2_size",
        type=int,
        default=448,
    )
    parser.add_argument(
        "--alter_x_embedder",
        action="store_true",
        help="Whether to alter x_embedder."
    )

    parser.add_argument(
        "--device",
        default='cuda',
        type=str,
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    parser.add_argument(
        "--dtype",
        default='bf16',
        type=str,
        help="Data type to use (e.g. fp16, fp32, etc.)")
    parser.add_argument(
        "--weight_dir",
        default=None,
        type=str,
        help="Directory to weights.")

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
    elif args.dtype == 'bf16':
        args.dtype = torch.bfloat16
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


def read_image(args, img):
    if args.vos_pkl is not None:
        if img.startswith('http'):
            img = img.replace('https://a.vpimg2.com/', 'http://a-appsimg.vip.vip.com/')
            img = download_pil_image(img)
        else:
            img = args.vos_client.download_vos_pil(img)
    return load_image(img)


def main(args):
    device = args.device
    dtype = args.dtype

    if args.vos_pkl is not None:
        img_list = load_file(args.vos_pkl)
        args.vos_client = VOSClient()
    else:
        assert exists(args.img_dir)
        img_list = os.listdir(args.img_dir)
        img_list = [join(args.img_dir, a) for a in img_list if splitext(a)[
            -1].lower() in ['.png', '.jpg', '.jpeg']]
        img_list = sorted(img_list)

    weight_dir = args.weight_dir or join(ROOT_DIR, "weights")
    pose_infer = HumanPose(
        det_cfg=join(POSE_CONFIG_DIR, "rtmdet_l_8xb32-300e_coco.py"),
        det_pth=join(weight_dir, "extensions/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"),
        bodypose_cfg=join(POSE_CONFIG_DIR, "rtmpose-l_8xb256-420e_body8-256x192.py"),
        bodypose_pth=join(weight_dir, "extensions/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"),
        wholebodypose_cfg=join(POSE_CONFIG_DIR, "dwpose_l_wholebody_384x288.py"),
        wholebodypose_pth=join(weight_dir, "extensions/dw-ll_ucoco_384.pth"),
        device=device,
        bbox_thr=0.45,
    )

    # Load vae and image_encoder
    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae").to(device, dtype=dtype)
    image_encoder = Dinov2Model.from_pretrained(
        args.image_encoder_model_path).to(device, dtype=dtype)
    feature_extractor = AutoImageProcessor.from_pretrained(args.image_encoder_model_path)
    feature_extractor.size['shortest_edge'] = args.dinov2_size
    feature_extractor.do_center_crop = False

    # Load transformer and scheduler
    transformer = FluxTransformer2DModel.from_pretrained(
        args.base_model_path, subfolder="transformer").to(device, dtype=dtype)
    transformer._init_image_variation(
        joint_attention_dim=image_encoder.config.hidden_size,
        pooled_projection_dim=image_encoder.config.hidden_size,
        state_dict=torch.load(
            join(args.image_variation_model_path, "image_variation.bin"),
            map_location=torch.device("cpu")),
        alter_x_embedder=args.alter_x_embedder,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    # Load controlnet
    controlnet = FluxControlNetXSModel.from_pretrained(args.controlnet_model_path, torch_dtype=dtype).to(device, dtype=dtype)

    pipe = FluxImageVariationPipeline(
        scheduler=scheduler,
        vae=vae,
        transformer=transformer,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        controlnet=controlnet,
        alter_x_embedder=args.alter_x_embedder
    ).to(device)

    pipe.load_lora_weights(join(args.image_variation_model_path, "pytorch_lora_weights.safetensors"))
    pipe.fuse_lora(lora_scale=1.0)
    pipe.unload_lora_weights()

    for i, file in enumerate(img_list):
        ref_img = read_image(args, file)
        ref_img = pad_image(ref_img)
        ref_img = ref_img.resize(args.resolution[::-1], 1)

        new_img = pad_image(read_image(args, random.choice(img_list)))
        new_img = new_img.resize(args.resolution[::-1], 1)
        pose_img, _ = pose_infer(new_img)
        pose_img = Image.fromarray(pose_img)
        print(f"{i + 1}/{len(img_list)}: {file}")

        out = pipe(
            image=ref_img,
            control_image=pose_img,
            height=args.resolution[0],
            width=args.resolution[1],
            strength=0.96,
            num_inference_steps=30,
            guidance_scale=3.0,
            num_images_per_prompt=1,
            controlnet_conditioning_scale=1.0,
        )[0]
        out_img = np.concatenate(
            [ref_img, pose_img, out], axis=1)
        Image.fromarray(out_img).save(join(args.out_dir, basename(file)))

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
