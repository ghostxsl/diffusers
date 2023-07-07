# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, basename
import random
import numpy as np
from PIL import Image
import torch
import pandas as pd

from diffusers import AutoencoderKL
from diffusers.models.vip.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.vip.flux_image_variation_img2img import FluxImageVariationPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.vip_utils import *
from diffusers.data.vos_client import VOSClient
from diffusers.data.utils import *
from diffusers.data.outer_vos_tools import download_pil_image, upload_pil_image

from aistudio.utils.loader import ROOT_DIR
from aistudio.extensions.HumanPose import HumanPose
from aistudio.extensions.HumanPose.utils import POSE_CONFIG_DIR


torch._dynamo.config.cache_size_limit = 200


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--out_csv",
        default="out_for_tryon.csv",
        type=str)

    parser.add_argument(
        "--resolution",
        type=str,
        default="1200",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )

    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/apps/dat/file/llm/model/FLUX.1-dev",
    )
    parser.add_argument(
        "--transformer_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pt_model_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--rank",
        default=None,
        type=int)
    parser.add_argument(
        "--num_ranks",
        default=None,
        type=int)

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
        default='bf16',
        type=str,
        help="Data type to use (e.g. fp16, fp32, etc.)")

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Whether to compile model."
    )

    args = parser.parse_args()

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


def draw_pose(pose, size=1024, kpt_thr=0.3, canvas=None, draw_hand=True):
    w, h = pose['width'], pose['height']
    if size is not None:
        w = int(size / h * w) if size != h else w
        h = size
    if canvas is None:
        canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    else:
        canvas = np.array(canvas)

    size_ = np.array([w, h])

    kpts = pose['body']['keypoints'][:1, :, :2] * size_
    kpt_valid = pose['body']['keypoints'][:1, :, 2] > kpt_thr
    canvas = draw_bodypose(canvas, kpts, kpt_valid)

    if draw_hand:
        kpts = pose['hand']['keypoints'][:1, :, :2] * size_
        kpt_valid = pose['hand']['keypoints'][:1, :, 2] > kpt_thr
        canvas = draw_handpose(canvas, kpts, kpt_valid)

    return Image.fromarray(canvas)


def long_side_resize(image, size=1200):
    w, h = image.size
    max_l = max(w, h)
    ratio = size / max_l
    new_w = size if w == max_l else int(w * ratio)
    new_h = size if h == max_l else int(h * ratio)
    return image.resize((new_w, new_h), 1)


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
    if img.startswith('http'):
        img = img.replace('https://a.vpimg2.com/', 'http://a-appsimg.vip.vip.com/')
        img = download_pil_image(img)
    else:
        img = args.vos_client.download_vos_pil(img)
    return load_image(img)


def write_to_csv(results, dst):
    df = pd.DataFrame(results)
    df.to_csv(dst, mode='a', index=False, header=not os.path.exists(dst))


def main(args):
    device = args.device
    dtype = args.dtype
    weight_dir = args.weight_dir or join(ROOT_DIR, "weights")
    args.vos_client = VOSClient()

    img_list = load_file(args.input_file)

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        img_list = img_list[args.rank::args.num_ranks]

    pose_infer = HumanPose(
        det_cfg=join(POSE_CONFIG_DIR, "rtmdet_l_8xb32-300e_coco.py"),
        det_pth=join(weight_dir, "extensions/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"),
        bodypose_cfg=join(POSE_CONFIG_DIR, "rtmpose-l_8xb256-420e_body8-256x192.py"),
        bodypose_pth=join(weight_dir,
                          "extensions/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"),
        wholebodypose_cfg=join(POSE_CONFIG_DIR, "dwpose_l_wholebody_384x288.py"),
        wholebodypose_pth=join(weight_dir, "extensions/dw-ll_ucoco_384.pth"),
        device=device,
        bbox_thr=0.25,
    )

    # Load vae
    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae").to(device, dtype=dtype)

    # Load transformer and scheduler
    transformer = FluxTransformer2DModel.from_pretrained(args.transformer_model_path).to(device, dtype=dtype)
    if args.pt_model_path:
        transformer.load_state_dict(
            torch.load(join(args.pt_model_path, "image_variation.bin"), map_location='cpu'), strict=False)

        # transformer._init_image_variation(
        #     joint_attention_dim=transformer.config.in_channels,
        #     pooled_projection_dim=None,
        #     state_dict=torch.load(join(args.pt_model_path, "image_variation.bin"), map_location=torch.device("cpu")),
        #     alter_x_embedder=True,
        # )

        # Load lora weights
        transformer.load_lora_adapter(join(args.pt_model_path, "pytorch_lora_weights.safetensors"))
        transformer.fuse_lora(lora_scale=1.0)
        transformer.unload_lora()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    if args.compile:
        transformer._compile_transformer_block()

    pipe = FluxImageVariationPipeline(
        scheduler=scheduler,
        vae=vae,
        transformer=transformer,
        structural_control=True,
    ).to(device)

    guidance_scales = [1.8, 2.0, 2.2, 2.4]

    for i, line in enumerate(img_list):
        src_url, pose_url = line[0], line[1]
        print(f"{i + 1}/{len(img_list)}: {src_url}")
        ref_img = read_image(args, src_url)
        ref_img = pad_image(ref_img)

        pose_image = read_image(args, pose_url)
        _, pose = pose_infer(pose_image, body=False, hand=False)
        pose_img = draw_pose(pose)
        pose_img = pad_image(pose_img, pad_values=0)

        name = f"{splitext(basename(src_url))[0]}_{splitext(basename(pose_url))[0]}"

        out_urls = []
        for j, cfg in enumerate(guidance_scales):
            out = pipe(
                image=ref_img,
                control_image=pose_img,
                height=args.resolution[0],
                width=args.resolution[1],
                strength=1.0,
                num_inference_steps=30,
                guidance_scale=cfg,
                num_images_per_prompt=1,
            )[0]
            url = upload_pil_image(f"{name}_{j}.jpg", out)
            out_urls.append(url)
        out_data = [{
            'src_img': src_url,
            'pose': pose_url,
            'generate': ";".join(out_urls)
        }]
        write_to_csv(out_data, args.out_csv)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
