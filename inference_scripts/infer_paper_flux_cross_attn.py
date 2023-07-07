# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, basename
import numpy as np
from PIL import Image
import torch

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from diffusers import AutoencoderKL
from diffusers.models.vip.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.vip.flux_vip_paper_pipeline import FluxVIPPaperPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.vip_utils import *
from diffusers.data.vos_client import VOSClient
from diffusers.data.utils import *
from diffusers.data.outer_vos_tools import download_pil_image


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
        "--ip_adapter_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_encoder_model_path",
        type=str,
        default="/apps/dat/cv/xsl/weights/Kolors-diffusers/image_encoder",
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
        "--device",
        default='cuda',
        type=str,
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    parser.add_argument(
        "--dtype",
        default='bf16',
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
    elif args.dtype == 'bf16':
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float32

    return args


def draw_pose(pose, size=512, kpt_thr=0.3, canvas=None, draw_hand=True):
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
    args.vos_client = VOSClient()

    img_dict = load_file(args.vos_pkl)

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        done_list = os.listdir(args.out_dir)
        done_list = set([splitext(a)[0] for a in done_list])
        new_dict = {k: v for k, v in img_dict.items() if k not in done_list}
        img_dict = new_dict
        total_num = len(img_dict)
        stride = int(total_num / args.num_ranks)

        start_idx = stride * args.rank
        end_idx = stride * (args.rank + 1) if args.rank + 1 < args.num_ranks else total_num

        img_keys = sorted(list(img_dict.keys()))
        img_dict = {img_keys[i]: img_dict[img_keys[i]] for i in range(start_idx, end_idx)}

    # Load vae and image_encoder
    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae").to(device, dtype=dtype)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.image_encoder_model_path).to(device, dtype=dtype)
    feature_extractor = CLIPImageProcessor.from_pretrained(args.image_encoder_model_path)

    # Load empty text prompt_embeds
    empty_prompt_embeds = torch.load(join(args.base_model_path, "empty_prompt_embeds"))
    prompt_embeds = empty_prompt_embeds["prompt_embeds_512"]
    pooled_prompt_embeds = empty_prompt_embeds["pooled_prompt_embeds"]

    prompt_embeds = prompt_embeds.to(device, dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device, dtype=dtype)

    # Load transformer
    transformer = FluxTransformer2DModel.from_pretrained(
        args.base_model_path, subfolder="transformer").to(device, dtype=dtype)
    if args.lora_model_path is not None:
        transformer.load_lora_adapter(args.lora_model_path)
        transformer.fuse_lora(lora_scale=1.0)
        transformer.unload_lora()

    transformer._init_ip_adapter_plus(
        embed_dims=image_encoder.config.hidden_size,
        num_attention_heads=image_encoder.config.num_attention_heads,
        state_dict=torch.load(args.ip_adapter_model_path, map_location=torch.device("cpu")),
        alter_x_embedder=True,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")

    pipe = FluxVIPPaperPipeline(
        scheduler=scheduler,
        vae=vae,
        transformer=transformer,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
    )

    for i, (k, v) in enumerate(img_dict.items()):
        ref_item, gt_item = v[0], v[1]
        ref_img = read_image(args, ref_item['image'])
        gt_img = read_image(args, gt_item['image'])
        gt_pose = args.vos_client.download_vos_pkl(gt_item['pose'])
        pose_img = draw_pose(gt_pose, args.resolution[0])

        print(f"{i + 1}/{len(img_dict)}: {k}")
        seed = get_fixed_seed(-1)
        generator = get_torch_generator(seed)

        out = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            image=ref_img,
            control_image=pose_img,
            height=args.resolution[0],
            width=args.resolution[1],
            strength=1.0,
            num_inference_steps=20,
            guidance_vector=3.5,
            num_images_per_prompt=1,
            generator=generator,
        )[0]
        out_img = np.concatenate(
            [ref_img.resize(args.resolution[::-1], 1),
             gt_img.resize(args.resolution[::-1], 1), out], axis=1)
        Image.fromarray(out_img).save(join(args.out_dir, k + '.png'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
