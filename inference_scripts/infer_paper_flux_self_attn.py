# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, basename
import numpy as np
from PIL import Image
import torch

from diffusers import AutoencoderKL
from diffusers.models.vip.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.vip.flux_vip_paper_pipeline import FluxVIPPaperPipeline
from diffusers import FluxPriorReduxPipeline
from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
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
        "--redux_model_path",
        type=str,
        default="/apps/dat/file/llm/model/FLUX.1-Redux-dev",
    )
    parser.add_argument(
        "--image_embedder_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_variation_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lora_model_path",
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

    # Load vae
    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae").to(device, dtype=dtype)

    # Load image_embedder and pipeline_prior_redux
    image_embedder = ReduxImageEncoder.from_pretrained(args.image_embedder_model_path).to(device, dtype=dtype)
    pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
        args.redux_model_path, image_embedder=image_embedder, torch_dtype=dtype).to(device)

    # Load transformer
    transformer = FluxTransformer2DModel.from_pretrained(
        args.base_model_path, subfolder="transformer").to(device, dtype=dtype)

    # Load lora weights
    if args.lora_model_path is not None:
        transformer.load_lora_adapter(args.lora_model_path)
        transformer.fuse_lora(lora_scale=1.0)
        transformer.unload_lora()
    transformer._init_fill_x_embedder(
        state_dict=torch.load(args.image_variation_model_path, map_location='cpu'))

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")

    pipe = FluxVIPPaperPipeline(
        scheduler=scheduler,
        vae=vae,
        transformer=transformer,
    ).to(device)

    for i, (k, v) in enumerate(img_dict.items()):
        ref_item, gt_item = v[0], v[1]
        ref_img = read_image(args, ref_item['image'])
        gt_img = read_image(args, gt_item['image'])
        gt_img = gt_img.resize(args.resolution[::-1], 1)

        gt_pose = args.vos_client.download_vos_pkl(gt_item['pose'])
        pose_img = draw_pose(gt_pose, args.resolution[0])

        print(f"{i + 1}/{len(img_dict)}: {k}")
        seed = get_fixed_seed(-1)
        generator = get_torch_generator(seed)

        prompt_embeds, pooled_prompt_embeds = pipe_prior_redux(ref_img, return_dict=False)

        out = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            image=ref_img,
            control_image=pose_img,
            height=args.resolution[0],
            width=args.resolution[1],
            strength=1.0,
            num_inference_steps=30,
            guidance_vector=2.5,
            num_images_per_prompt=1,
            generator=generator,
            max_sequence_length=512 + 729,
        )[0]
        out_img = np.concatenate(
            [ref_img.resize(args.resolution[::-1], 1), gt_img, out], axis=1)
        Image.fromarray(out_img).save(join(args.out_dir, k + '.png'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
