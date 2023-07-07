# Copyright (c) wilson.xu. All rights reserved.
import argparse
import os
from os.path import join, splitext, basename, exists
import random
import torch
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
import decord
import numpy as np
from PIL import Image

from diffusers import AutoencoderKLWan
from diffusers.data.utils import load_file, json_save
from diffusers.data.vos_client import VOSClient


decord.bridge.set_bridge("torch")


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file",
        default="/apps/dat/cv/zjn/data/videos/train_100/train_100.json",
        type=str)
    parser.add_argument(
        "--save_dir",
        default="/apps/dat/cv/zjn/data/videos/train_100/latents",
        type=str,
        help="Directory to save.")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="/home/llms/Wan-FLF2V-14B-720P-dif",
    )
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=49,
    )
    parser.add_argument(
        "--i2v",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_vos",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--vos_bucket",
        type=str,
        default="public",
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

    args.device = torch.device(args.device)
    if args.dtype == 'fp16':
        args.dtype = torch.float16
    elif args.dtype == 'bf16':
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float32

    return args


@torch.no_grad()
def encode_vae_image(
        pixels: torch.Tensor,
        vae: torch.nn.Module,
        latents_mean,
        latents_std,
        weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.mode()
    pixel_latents = (pixel_latents - latents_mean) * latents_std
    return pixel_latents.to(weight_dtype)


def aspect_ratio_resize(frames, img_size=(960, 720), mod_value=16):
    h, w = frames.shape[1:3]
    aspect_ratio = h / w
    max_area = img_size[0] * img_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    frames = frames.permute([0, 3, 1, 2]).to(torch.float32)
    frames = torch.nn.functional.interpolate(frames, size=[height, width], mode="bilinear")
    frames = frames / 255.0 * 2.0 - 1.0
    return frames


def get_video_images(args, item, model_fps=16, vae_scale_factor_temporal=4):
    video_reader = decord.VideoReader(item['file_path'])
    video_length = len(video_reader)
    video_fps = int(round(video_reader.get_avg_fps()))
    # 视频帧的采样步长
    video_sample_stride = int(round(video_fps / model_fps))

    batch_index_list = []
    video_sample_length = int(video_length // video_sample_stride)
    nums = video_sample_length / args.max_num_frames
    if nums < 1.33:
        video_sample_length = min(video_sample_length, args.max_num_frames)
        clip_length = (video_sample_length - 1) * video_sample_stride + 1
        batch_index = np.linspace(0, clip_length - 1, video_sample_length, dtype=int)

        sample_video_length = len(batch_index)
        sample_video_length = (sample_video_length - 1) // vae_scale_factor_temporal
        sample_video_length = sample_video_length * vae_scale_factor_temporal + 1
        batch_index = batch_index[:sample_video_length]
        batch_index_list.append(batch_index)
    else:
        nums = int(nums) + 1 if nums % 1 >= 0.33 else int(nums)

        sample_length = (args.max_num_frames - 1) * video_sample_stride
        for i in range(nums):
            start = i * sample_length
            end = (i + 1) * sample_length
            if end > video_length - 1:
                end = video_length - 1
                start = video_length - 1 - sample_length

            batch_index = np.linspace(start, end, args.max_num_frames, dtype=int)
            batch_index_list.append(batch_index)

    video_list = []
    pixel_values_list = []
    images_list = []
    for batch_index in batch_index_list:
        video = video_reader.get_batch(batch_index)
        video = aspect_ratio_resize(video)
        video = video.permute([1, 0, 2, 3])
        video_list.append(video.unsqueeze(0))

        if args.i2v:
            pixel_values = torch.cat(
                [video[:, 0:1], video.new_zeros(video.shape[0], video.shape[1] - 1, video.shape[2], video.shape[3])],
                dim=1,
            )
        else:
            pixel_values = torch.cat(
                [video[:, 0:1], video.new_zeros(video.shape[0], video.shape[1] - 2, video.shape[2], video.shape[3]),
                 video[:, -1:]],
                dim=1,
            )
        pixel_values_list.append(pixel_values.unsqueeze(0))

        if args.i2v:
            image = video[:, 0].permute([1, 2, 0])
            image = Image.fromarray(np.uint8((image + 1) / 2 * 255))
            images_list.append(image)
        else:
            images = [video[:, 0], video[:, -1]]
            images = [a.permute([1, 2, 0]) for a in images]
            images = [Image.fromarray(np.uint8((a + 1) / 2 * 255)) for a in images]
            images_list.append(images)

    return video_list, pixel_values_list, images_list


def main(args):
    if not args.use_vos:
        os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device)
    dtype = args.dtype
    args.vos_client = VOSClient(args.vos_bucket)

    # Load the vae
    vae = AutoencoderKLWan.from_pretrained(args.pretrained_model_path, subfolder="vae")
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(device, dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        device, dtype)
    vae_scale_factor_temporal = 4
    vae.requires_grad_(False)
    vae.to(device, dtype=dtype)

    # Load image processor and image encoder
    image_processor = CLIPImageProcessor.from_pretrained(args.pretrained_model_path,
                                                         subfolder="image_processor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_path,
                                                                  subfolder="image_encoder")

    image_encoder.requires_grad_(False)
    image_encoder.to(device, dtype=dtype)

    # load data list
    img_list = load_file(args.input_file)

    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks

        img_list = img_list[args.rank::args.num_ranks]

    out_list = []
    for item in tqdm(img_list):
        try:
            video_list, pixel_values_list, images_list = get_video_images(args, item)

            for i, (video, pixel_values, images) in enumerate(zip(video_list, pixel_values_list, images_list)):
                with torch.no_grad():
                    video = video.to(device, dtype=dtype)
                    video_latents = encode_vae_image(
                        video, vae, latents_mean, latents_std, dtype)

                    images = image_processor(images, return_tensors="pt").pixel_values
                    images = images.to(device, dtype=dtype)
                    image_embeds = image_encoder(
                        images, output_hidden_states=True).hidden_states[-2]

                    pixel_values = pixel_values.to(device, dtype=dtype)
                    condition = encode_vae_image(
                        pixel_values, vae, latents_mean, latents_std, dtype)

                # make mask
                num_latent_frames, latent_height, latent_width = video_latents.shape[2:]
                num_frames = (num_latent_frames - 1) * vae_scale_factor_temporal + 1
                mask_lat_size = torch.ones(1, 1, num_frames, latent_height, latent_width)
                if args.i2v:
                    mask_lat_size[:, :, 1:] = 0
                else:
                    mask_lat_size[:, :, 1: -1] = 0
                first_frame_mask = mask_lat_size[:, :, 0:1]
                first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2,
                                                           repeats=vae_scale_factor_temporal)
                mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
                mask_lat_size = mask_lat_size.view(1, -1, vae_scale_factor_temporal, latent_height,
                                                   latent_width)
                mask_lat_size = mask_lat_size.transpose(1, 2)
                mask_lat_size = mask_lat_size.to(device, dtype=dtype)

                save_name = splitext(basename(item['file_path']))[0] + f'_clip_{i}.pt'
                latents = {
                    "latents": video_latents.cpu(),
                    "image_embeds": image_embeds.cpu(),
                    "mask": mask_lat_size.cpu(),
                    "condition": condition.cpu(),
                }
                if not args.use_vos:
                    torch.save(latents, join(args.save_dir, save_name))
                else:
                    args.vos_client.upload_vos_pt(latents, join(args.save_dir, save_name))
                    out_list.append(join(args.save_dir, save_name))

        except Exception as e:
            print(e)

    if args.use_vos:
        json_save(out_list, "vos_list.json")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
