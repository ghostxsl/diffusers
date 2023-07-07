# Copyright (c) wilson.xu. All rights reserved.
import argparse
import random
import os
from os.path import join, splitext, basename
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision.transforms import CenterCrop

from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
    CLIPTextModel,
    T5EncoderModel,
)
from diffusers import AutoencoderKL
from diffusers.models.vip.transformer_flux import FluxTransformer2DModel
from diffusers.models.vip.controlnetxs_flux import FluxControlNetXSModel
from diffusers.pipelines.vip.pipeline_flux_inpaint import FluxInpaintPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.vip_utils import *
from diffusers.data.vos_client import VOSClient
from diffusers.data.utils import *
from diffusers.data.outer_vos_tools import download_pil_image


gender_map = {
        "男性": "man",
        "女性": "woman",
        "男童": "boy",
        "女童": "girl",
        "man": "man",
        "woman": "woman",
        "boy": "boy",
        "girl": "girl",
        "male": "man",
        "female": "woman",
}


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
        default="896x672",
        help=(
            "The resolution(h x w) for input images, all the images"
            " will be resized to this resolution."
        ),
    )
    parser.add_argument(
        "--num_images",
        type=str,
        default="1x2",
    )
    parser.add_argument(
        "--inpainting",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--pose_lib",
        default=None,
        type=str,
        help="Path to pose lib.")
    parser.add_argument(
        "--gpt4v",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--llava",
        action="store_true",
        default=False,
    )

    # Model Path
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/apps/dat/cv/wty/models/flux/FLUX.1-dev",
    )
    parser.add_argument(
        "--transformer_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--controlnet_model_path",
        type=str,
        default=None,
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

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if len(args.resolution.split("x")) == 1:
        args.resolution = [int(args.resolution),] * 2
    elif len(args.resolution.split("x")) == 2:
        args.resolution = [int(r) for r in args.resolution.split("x")]
    else:
        raise Exception(f"Error `resolution` type({type(args.resolution)}): {args.resolution}.")

    if len(args.num_images.split("x")) == 1:
        args.num_images = [1, int(args.num_images)]
    elif len(args.num_images.split("x")) == 2:
        args.num_images = [int(r) for r in args.num_images.split("x")]
    else:
        raise Exception(f"Error `num_images` type({type(args.num_images)}): {args.num_images}.")

    args.device = torch.device(args.device)
    if args.dtype == 'fp16':
        args.dtype = torch.float16
    elif args.dtype == 'bf16':
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float32

    args.vos_client = VOSClient()
    args.centercrop = CenterCrop(args.resolution)
    if args.pose_lib is not None:
        args.pose_lib = load_file(args.pose_lib)

    return args


def draw_pose(body_pose, draw_size, hand_pose=None, face_pose=None, kpt_thr=0.3):
    height, width = draw_size
    canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    canvas = draw_bodypose(canvas, body_pose[..., :2], body_pose[..., 2] > kpt_thr)

    if hand_pose is not None:
        canvas = draw_handpose(canvas, hand_pose[..., :2], hand_pose[..., 2] > kpt_thr)
    if face_pose is not None:
        canvas = draw_facepose(canvas, face_pose[..., :2], face_pose[..., 2] > kpt_thr)

    return Image.fromarray(canvas)


def random_pose_image(pose_lib, resolution=(768, 576), num_images=(1, 2), gender="person"):
    pose_idx = random.sample(range(len(pose_lib["body"][gender])), num_images[0] * num_images[1])
    out_img = []
    for i in range(num_images[0]):
        row_img = []
        for j in range(num_images[1]):
            body_pose = pose_lib["body"][gender][pose_idx[i * num_images[1] + j]]
            body_pose[..., :2] /= np.array([576, 768])
            body_pose[..., :2] *= np.array(resolution[::-1])

            hand_pose = pose_lib["hand"][gender][pose_idx[i * num_images[1] + j]]
            hand_pose[..., :2] /= np.array([576, 768])
            hand_pose[..., :2] *= np.array(resolution[::-1])

            row_img.append(draw_pose(body_pose[None], resolution, hand_pose))
        row_img = np.concatenate(row_img, axis=1)
        out_img.append(row_img)
    out_img = np.concatenate(out_img, axis=0)

    return Image.fromarray(out_img)


def pad_image(img, pad_values=(255, 255, 255), size=None):
    w, h = img.size
    img = np.array(img)
    pad_border = [[0, 0], [0, 0], [0, 0]]
    if size is not None:
        # padding成指定size
        assert isinstance(size, (list, tuple)) and len(size) == 2
        oh, ow = size

        if oh > h:
            pad_ = oh - h
            pad_border[0] = [pad_ // 2, pad_ - pad_ // 2]
        if ow > w:
            pad_ = ow - w
            pad_border[1] = [pad_ // 2, pad_ - pad_ // 2]

        if pad_values == 'border':
            img = cv2.copyMakeBorder(
                img,
                pad_border[0][0], pad_border[0][1],
                pad_border[1][0], pad_border[1][1],
                cv2.BORDER_REPLICATE)
        else:
            img = cv2.copyMakeBorder(
                img,
                pad_border[0][0], pad_border[0][1],
                pad_border[1][0], pad_border[1][1],
                cv2.BORDER_CONSTANT, value=pad_values)
    else:
        # padding成1:1方图
        if w > h:
            pad_ = w - h
            pad_border = ((pad_ // 2, pad_ - pad_ // 2), (0, 0), (0, 0))
        elif h > w:
            pad_ = h - w
            pad_border = ((0, 0), (pad_ // 2, pad_ - pad_ // 2), (0, 0))

        if pad_values == 'border':
            img = cv2.copyMakeBorder(
                img,
                pad_border[0][0], pad_border[0][1],
                pad_border[1][0], pad_border[1][1],
                cv2.BORDER_REPLICATE)
        else:
            img = cv2.copyMakeBorder(
                img,
                pad_border[0][0], pad_border[0][1],
                pad_border[1][0], pad_border[1][1],
                cv2.BORDER_CONSTANT, value=pad_values)

    return Image.fromarray(img), pad_border


def resize_image(args, img, size):
    w, h = img.size
    oh, ow = size
    ow = int(oh / h * w)
    img = img.resize((ow, oh), 1)

    if img.width > size[1]:
        img = args.centercrop(img)
    else:
        img, _ = pad_image(img, size=size)

    return img


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

    img_dict = load_file(args.vos_pkl)
    if args.gpt4v:
        img_dict = random.sample(img_dict, k=100)
        img_dict = {a[0]: eval(a[1]) for a in img_dict}
    elif args.llava:
        # img_dict = random.sample(img_dict, k=100)
        img_dict = {a[0]: eval(a[1])['result'] for a in img_dict}

    # Load vae and image_encoder
    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae").to(device, dtype=dtype)

    # Load the tokenizer
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.base_model_path, subfolder="tokenizer")
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.base_model_path, subfolder="tokenizer_2")

    # import correct text encoder class
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.base_model_path, subfolder="text_encoder").to(device, dtype=dtype)
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.base_model_path, subfolder="text_encoder_2").to(device, dtype=dtype)

    # Load transformer and scheduler
    if args.transformer_model_path:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.transformer_model_path).to(device, dtype=dtype)
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.base_model_path, subfolder="transformer").to(device, dtype=dtype)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")

    controlnet = None
    if args.controlnet_model_path is not None and args.pose_lib is not None:
        controlnet = FluxControlNetXSModel.from_pretrained(args.controlnet_model_path).to(device, dtype=dtype)

    pipe = FluxInpaintPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        transformer=transformer,
        controlnet=controlnet,
    )
    if args.lora_model_path is not None:
        pipe.load_lora_weights(args.lora_model_path)
        pipe.fuse_lora(lora_scale=1.0)
        pipe.unload_lora_weights()

    candidate_prompt = [
        "Standing posture, left hand on the shoulder, right hand slightly raised and touching the cloth.",
        "Standing posture, the person is standing with their hands naturally hanging down at their sides.",
        "Standing posture, hands in pockets, body slightly turned to the left.",
        "Standing posture, left hand touching the hair.",
        "Standing posture, left hand making a V sign, right hand touching hair.",
        "Standing posture, right hand making a V sign gesture.",
        "Standing posture, left hand raised to head level, right hand down by the side.",
        "Standing posture, left hand lightly touching the right arm which is bent at the elbow.",
        "Standing posture, arms crossed in front of chest.",
        "Standing posture, body slightly turned to the side, left hand naturally hanging down, right hand slightly bent resting on the waist.",
        "Seated posture, right hand touching hair.",
        "Seated posture, right hand touching the chin.",
        "Seated posture, left hand resting on the thigh, right hand making a V sign gesture.",

        "Walking posture, left leg in mid-step, looking downward to the left, hair covering part of the face.",
        "Standing posture, left hand on hip, right arm hanging by the side.",
        "Standing posture, left hand touching hair.",
        "Standing posture, left hand raised to chest level, right hand down by the side.",
        "Standing posture, body slightly turned to the side, head slightly tilted downward, hands clasped together in front of chest.",
        "Standing posture, body slightly turned to the side, left hand holding a bag down.",
        "Standing posture, left leg bent and lifted, right hand holding a bag down.",
        "Seated posture, left leg crossed over right, left hand resting on the thigh.",
        "Seated on the floor, left arm extended outward, legs crossed at the ankles.",
        "Seated posture, left leg crossed over right, right hand resting on right leg",

        "Standing posture, upper-body photograph.",
        "Seated posture, upper-body photograph.",
        # "Standing posture, full-body photograph.",
    ]
    num_images = args.num_images or [1, 2]
    global_prompt = "[Pose Transition], This group of images illustrates the pose transitions of the same person wearing the same outfit in similar scenes."
    for i, (k, v) in enumerate(img_dict.items()):
        save_name = f"{splitext(basename(k))[0]}.jpg" if args.gpt4v or args.llava else f"{k}.jpg"
        print(f"{i + 1}/{len(img_dict)}: {save_name}")

        if args.gpt4v:
            prompt_2 = f"This group of images illustrates the {v['Gender']}, {v['Character']}; {v['Background']};"
            prompt_2 += f" [IMAGE1] {v['Posture']};"
            candidate_ = random.sample(candidate_prompt, k=num_images[0] * num_images[1] - 1)
            for j in range(1, num_images[0] * num_images[1]):
                prompt_2 += f" [IMAGE{j + 1}] {candidate_[j - 1]}"

            item1 = {'image': k}
        elif args.llava:
            prompt_2 = f"This group of images illustrates the {v['gender']}, {v['group']}, {v['clothes']}; {v['background']};"
            prompt_2 += f" [IMAGE1] {v['posture']};"
            candidate_ = random.sample(candidate_prompt, k=num_images[0] * num_images[1] - 1)
            for j in range(1, num_images[0] * num_images[1]):
                prompt_2 += f" [IMAGE{j + 1}] {candidate_[j - 1]}"

            item1 = {'image': v['image_url']}
        else:
            item1 = random.choice(v)
            prompt_2 = f"This group of images illustrates the {item1['llava']['gender']}, {item1['llava']['group']}, {item1['llava']['clothes']}; {item1['llava']['background']};"
            for j, idx in enumerate(random.sample(range(len(v)), num_images[0] * num_images[1])):
                if j == 0:
                    item = item1
                else:
                    item = v[idx]
                prompt_2 += f" [IMAGE{j + 1}] {item['llava']['posture']};"

            # prompt_2 = f"This group of images illustrates {item1['gpt_4v']['Character']}; {item1['gpt_4v']['Background']};"
            # for j, idx in enumerate(random.sample(range(len(v)), num_images[0] * num_images[1])):
            #     if j == 0:
            #         item = item1
            #     else:
            #         item = v[idx]
            #     prompt_2 += f" [IMAGE{j + 1}] {item['gpt_4v']['Posture']};"

        if args.inpainting:
            img1 = read_image(args, item1['image'])
            size = [args.resolution[0], args.resolution[1]]
            img1 = resize_image(args, img1, size)
            img1 = np.array(img1)
            img1 = np.concatenate([img1,] + [np.zeros_like(img1),] * (num_images - 1), axis=1)
            img1 = Image.fromarray(img1)

            mask_image = np.concatenate(
                [np.zeros(size, dtype=np.uint8),] + [np.ones(size, dtype=np.uint8) * 255,] * (num_images - 1), axis=1)
            mask_image = Image.fromarray(mask_image)

        if controlnet is not None:
            pose_image = random_pose_image(
                args.pose_lib, args.resolution, num_images,
                gender=gender_map.get(item1['llava']['gender'].lower(), "person"))

        # img1 = read_image(args, item1['image'])
        # size = [args.resolution[0], args.resolution[1]]
        # img1 = resize_image(args, img1, size)
        # img1 = np.concatenate([img1, img1], axis=1)
        # img1 = np.concatenate([img1, img1], axis=0)
        # img1 = Image.fromarray(img1)

        out_img = pipe(
            prompt=global_prompt,
            prompt_2=prompt_2,
            image=img1 if args.inpainting else None,
            mask_image=mask_image if args.inpainting else None,
            control_image=pose_image if controlnet is not None else None,
            height=args.resolution[0],
            width=args.resolution[1],
            num_images=num_images,
            num_inference_steps=40,
            guidance_scale=3.5,
            strength=1.0,
            num_images_per_prompt=1,
            generator=torch.Generator("cpu").manual_seed(42),
            max_sequence_length=512,
        )[0]

        if controlnet is not None:
            out_img = np.concatenate([out_img, pose_image], axis=0)
            Image.fromarray(out_img).save(join(args.out_dir, save_name))
        else:
            out_img.save(join(args.out_dir, save_name))


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
