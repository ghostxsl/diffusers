# Copyright (c) wilson.xu. All rights reserved.
import os
import argparse
from os.path import join, splitext, exists, basename
import numpy as np
from PIL import Image
import torch

from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
    CLIPTextModel,
    T5EncoderModel,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    Dinov2Model,
    AutoImageProcessor,
)
from diffusers import AutoencoderKL
from diffusers.models.vip.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.vip.flux_ipa_img2img import FluxImg2ImgPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.utils.vip_utils import *
from diffusers.data.vos_client import VOSClient
from diffusers.data.utils import *
from diffusers.data.outer_vos_tools import download_pil_image


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
        "--prompt_file",
        default=None,
        type=str,
        help="File path to prompt.")
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
        default="/apps/dat/cv/xsl/exp_animate/flux_ipa_dino_512_25k/ip_adapter/ip_adapter_plus.bin",
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_encoder_model_path",
        type=str,
        # default="/apps/dat/cv/xsl/weights/Kolors-diffusers/image_encoder",
        default="/apps/dat/cv/xsl/weights/dinov2-large",
    )
    parser.add_argument(
        "--use_dinov2",
        action="store_true",
        help="Whether to use dinov2 for image encoder."
    )
    parser.add_argument(
        "--dinov2_size",
        type=int,
        default=448,
    )
    parser.add_argument(
        "--alter_x_embedder",
        action="store_true",
        help="Whether to concat `image` for input."
    )
    parser.add_argument(
        "--use_self_attn",
        action="store_true",
        help="Whether to use self_attn for ip-adapter."
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

    # Load vae and image_encoder
    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae").to(device, dtype=dtype)
    if args.use_dinov2:
        image_encoder = Dinov2Model.from_pretrained(
            args.image_encoder_model_path).to(device, dtype=dtype)
        feature_extractor = AutoImageProcessor.from_pretrained(args.image_encoder_model_path)
        feature_extractor.size['shortest_edge'] = args.dinov2_size
        feature_extractor.do_center_crop = False
    else:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.image_encoder_model_path).to(device, dtype=dtype)
        feature_extractor = CLIPImageProcessor.from_pretrained(args.image_encoder_model_path)

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
    transformer = FluxTransformer2DModel.from_pretrained(
        args.base_model_path, subfolder="transformer").to(device, dtype=dtype)
    transformer._init_ip_adapter_plus(
        num_image_text_embeds=0,
        state_dict=torch.load(args.ip_adapter_model_path, map_location=torch.device("cpu")),
        use_self_attn=args.use_self_attn,
        alter_x_embedder=args.alter_x_embedder,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    dpm_scheduler = DPMSolverMultistepScheduler(
        algorithm_type="dpmsolver++", prediction_type="flow_prediction", use_dynamic_shifting=True)

    pipe = FluxImg2ImgPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        transformer=transformer,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        use_dinov2=args.use_dinov2,
        use_self_attn=args.use_self_attn,
        alter_x_embedder=args.alter_x_embedder,
    )
    # pipe.transformer.set_ip_adapter_scale(0.75)
    if args.lora_model_path is not None:
        pipe.load_lora_weights(args.lora_model_path)
        pipe.fuse_lora(lora_scale=1.0)
        pipe.unload_lora_weights()

    if args.prompt_file is not None:
        prompt_dict = load_file(args.prompt_file)

    for i, file in enumerate(img_list):
        ref_img = read_image(args, file)
        ref_img = pad_image(ref_img)
        print(f"{i + 1}/{len(img_list)}: {file}")
        seed = get_fixed_seed(-1)
        generator = get_torch_generator(seed)

        prompt = None
        if args.prompt_file is not None:
            # prompt = prompt_dict[basename(file)]['prompt_full']
            prompt_info = prompt_dict[basename(file)]
            gender = prompt_info["gender"]
            prompt = f"Asian {gender} posing for a photo"

        out = pipe(
            prompt=prompt,
            negative_prompt=None,
            image=ref_img,
            ip_adapter_image=ref_img,
            height=args.resolution[0],
            width=args.resolution[1],
            strength=1.0,
            num_inference_steps=25,
            guidance_vector=1.0,
            guidance_scale=1.0,
            num_images_per_prompt=1,
            generator=generator,
            max_sequence_length=128,
        )[0]
        out_img = np.concatenate(
            [ref_img.resize(args.resolution[::-1], 1), out], axis=1)
        Image.fromarray(out_img).save(join(args.out_dir, basename(file)))

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done!')
