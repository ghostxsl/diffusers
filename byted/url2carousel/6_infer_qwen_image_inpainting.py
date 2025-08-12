import os
import argparse
from os.path import join, splitext, basename
from tqdm import tqdm
import random
import torch
import numpy as np
from PIL import Image

from diffusers.pipelines.byted.pipeline_qwenimage_t2i_edit import QwenImageT2IEditPipeline
from diffusers.data.utils import load_file, json_save
from diffusers.data.outer_vos_tools import load_or_download_image
from diffusers.data.byted.tos import _gen_name
from diffusers.data.byted.clients.creative_ai_capability import image_subject_seg


aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472)
}
width, height = aspect_ratios["9:16"]

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": "，超清，4K，电影级构图", # for chinese prompt
}
negative_prompt = "Vague, unclear, overexposure, low quality."


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--save_dir",
        default="qwen_t2i_inpainting",
        type=str,
        help="Directory to save.")
    parser.add_argument(
        "--output_file",
        default="output.json",
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="/mnt/bn/creative-algo/xsl/models/Qwen-Image",
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default="/mnt/bn/creative-algo/xsl/xsl-lora-0903/checkpoint-180000/pytorch_lora_weights.safetensors",
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

    os.makedirs(args.save_dir, exist_ok=True)

    args.device = torch.device(args.device)
    if args.dtype == 'fp16':
        args.dtype = torch.float16
    elif args.dtype == 'bf16':
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float32

    return args


def get_qwen_inpainting_images(image_url, position="center", ratio="9:16", margin=20):
    W, H = aspect_ratios[ratio]
    max_w, max_h = 0.7 * W, 0.5 * H

    product_img = load_or_download_image(image_url).convert('RGB')
    resize_ratio = min(max_w / product_img.width, max_h / product_img.height)
    out_w, out_h = int(product_img.width * resize_ratio), int(product_img.height * resize_ratio)

    # subject segmentation
    for i in range(3):
        try:
            mask_url = image_subject_seg(
                image_urls=[image_url], only_mask=1, refine_mask=2).success_image_urls[0]
            if isinstance(mask_url, str):
                break
        except Exception as e:
            if i == 2:
                raise Exception(e)

    mask = load_or_download_image(mask_url).convert('RGB')
    mask = mask.convert('L').resize((out_w, out_h), 1)
    mask = np.float32(mask)[..., None] / 255

    image = np.float32(product_img.resize((out_w, out_h), 1))

    canvas = np.zeros([H, W, 3], dtype=np.float32)
    canvas_mask = np.zeros([H, W], dtype=np.float32)
    half_w, half_h = out_w / 2, out_h / 2
    rand_x = random.randint(0, int(W * 0.05))
    rand_y = random.randint(0, int(H * 0.05))
    sign = random.choice([-1, 1])

    if position == "left":
        x1 = margin + rand_x * random.choice([0, 1])
        y1 = int(H / 2 - half_h) + rand_y * sign
    elif position == "right":
        x1 = W - out_w - 2 * margin + rand_x * random.choice([0, -1])
        y1 = int(H / 2 - half_h) + rand_y * sign
    elif position == "bottom-center":
        x1 = int(W / 2 - half_w) + rand_x * sign
        y1 = int(H - out_h - 2 * margin) + rand_y * random.choice([0, -1])
    elif position == "bottom-left":
        x1 = margin + rand_x * random.choice([0, 1])
        y1 = int(H - out_h - 2 * margin) + rand_y * random.choice([0, -1])
    elif position == "bottom-right":
        x1 = W - out_w - 2 * margin + rand_x * random.choice([0, -1])
        y1 = int(H - out_h - 2 * margin) + rand_y * random.choice([0, -1])
    else:
        x1 = int(W / 2 - half_w) + rand_x * sign
        y1 = int(H / 2 - half_h) + rand_y * sign

    canvas[y1: y1 + out_h, x1: x1 + out_w] = image * mask
    canvas_mask[y1: y1 + out_h, x1: x1 + out_w] = mask[..., 0] * 255

    canvas = Image.fromarray(np.uint8(np.clip(canvas, 0, 255)))
    canvas_mask = Image.fromarray(np.uint8(np.clip(canvas_mask, 0, 255)))
    return product_img, canvas, canvas_mask


def main(args):
    width, height = aspect_ratios["9:16"]
    device = args.device
    dtype = args.dtype

    pipe = QwenImageT2IEditPipeline.from_pretrained(args.pretrained_model_path, torch_dtype=dtype)
    pipe = pipe.to(device)

    pipe.load_lora_weights(args.lora_model_path)
    pipe.fuse_lora(lora_scale=1.0)
    pipe.unload_lora_weights()

    # load dataset
    data_list = load_file(args.input_file)
    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        data_list = data_list[args.rank::args.num_ranks]

    for line in tqdm(data_list):
        product_img, image, mask_image = get_qwen_inpainting_images(
            line['product_url'],
            position=line['gpt_prompt_result']['product_position'],
            ratio="9:16",
        )

        prompt = line['prompt']
        out_img = pipe(
            image_reference=product_img,
            prompt=prompt + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=40,
            true_cfg_scale=4.0,
            reference_image_size=(672, 672),
            image=image,
            mask_image=mask_image,
            invert_mask=True,
            mask_blur=2,
            blend_src_image=True,
        ).images[0]

        name = _gen_name("") + '_gen.jpg'
        out_img.save(join(args.save_dir, name))
        line["generate_image"] = name

    json_save(data_list, args.output_file)


if __name__ == "__main__":
    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"
    os.environ["TCE_PSM"] = "ad.creative.image_core_solution"

    args = parse_args()
    main(args)
    print('Done!')
