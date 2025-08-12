import os
import argparse
from os.path import join, splitext, basename
from tqdm import tqdm
import torch

from diffusers.pipelines.byted.pipeline_qwenimage_t2i_edit import QwenImageT2IEditPipeline
from diffusers.data.utils import load_file, json_save
from diffusers.data.outer_vos_tools import load_or_download_image, decode_pil_bytes
from diffusers.data.byted.tos import _gen_name, get_file_from_tos


aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472)
}
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
        default="qwen_t2i_test",
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

    os.makedirs(args.save_dir, exist_ok=True)

    args.device = torch.device(args.device)
    if args.dtype == 'fp16':
        args.dtype = torch.float16
    elif args.dtype == 'bf16':
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float32

    return args

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
        product_img = decode_pil_bytes(get_file_from_tos(line['product_url']).data, False).convert('RGB')
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
        ).images[0]

        name = _gen_name("") + '_gen.jpg'
        out_img.save(join(args.save_dir, name))
        line["generate_image"] = name

    json_save(data_list, args.output_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
