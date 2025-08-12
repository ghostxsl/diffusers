# Copyright (c) wilson.xu. All rights reserved.
import argparse
from tqdm import tqdm

from diffusers.data.byted.clients.azure_mllm import gemini_25_flash_image_gen
from diffusers.data.utils import load_file, json_save
from diffusers.data.outer_vos_tools import encode_pil_bytes
from diffusers.data.byted.tos import save_tos


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--output_file",
        default="output.json",
        type=str,
        help="Path to image list on vos.")

    parser.add_argument(
        "--rank",
        default=None,
        type=int)
    parser.add_argument(
        "--num_ranks",
        default=None,
        type=int)
    args = parser.parse_args()

    return args


def main(args):
    # Load dataset
    data_list = load_file(args.input_file)
    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        data_list = data_list[args.rank::args.num_ranks]

    out = []
    for line in tqdm(data_list):
        try:
            prompt = line["prompt"]
            image_url = line["image_url"]
            res = gemini_25_flash_image_gen(
                prompt, image_url, specify_gen_ratio=True, ratio="9:16")
            if res == {}:
                continue

            gen_url = save_tos(encode_pil_bytes(res["image"], False))
            line["gemini_url"] = gen_url
            out.append(line)

            if len(out) % 10 == 0:
                json_save(out, args.output_file)
        except Exception as e:
            print(line)
            print(e)

    # Final save
    json_save(out, args.output_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
