# Copyright (c) wilson.xu. All rights reserved.
import os
os.environ["TCE_PSM"] = "ad.creative.image_core_solution"
from os.path import join
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from diffusers.data.utils import load_file, load_csv_or_xlsx_to_dict, json_save, get_product_and_mask_image
from diffusers.data.outer_vos_tools import load_or_download_image, encode_pil_bytes
from diffusers.data.byted.clients.creative_ai_capability import image_subject_seg
from overpass_ad_creative_ai_capabilities.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.capability_thrift import ImageInfo


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
        "--pretrained_model_path",
        default="/mnt/bn/creative-algo/xsl/models/dinov2-large",
        type=str)

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
        help="Device to use (e.g. cpu, gpu, gpu:0, etc.)")

    args = parser.parse_args()
    args.device = torch.device(args.device)

    return args


def get_subject_image(image_path, mask_url=None):
    image = load_or_download_image(image_path)
    if mask_url is None:
        image_info = ImageInfo(Binary=encode_pil_bytes(image, False))
        mask_url = image_subject_seg(
            image_urls=[], image_infos=[image_info], only_mask=1, refine_mask=2).success_image_infos[0].URL

    mask = load_or_download_image(mask_url)
    product_image, _ = get_product_and_mask_image(image, mask, 10)
    return product_image


@torch.inference_mode()
def get_last_hidden_state(image, processor, model, **kwargs):
    inputs = processor(images=image, return_tensors="pt", **kwargs).to(model.device)
    outputs = model(**inputs)

    return outputs.last_hidden_state


def main(args):
    device = args.device
    processor = AutoImageProcessor.from_pretrained(args.pretrained_model_path)
    processor.do_center_crop = False
    processor.size = {"height": 336, "width": 336}
    model = AutoModel.from_pretrained(args.pretrained_model_path).to(device)

    # load dataset
    try:
        data_list = load_file(args.input_file)
    except Exception:
        data_list = load_csv_or_xlsx_to_dict(args.input_file)
    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        data_list = data_list[args.rank::args.num_ranks]

    out = []
    for line in tqdm(data_list):
        try:
            if not line['ocr_char']['char_tag']:
                continue

            gt_image = load_or_download_image(line['product_url'])
            gen_image = get_subject_image(line['generate_image'], line['generate_mask_url'])

            gt_hid = get_last_hidden_state(gt_image, processor, model)
            gen_hid = get_last_hidden_state(gen_image, processor, model)

            line["subject_similarity"] = F.cosine_similarity(
                gt_hid.mean(dim=1), gen_hid.mean(dim=1), dim=1
            ).item()
            line["subject_detail_similarity"] = F.cosine_similarity(
                gt_hid[:, 1:].flatten(), gen_hid[:, 1:].flatten(), dim=0
            ).item()

            out.append(line)
            if len(out) % 50 == 0:
                json_save(out, args.output_file)
        except Exception as e:
            print(line)
            print(e)

    # Final Save
    json_save(out, args.output_file)


if __name__ == "__main__":
    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"

    args = parse_args()
    main(args)
    print('Done!')
