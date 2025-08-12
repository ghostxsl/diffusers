import os
os.environ["TCE_PSM"] = "ad.creative.image_core_solution"
from tqdm import tqdm
import argparse

from overpass_ad_creative_image_core_solution.euler_gen.idl.i18n_ad.creative.creative_factory.creative_image_core_solution_thrift import (
    URLImageFilterParam,
    URLImageTag
)
from diffusers.data.byted.clients.ad_creative_url2product import url2products
from diffusers.data.utils import load_csv_or_xlsx_to_dict, json_save


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--output_file",
        default="output_product.json",
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
    # url2product
    image_filter_param = URLImageFilterParam(
        tag_filter=[URLImageTag.target_product, URLImageTag.product_user],
        only_relevant=False,
    )
    source = 'TTAM/ImageGenerationAutomationDelivery/CarouselWeb'
    print(image_filter_param, source)

    data = load_csv_or_xlsx_to_dict(args.input_file)
    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        data = data[args.rank::args.num_ranks]

    out = []
    for line in tqdm(data):
        try:
            resp = url2products(
                URL=line['External Website URL'],
                source=source,
                adv_id=int(line['Advertiser ID']),
                need_selling_point=True,
                image_filter_param=image_filter_param)
            image_urls = [a.URL for a in resp.products[0].image_infos]
            if len(image_urls) == 0:
                continue

            out.append({
                "ad_id": line['Ad ID'],
                "advertiser_id": line['Advertiser ID'],
                "l2_vertical_tag": line['L2 Vertical Tag'],
                "l3_vertical_tag": line['L3 Vertical Tag'],
                "l4_vertical_tag": line['L4 Vertical Tag'],
                "external_website_url": line['External Website URL'],
                "ad_country": line["Ad Country"],
                "ad_title": line['ad_title'],
                "product_name": resp.products[0].product_name,
                "brand": resp.products[0].brand,
                "description": resp.products[0].description,
                "selling_points": resp.products[0].selling_points,
                "image_urls": image_urls,
            })

            if len(out) % 10 == 0:
                json_save(out, args.output_file)
        except Exception as e:
            print(line['External Website URL'], line['Advertiser ID'])
            print(e)

    # Final Save
    json_save(out, args.output_file)


if __name__ == "__main__":
    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token  # os.environ['SEC_TOKEN_STRING'] = "toutiao.growth.xenon"
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"

    args = parse_args()
    main(args)
    print('Done!')
