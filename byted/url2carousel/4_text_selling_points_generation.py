from typing import List
import argparse
from diffusers.data.utils import load_file, json_save
from tqdm import tqdm

from diffusers.data.byted.parallel import execute_concurrently
from diffusers.data.byted.clients.ad_creative_test_text_title import generate_carousel_selling_point

from overpass_ad_creative_image_core_solution.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.capability_thrift import Product, ImageInfo


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--output_file",
        default="output_selling_points.json",
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


def generate_carousel_text(target_image: str, product: Product, l3_vertical_tag: str = '', video_id: str = '', ad_title: str = '', creative_id: int = 0):
    url_product_name = product.product_name or ''
    url_brand_name = product.brand or ''
    url_description = product.description or ''
    url_selling_point = str(product.selling_points or [])
    target_language = product.language or 'en'

    sp_resp = generate_carousel_selling_point(
        target_image=target_image,
        creative_id=creative_id,
        url_product_name=url_product_name,
        url_brand_name=url_brand_name,
        url_description=url_description,
        url_selling_point=url_selling_point,
        target_language=target_language,
        video_id=video_id,
        ad_title=ad_title,
        vertical=l3_vertical_tag
    )

    return {'product_name': sp_resp.product_name,
            'primary_selling_points': sp_resp.primary_selling_points,
            'secondary_selling_points': sp_resp.secondary_selling_points,
            'cta': sp_resp.cta}


def generate_carousel_text_batch(image_infos: List[ImageInfo], product: Product, l3_vertical_tag: str = '', video_id: str = '', ad_title: str = '', creative_id: int = 0):
    text_list_batch = execute_concurrently(
        generate_carousel_text,
        [(image_info.URL, product, l3_vertical_tag, video_id, ad_title, creative_id) for image_info in image_infos],
        len(image_infos),
        timeout=300,
    )

    return text_list_batch


def main(args):
    data = load_file(args.input_file)
    if args.rank is not None and args.num_ranks is not None:
        assert args.rank < args.num_ranks
        data = data[args.rank::args.num_ranks]

    out = []
    for line in tqdm(data):
        try:
            product = Product(
                product_name=line["product_name"],
                brand=line["brand"],
                description=line["description"],
                selling_points=line["selling_points"],
                language="en"
            )
            image_infos = [ImageInfo(URL=a) for a in line["image_urls"]]
            ad_title = line["ad_title"]
            l3_vertical_tag = line["l3_vertical_tag"]

            resp = generate_carousel_text_batch(image_infos, product, l3_vertical_tag, ad_title=ad_title)
            line["text_carousels"] = resp
            out.append(line)

            if len(out) % 10 == 0:
                json_save(out, args.output_file)
        except Exception as e:
            print(line["ad_id"], line["ad_title"])
            print(e)

    # Final Save
    json_save(out, args.output_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
