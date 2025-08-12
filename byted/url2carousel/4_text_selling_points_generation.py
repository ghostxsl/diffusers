from typing import List

from diffusers.data.byted.parallel import execute_concurrently
from diffusers.data.byted.clients.ad_creative_test_text_title import generate_carousel_selling_point

from overpass_ad_creative_image_core_solution.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.capability_thrift import Product, ImageInfo


def generate_carousel_text(target_image: str, product: Product, l3_vertical_tag: str = '', video_id: str = '', ad_title: str = '', creative_id: int = 0) -> List[str]:

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
        industry=l3_vertical_tag
    )

    product_name = sp_resp.product_name
    primary_selling_points = sp_resp.primary_selling_points
    secondary_selling_points = sp_resp.primary_selling_points
    cta = sp_resp.cta

    text_list = []
    text_list.append(f"Title: {primary_selling_points[0]}")
    text_list.append(f"SubTitle: {secondary_selling_points[0]}")

    return text_list




def generate_carousel_text_batch(image_infos: List[ImageInfo], product: Product, l3_vertical_tag: str = '', video_id: str = '', ad_title: str = '', creative_id: int = 0):
    text_list_batch = execute_concurrently(
        generate_carousel_text,
        [(image_info.URL, product, l3_vertical_tag, video_id, ad_title, creative_id) for image_info in image_infos],
        len(image_infos),
        timeout=120
    )

    return text_list_batch


if __name__ == '__main__':

    product = Product(
        product_name="Speaker",
        brand="turtleboxaudio",
        description="Rugged, portable speaker with premium sound Engineered for the outdoors: IP-67 water, drop, and crush-proof 3-day battery life Pairs with Original Gen 3’s, Rangers, & Grandes in Party Mode or TWS (true wireless stereo), but does NOT pair with the Original Gen 2",
        selling_points=[
            "Rugged, portable speaker with premium sound",
            "IP-67 water, drop, and crush-proof 3-day battery life",
            "Pairs with Original Gen 3’s, Rangers, & Grandes in Party Mode or TWS (true wireless stereo)",
            "but does NOT pair with the Original Gen 2"
        ],
        language="en"
    )
    image_infos = [
        ImageInfo(URL="https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20250711336e56df539e08644be0aa6d")
    ] * 4
    ad_title = "Rugged, Waterproof, Premium Sound. That's what you get with a Turtlebox speaker."
    l3_vertical_tag = "Consumer, Computer, Communication Electronics"

    resp = generate_carousel_text_batch(image_infos, product, l3_vertical_tag, ad_title=ad_title)
    print(resp)