import logging
import diffusers.data.byted.errno as err

from euler.base_compat_middleware import gdpr_auth_middleware
from overpass_ad_creative_test_text_title.clients.rpc.ad_creative_test_text_title import AdCreativeTest_Text_TitleClient
from overpass_ad_creative_test_text_title.euler_gen.ad.test_text_title.idl.test_text_title_thrift import (
    GenerateCarouselSellingPointReq,
    GenerateCarouselSellingPointResp,
)


AdCreativeTestTextTitle_Cli = AdCreativeTest_Text_TitleClient(cluster="default", idc="my", transport="ttheader", timeout=300)
AdCreativeTestTextTitle_Cli.set_euler_client_middleware(gdpr_auth_middleware)


def generate_carousel_selling_point(
    target_image: str,
    creative_id: int = 0,
    video_id: str = "",
    url_product_name: str = "",
    url_brand_name: str = "",
    url_title: str = "",
    url_description: str = "",
    url_selling_point: str = "",
    reference_image_list: str = "",
    target_language: str = "en",
    ad_title: str = "",
    vertical: str = "",
    video_caption: str = "",
) -> GenerateCarouselSellingPointResp:
    """
    卖点生成能力
    """

    req = GenerateCarouselSellingPointReq(
        target_image=target_image,
        creative_id=creative_id,
        video_id=video_id,
        url_product_name=url_product_name,
        url_brand_name=url_brand_name,
        url_title=url_title,
        url_description=url_description,
        url_selling_point=url_selling_point,
        reference_image_list=reference_image_list,
        target_language=target_language,
        ad_title=ad_title,
        industry=vertical,
        video_caption=video_caption,
    )

    logging.info(f"[generate_carousel_selling_point] req: {req}")

    code, msg, resp = AdCreativeTestTextTitle_Cli.GenerateCarouselSellingPoint(req_object=req)
    if not resp or code != 0:
        raise err.WithCodeError(err.ErrorCodeSellingPointGenerationError, f"[generate_carousel_selling_point] failed, code: {code}, msg: {msg}")

    logging.info("[generate_carousel_selling_point] success: {}".format(resp))

    return resp


if __name__ == "__main__":
    params = {
        "creative_id": 1111,
        "video_id": "",
        "url_product_name": "Speaker",
        "url_brand_name": "turtleboxaudio",
        "url_title": "Original: Gen 3",
        "url_description": "Rugged, portable speaker with premium sound Engineered for the outdoors: IP-67 water, drop, and crush-proof 3-day battery life Pairs with Original Gen 3’s, Rangers, & Grandes in Party Mode or TWS (true wireless stereo), but does NOT pair with the Original Gen 2",
        "url_selling_point": "[]",
        "reference_image_list": "",
        "target_image": "https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20250711336e56df539e08644be0aa6d",
        "target_language": "en",
        "ad_title": "Rugged, Waterproof, Premium Sound. That's what you get with a Turtlebox speaker.",
        "industry": "Consumer, Computer, Communication Electronics",
    }

    resp = generate_carousel_selling_point(**params)
    print(resp)
