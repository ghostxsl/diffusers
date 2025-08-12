import os
from typing import List

from euler.base_compat_middleware import gdpr_auth_middleware
from overpass_ad_creative_image_core_solution.euler_gen.idl.i18n_ad.creative.creative_factory.creative_image_core_solution_thrift import (
    URL2ProductsReq,
    URL2ProductsResp,
    URLImageFilterParam,
    URLImageTag
)
from overpass_ad_creative_image_core_solution.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.capability_thrift import Product
from overpass_ad_creative_image_core_solution.clients.rpc.ad_creative_image_core_solution import AdCreativeImage_Core_SolutionClient


client = AdCreativeImage_Core_SolutionClient(idc="maliva", cluster="products2carousel", transport="ttheader", timeout=300)
client.set_euler_client_middleware(gdpr_auth_middleware)


def url2products(
    URL: str,
    source: str,
    adv_id: int,
    need_selling_point: bool = False,
    image_filter_param: URLImageFilterParam = URLImageFilterParam(),
    products: List[Product] = [],
    need_crawl: bool = True,
    use_crawl_cache: bool = True,
    crawl_cache_ttl: float = 7,
    use_understanding_cache: bool = True,
    understanding_cache_ttl: float = 7,
) -> URL2ProductsResp:
    req = URL2ProductsReq(
        URL=URL,
        source=source,
        adv_id=adv_id,
        need_selling_point=need_selling_point,
        image_filter_param=image_filter_param,
        use_crawl_cache=use_crawl_cache,
        use_understanding_cache=use_understanding_cache,
        products=products,
        need_crawl=need_crawl,
        crawl_cache_ttl=crawl_cache_ttl,
        understanding_cache_ttl=understanding_cache_ttl,
    )

    code, msg, resp = client.URL2Products(req_object=req)
    if code != 0:
        raise Exception(msg)

    return resp


if __name__ == "__main__":
    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token  # os.environ['SEC_TOKEN_STRING'] = "toutiao.growth.xenon"
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"
    os.environ["TCE_PSM"] = "ad.creative.image_core_solution"  # toutiao.growth.xenon

    image_filter_param = URLImageFilterParam(
        tag_filter=[URLImageTag.target_product, URLImageTag.product_user],
        only_relevant = False
    )
    print(image_filter_param)

    URL = 'https://tushbaby.com/products/tushbaby?variant=15109665390658'
    source = 'TTAM/ImageGenerationAutomationDelivery/CarouselWeb'
    adv_id = 7033034570199023617
    need_selling_point = True
    resp = url2products(URL=URL, source=source, adv_id=adv_id, need_selling_point=need_selling_point, image_filter_param=image_filter_param)
    print(resp)
    