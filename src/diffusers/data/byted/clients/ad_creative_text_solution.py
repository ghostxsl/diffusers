import logging
from typing import List, Optional

from overpass_ad_creative_text_solution.clients.rpc.ad_creative_text_solution import AdCreativeText_SolutionClient
from overpass_ad_creative_text_solution.euler_gen.idl.i18n_ad.creative.creative_factory.strategy_text.creative_text_solution_thrift import (
    GeneratePSASellingPointsReq,
    GeneratePSASellingPointsResp,
)

import diffusers.data.byted.errno as err
from euler.base_compat_middleware import gdpr_auth_middleware


AdCreativeTextSolution_Cli = AdCreativeText_SolutionClient(cluster="default", idc="my", transport="ttheader", timeout=300)
AdCreativeTextSolution_Cli.set_euler_client_middleware(gdpr_auth_middleware)


language_map = {
    "ID": "Indonesian",
    "MY": "Malay",
    "PH": "Filipino",
    "TH": "Thai",
    "VN": "Vietnamese",
}


def generate_psa_sellings_points(
    product_id: int,
    target_language: str = "English",
    strategy_version: str = "2001",
    source: str = "",
    product_description: str = "",
    product_name: str = "",
    brand_name: str = "",
    first_category_name: Optional[str] = None,
    second_category_name: Optional[str] = None,
    third_category_name: Optional[str] = None,
    video_caption: Optional[str] = None,
) -> GeneratePSASellingPointsResp:
    req = GeneratePSASellingPointsReq(
        product_id=product_id,
        target_language=target_language,
        strategy_version=strategy_version,
        source=source,
        product_description=product_description,
        product_name=product_name,
        brand_name=brand_name,
        first_category_name=first_category_name,
        second_category_name=second_category_name,
        third_category_name=third_category_name,
        video_caption=video_caption,
    )
    logging.info(f"[GeneratePSASellingPoints] req: {req}")

    code, msg, resp = AdCreativeTextSolution_Cli.GeneratePSASellingPoints(req_object=req)
    if not resp or code != 0:
        logging.error(f"[GeneratePSASellingPoints] failed, code: {code}, msg: {msg}")
        raise err.WithCodeError(err.ErrorCodeSellingPointGenerationError, f"[GeneratePSASellingPoints] failed, code: {code}, msg: {msg}")

    logging.info("[GeneratePSASellingPoints] success: {}".format(resp))
    return resp
