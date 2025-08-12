import logging
import os
from typing import List, Union, Optional, Tuple, Dict

from overpass_ad_creative_data_forge.clients.rpc.ad_creative_data_forge import AdCreativeData_ForgeClient
from overpass_ad_creative_data_forge.euler_gen.idl.i18n_ad.creative.creative_factory.creative_data_forge_thrift import (
    GetProductInfoRequest,
    ProductInfo,
    ProductKeyType,
)

import diffusers.data.byted.errno as err
from euler.base_compat_middleware import gdpr_auth_middleware


DataForgeCli = AdCreativeData_ForgeClient(idc="my", cluster="default", transport="ttheader", timeout=300)
DataForgeCli.set_euler_client_middleware(gdpr_auth_middleware)


# get_product_info
# # product_key_type
# # open_loop=1
# # url=2
# # open_loop_sku=3
def get_product_info(
    product_key: List[str],
    key_type: Union[ProductKeyType, int] = ProductKeyType.closed_loop,
    source: Optional[str] = None,
    bizSource: Optional[str] = None,
    adv_id: Optional[str] = None,
    catalog_id: Optional[str] = None,
) -> List[ProductInfo]:
    req = GetProductInfoRequest(product_key=product_key, product_key_type=key_type, source=source, bizSource=bizSource, adv_id=adv_id)
    logging.info(f"[get_product_info] req: {req}")
    code, msg, resp = DataForgeCli.GetProductInfo(req_object=req)
    logging.info(f"[get_product_info] code: {code}, msg: {msg}, resp: {resp}")
    if not resp or code != 0:
        raise err.WithCodeError(err.ErrCodeCreativeToolboxError, f"[get_product_info] failed, code: {code}, msg: {msg}")

    return resp.products
