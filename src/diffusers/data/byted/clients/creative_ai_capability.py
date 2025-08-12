import json
import logging
import os
# os.environ["CONSUL_HTTP_HOST"] = "10.105.215.100"
# os.environ["TCE_PSM"] = "ad.creative.image_core_solution"
from retrying import retry
from typing import List, Optional, Dict, Tuple, Union

import diffusers.data.byted.errno as err
from diffusers.data.byted.middleware import client_logid, calc_runtime_middleware

import byteddps
from euler.base_compat_middleware import gdpr_auth_middleware
from overpass_ad_creative_image_core_solution.euler_gen.idl.base.base_thrift import Base

# Overpass写法
from overpass_ad_creative_ai_capabilities.clients.rpc.ad_creative_ai_capabilities import AdCreativeAi_CapabilitiesClient
from overpass_ad_creative_ai_capabilities.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.strategy_thrift import AnyType
from overpass_ad_creative_ai_capabilities.euler_gen.idl.i18n_ad.creative.creative_factory.capabilities.image_capability_thrift import (
    ColorExtractRequest,
    ImageOcrRequest,
    ImageOcrResponse,
    PainterLayerMergeRequest,
    ImageAutoResizeRequest,
    UrlToImageInfoRequest,
    ImageBodyFaceDetectRequest,
    ImageBodyFaceDetectResponse,
    ImageAestheticScoringRequest,
    ImageAestheticScoringResponse,
    ImageParsingSegRequest,
    ImageAutoCropRequest,
)
from overpass_ad_creative_ai_capabilities.euler_gen.idl.i18n_ad.creative.creative_factory.capabilities.text_capability_thrift import (
    GenerateProductsSellingPointsResp,
    TextAttributeType,
    GetTextAttributeReq,
    GetTextAttributeResp,
    GenerateCTARequest,
    GenerateCTAResponse,
)
from overpass_ad_creative_ai_capabilities.euler_gen.idl.i18n_ad.creative.creative_factory.creative_ai_capability_thrift import (
    GenerateTextCapabilityReq,
    ImageXEnhanceRequest,
    ImageXEnhanceResponse,
    ImageSubjectSegResponse,
    ImageSubjectSegRequest,
    CallStrategyCapabilityResp,
    CallStrategyCapabilityReq,
    GenImageByPromptResponse,
    GenImageByPromptRequest,
    ImageSuperResolutionRequest,
)
from overpass_ad_creative_ai_capabilities.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.capability_thrift import (
    Product,
    ImageInfo,
    AdvanceSetting,
)


AiCapabilityCli = AdCreativeAi_CapabilitiesClient(idc="my", cluster="default", transport="ttheader", timeout=300)
AiCapabilityCli.set_euler_client_middleware(gdpr_auth_middleware)
AiCapabilityCli.set_euler_client_middleware(client_logid)
AiCapabilityCli.set_euler_client_middleware(calc_runtime_middleware)


@retry(stop_max_attempt_number=2, wait_fixed=300)
def image_subject_seg(
    image_urls: Optional[List[str]],
    image_infos: List[ImageInfo] = None,
    only_mask: int = 3,
    rgb: List[int] = [-1, -1, -1],
    refine_mask: int = 2,
    with_contour: int = 0,
    ret_encode_method: int = 0,
    operation_codes: List[str] = None,
    seg_model: str = "zhichuang",
) -> ImageSubjectSegResponse:
    """
    ai_capabilities主体分割能力
    智创模型参考https://bytedance.larkoffice.com/docx/doxcnPTkZ2j3ZQRPb5ahgFzqtxh
    Parameters:
    - image_urls (List[str]): 输入图片url列表，目前只支持单图；
    - image_infos (List[ImageInfo]): 和image_url二选一；
    - only_mask (int): mask option
        0->返回裁剪出主体区域的BGRA透明图片
        1->返回原图大小Mask分割图
        2->返回裁剪出主体区域的BGR前景图 叠加 方形纯色背景
        3->返回原图大小的BGRA透明前景图
        4->返回原图大小的BGRA透明前景图+原图大小mask两个产物
    - rgb (List[int]): background color
    - refine_mask (int): 0：不对边缘增强 1：对边缘增强（matting lite 更快) 2：对边缘增强（matting large) （需精细化分割推荐开启该参数）
    - with_contour (int): 0=不返回
    - ret_encode_method (int):
        0: png编码（默认）
        1: webp编码（对于rgba4通道返回结果图片时相对png编码耗时更久，但能够有效减少产物体积）
        2: jpg编码（quality 100）（对于rgba返回结果，将rgb采用jpg编码，alpha采用png编码拆分“rgb-alpha”成两个图片返回。
    - operation_codes (List[str]): 操作指令，用于指导模块如何操作
    - seg_model (str): default=zhichuang, sam=sam2
    Returns:
    - ImageSubjectSegResponse: 主体分割结果，请求原始结构体
    """
    req = ImageSubjectSegRequest(
        image_urls=image_urls,
        image_infos=image_infos,
        only_mask=only_mask,
        rgb=rgb,
        refine_mask=refine_mask,
        with_contour=with_contour,
        ret_encode_method=ret_encode_method,
        operation_codes=operation_codes,
        seg_model=seg_model,
    )
    code, msg, resp = AiCapabilityCli.ImageSubjectSeg(req_object=req)
    logging.info(f"[image_subject_seg] code: {code}, msg: {msg}, resp: {resp}")

    if "21003" in msg:
        raise err.WithCodeError(err.ErrCodeNoBackgroundImage, f"[image_subject_seg] failed, code: {code}, msg: Input Image Has No Background.")
    if not resp or code != 0:
        raise err.WithCodeError(err.ErrorCodeSegmentError, f"[image_subject_seg] failed, code: {code}, msg: {msg}")

    return resp


def image_parsing_seg(
    image_urls: List[str] = None, image_infos: List[ImageInfo] = None, parsing_type: int = 2
) -> Tuple[List[ImageInfo], List[ImageInfo], List[Dict]]:
    req = ImageParsingSegRequest(
        image_urls=image_urls,
        image_infos=image_infos,
        parsing_type=parsing_type,
    )
    logging.info(f"[image_parsing_seg] req: {req}")
    code, msg, resp = AiCapabilityCli.ImageParsingSeg(req_object=req)
    logging.info(f"[image_parsing_seg] code: {code}, msg: {msg}, resp: {resp}")
    if not resp or code != 0:
        raise Exception(f"[image_parsing_seg] failed, code: {code}, msg: {msg}")
    assert (
        len(resp.prod_infos) == len(resp.bg_infos) == len(resp.extra_infos)
    ), f"[image_parsing_seg] length mismatch: prod_infos={len(resp.prod_infos)}, bg_infos={len(resp.bg_infos)}, extra_infos={len(resp.extra_infos)}"

    cc_templates = []
    for extra_info_str in resp.extra_infos:
        templ = json.loads(json.loads(json.loads(extra_info_str)["cc_template"]))
        cc_templates.append(templ)

    return resp.prod_infos, resp.bg_infos, cc_templates


def image_body_face_detect(urls: List[str], only_judge_people: bool = True, max_obj_num: int = 2) -> List[Dict]:
    face_body_detect_res = []
    req = ImageBodyFaceDetectRequest(
        image_urls=urls,
        only_judge_people=only_judge_people,
        max_obj_num=max_obj_num,
    )
    logging.info(f"[image_body_face_detect] req: {req}")
    code, msg, resp = AiCapabilityCli.ImageBodyFaceDetect(req_object=req)
    logging.info(f"[image_body_face_detect] code: {code}, msg: {msg}, resp: {resp}")
    if not resp or code != 0:
        raise Exception(f"[image_body_face_detect] failed, code: {code}, msg: {msg}")
    if resp.with_people:
        face_body_detect_res = resp.detected_items[0]  # a list of ObjectDetectItem

    return face_body_detect_res


def image_body_face_detect_raw(urls: List[str], only_judge_people: bool = True, max_obj_num: int = 2, judge_people_thresold: float = 0.8) -> ImageBodyFaceDetectResponse:
    req = ImageBodyFaceDetectRequest(
        image_urls=urls,
        only_judge_people=only_judge_people,
        max_obj_num=max_obj_num,
        judge_people_thresold=judge_people_thresold
    )
    logging.info(f"[image_body_face_detect_raw] req: {req}")
    code, msg, resp = AiCapabilityCli.ImageBodyFaceDetect(req_object=req)
    logging.info(f"[image_body_face_detect_raw] code: {code}, msg: {msg}, resp: {resp}")
    if not resp or code != 0:
        raise err.WithCodeError(err.ErrCodeAiCapabilityError, f"[image_body_face_detect_raw] failed, code: {code}, msg: {msg}")

    return resp


def color_extract(
    image_urls: List[str],
    top_color_num: Optional[int] = 3,
    use_weight: Optional[bool] = True,
    only_for_subject: Optional[bool] = False,
    method: Optional[int] = 0,
) -> Tuple[Optional[List[Dict[float, str]]], Optional[List[List[str]]]]:
    """
    :param method: 目前支持 不填/0:智能取色  2：文本配色 4：调色盘推荐 5：渐变色推荐
    :param image_urls: // only accept one now. Need to process urls. tos/http url
    :param top_color_num: // color u want to extract, default = 4
    :param use_weight: // 取色是否优先考虑人眼易感知的颜色（默认为true）
    :param only_for_subject: // 是否仅对主体进行颜色提取
    :return:
    1: optional list<map<double, string>>  ExtractedColor  // 智能取色得到的rgb色盘， list是输入图片的idx顺，map是比例+rgb色盘。 rgb形如"[r,g,b]"的字符串
    //    2: optional list<list<string>>   ToCreateColor   // 用于后续配色的RGB色盘, 目前没用
    """
    # code, msg, resp
    req: ColorExtractRequest = ColorExtractRequest(
        image_urls=image_urls,
        top_color_num=top_color_num,
        use_weight=use_weight,
        only_for_subject=only_for_subject,
        method=method,
    )
    logging.info("[ColorExtract] req: {}".format(req))
    code, msg, resp = AiCapabilityCli.ColorExtract(req_object=req)
    # resp = AiCapabilityCli.ColorExtract(req)
    if code != 0:
        logging.error(f"[ColorExtract] failed, code: {code}, msg: {msg}, image_urls: {image_urls}")
        raise Exception(f"[ColorExtract] failed, code: {code}, msg: {msg}")
    if resp and resp.ExtractedColor is None or (len(resp.ExtractedColor) == 0 and len(image_urls) != 0):
        logging.error(f"[ColorExtract] failed, code: {code}, msg: {msg}, image_urls: {image_urls}")
        raise Exception("[ColorExtract] failed, resp.ExtractedColor is None or resp.ExtractedColor == []")
    logging.info("[ColorExtract] resp: {}".format(resp))
    return resp.ExtractedColor or None, resp.color_palette or None


def image_ocr(
    image_url: Union[str, List[str]],
    image_info: Union[ImageInfo, List[ImageInfo]] = None,
) -> ImageOcrResponse:
    """
    图片OCR能力
    Parameters:
    - image_url (Union[str, List[str]]): 输入图片url或者url_list
    - image_info (Union[ImageInfo, List[ImageInfo]]): 和image_url二选一；
    Returns:
    - ImageOcrResponse: OCR识别结果，请求原始结构体.
    """
    if isinstance(image_url, str):
        image_urls = [image_url]
    else:
        image_urls = image_url

    if isinstance(image_info, ImageInfo):
        image_infos = [image_info]
    else:
        image_infos = image_info

    req = ImageOcrRequest(image_urls=image_urls, image_infos=image_infos)
    code, msg, resp = AiCapabilityCli.ImageOcr(req_object=req)
    if code != 0:
        raise Exception(f"[image_ocr] failed, code: {code}, msg: {msg}")

    logging.info("[image_ocr] success.")

    return resp


def image_auto_resize(
    image_urls: List[str] = None,
    image_infos: List[ImageInfo] = None,
    ratio: float = 0.5625,
    target_width: int = 720,
    target_height: int = 1280,
    mode: int = 3,
    default_pad_color: str = None,
) -> Tuple[List[str], List[ImageInfo]]:
    req = ImageAutoResizeRequest(
        image_urls=image_urls,
        image_infos=image_infos,
        ratio=ratio,
        target_width=target_width,
        target_height=target_height,
        mode=mode,
        default_pad_color=default_pad_color,
    )
    logging.info(f"[image_auto_resize] req: {req}")
    code, msg, resp = AiCapabilityCli.ImageAutoResize(req_object=req)
    logging.info(f"[image_auto_resize] code: {code}, msg: {msg}, resp: {resp}")
    if not resp or code != 0:
        raise err.WithCodeError(err.ErrCodeAiCapabilityError, f"[image_auto_resize] failed, code: {code}, msg: {msg}")

    return resp.success_image_urls, resp.success_image_infos


def gen_image_by_prompt(
    prompt: str,
    width: int,
    height: int,
    negative_prompt: str = "",
    model: str = "general_v2.0",
    num: int = 1,
    sample_steps: int = 30,
    setting: AdvanceSetting = AdvanceSetting(),
) -> GenImageByPromptResponse:
    """
    文生图能力，支持高美感v1.4 & seed2.0。
    Parameters:
    - prompt (str): 输入prompt;
    - width (int): 输出图片宽度;
    - height (int): 输出图片高度;
    - negative_prompt (str): 负向prompt;
    - model (str): 模型类型，general_v2.0=seed2.0，置空调用高美感v1.4;
    - num (int): 输出图片数量;
    - sample_steps (int): 采样步数;
    - setting (AdvanceSetting): 高级设置;
    Returns:
    - GenImageByPromptResponse: 文生图结果，请求原始结构体。
    """
    req = GenImageByPromptRequest(
        prompt=prompt,
        width=width,
        height=height,
        negative_prompt=negative_prompt,
        model=model,
        num=num,
        sample_steps=sample_steps,
        setting=setting,
    )
    code, msg, resp = AiCapabilityCli.GenImageByPrompt(req_object=req)
    if code != 0:
        raise err.WithCodeError(err.ErrCodeAiCapabilityError, f"[gen_image_by_prompt] failed, code: {code}, msg: {msg}")
    logging.info("[gen_image_by_prompt] resp: {}".format(resp))
    return resp


def call_strategy_capability(strategy_id: str, strategy_req: Dict[str, AnyType], source: int) -> CallStrategyCapabilityResp:
    req = CallStrategyCapabilityReq(strategy_id=strategy_id, strategy_req=strategy_req, source=source)
    logging.info(f"[call_strategy_capability] req: {req}")
    code, msg, resp = AiCapabilityCli.CallStrategyCapability(req_object=req)
    logging.info(f"[call_strategy_capability] code: {code}, msg: {msg}, resp: {resp}")
    if not resp or code != 0:
        raise err.WithCodeError(err.ErrCodeAiCapabilityError, f"[call_strategy_capability] failed, code: {code}, msg: {msg}")

    return resp


def url_to_imageinfo(image_urls: List[str], skip_attr_calc: bool = False) -> List[ImageInfo]:
    req = UrlToImageInfoRequest(urls=image_urls, skip_attr_calc=skip_attr_calc)
    logging.info(f"[url_to_imageinfo] req: {req}")
    code, msg, resp = AiCapabilityCli.UrlToImageInfo(req_object=req)
    logging.info(f"[url_to_imageinfo] code: {code}, msg: {msg}, resp: {resp}")
    if not resp or code != 0:
        raise Exception(f"[url_to_imageinfo] failed, code: {code}, msg: {msg}")

    return resp.image_infos


# generate_products_selling_points 多卖点生成，hybrid同款。 输入
def generate_products_selling_points(products: List[Product], skip_multi_language: bool) -> List[Product]:
    logging.info("[generate_products_selling_points] req products: {}, skip_multi_language: {}".format(products, skip_multi_language))
    is_multi_lang: List[bool] = []
    if skip_multi_language:
        is_multi_lang = judge_text_is_multi_language(products)
        to_gen_products: List[Product] = [products[idx] for idx in range(len(products)) if is_multi_lang[idx] is False]
        logging.info("[generate_products_selling_points] skip_multi_language...req products: {}".format(to_gen_products))
    else:
        to_gen_products: List[Product] = products
        is_multi_lang = [False for _ in range(len(products))]
    if len(to_gen_products) < 1:  # 不需要处理，直接返回
        return products
    resp: GenerateProductsSellingPointsResp
    code, msg, resp = AiCapabilityCli.GenerateProductsSellingPoints(products=to_gen_products)
    logging.info("[GenerateProductsSellingPoints] code: {}, msp: {}, resp: {}".format(code, msg, resp))
    processed_products = resp.products or [] if resp else []
    if code != 0 or len(processed_products) != len(to_gen_products):
        raise Exception(f"[GenerateProductsSellingPoints] failed, code: {code}, msg: {msg}, len(resp.products): {processed_products}")
    res_idx = 0  # 用来resp索引位置，进行顺序补位
    res_product: List[Product] = []
    for idx in range(len(products)):
        if is_multi_lang[idx]:
            res_product.append(products[idx])
        else:
            res_product.append(processed_products[res_idx])
            res_idx += 1
    logging.info("[generate_products_selling_points] resp products: {}".format(res_product))
    return res_product


# get_text_attribute 获得输入文本对应的
def judge_text_is_multi_language(products: List[Product]) -> List[bool]:
    req = GetTextAttributeReq(products=products, attribute_type=int(TextAttributeType.IsMultiLanguage.value))
    logging.info("[getTextAttribute] req: {}".format(req))
    resp: GetTextAttributeResp
    code, msg, resp = AiCapabilityCli.GetTextAttribute(req_object=req)
    logging.info("[getTextAttribute] code: {}, msp: {}, resp: {}".format(code, msg, resp))
    if code != 0:
        raise Exception(f"[getTextAttribute] failed, code: {code}, msg: {msg}")
    return resp.judge_attributes or []


# generate_text_widgets_from_product_and_template 使用template id、product，根据类似hybrid的规则生成文字图层list[json],可根据id回填。其中product的selling point字段会被拿去填充.json=list[widget]
def generate_text_widgets_from_product_and_template(template_id: str, product: Product, gen_num=1, pre_gen_style=True) -> List[str]:
    if pre_gen_style:
        req = GenerateCTARequest(
            product=product, template_id=template_id, image_urls=["random" for _ in range(gen_num)], task_type="selling_point_pre_gen"
        )
    else:
        req = GenerateCTARequest(product=product, template_id=template_id, image_urls=["random" for _ in range(gen_num)])
    logging.info("[generate_text_widgets_from_product_and_template] req: {}".format(req))
    resp: GenerateCTAResponse
    code, msg, resp = AiCapabilityCli.GenerateSellingPoint(req_object=req)
    logging.info("[generate_text_widgets_from_product_and_template] code: {}, msp: {}, resp: {}".format(code, msg, resp))
    if code != 0 or len(resp.cta_painter_layers) != gen_num:
        raise Exception(f"[generate_text_widgets_from_product_and_template] failed, code: {code}, msg: {msg}")
    return resp.cta_painter_layers or []


def image_x_enhance(image_urls: List[str], templates: List[str]) -> List[str]:
    req = ImageXEnhanceRequest(image_urls=image_urls, templates=templates)
    logging.info("[image_x_enhance] req: {}".format(req))
    resp: ImageXEnhanceResponse
    code, msg, resp = AiCapabilityCli.ImageXEnhance(req_object=req)
    logging.info("[image_x_enhance] code: {}, msp: {}, resp: {}".format(code, msg, resp))
    if code != 0 or len(resp.success_image_urls) != len(image_urls):
        raise Exception(f"[image_x_enhance] failed, code: {code}, msg: {msg}")
    return resp.success_image_urls or []


# 返回每张图的ocr结果，每张图的结果是一个list[dict]，每个dict是一个word的信息，包括text、bounding_box、font_size，bounding_box是[(x,y), (x,y), (x,y), (x,y)] 四个点围起来的区域是文字区域
def image_ocr_to_bbox(image_urls: List[str] = [], image_infos: List[ImageInfo] = None) -> List[List[dict]]:
    if image_infos is not None:
        req = ImageOcrRequest(image_infos=image_infos)
    else:
        req = ImageOcrRequest(image_urls=image_urls)
    resp: ImageOcrResponse
    logging.info("[image_ocr] req: {}".format(req))
    code, msg, resp = AiCapabilityCli.ImageOcr(req_object=req)
    logging.info("[image_ocr] code: {}, msp: {}, resp: {}".format(code, msg, resp))
    if code != 0 or (image_urls and len(resp.results) != len(image_urls)) or (image_infos and len(resp.results) != len(image_infos)):
        raise Exception(f"[image_ocr] failed, code: {code}, msg: {msg}")
    # parsing
    ocr_res = []
    for idx, result in enumerate(resp.results):
        this_img_res = []
        if len(result) < 2:
            continue
        # 没张图的ocr结果
        ocrResDict = json.loads(result)
        wordResDict = ocrResDict["words"]
        # 解析出每个word的信息
        # 有用的是 det_points_abs是bounding box [dict{x:0,y:0}]。 text是文本。 extra["font_size"] 是字体大小
        for word in wordResDict:
            this_img_res.append(
                {
                    "text": word.get("text", ""),
                    "bounding_box": [
                        (point["x"], point["y"]) for point in word["det_points_abs"] or []
                    ],  # [(x,y), (x,y), (x,y), (x,y)] 四个点围起来的区域是文字区域
                    "font_size": word["extra"].get("font_size", ""),
                }
            )
        ocr_res.append(this_img_res)
    return ocr_res or []


def image_aesthetic_scoring(image_urls: List[str] = None, image_infos: List[ImageInfo] = None) -> List[float]:
    """
    图片美学评分能力
    Parameters:
    - image_urls (List[str]): 输入图片url列表
    - image_infos (List[ImageInfo]): 和image_urls二选一
    Returns:
    - List[float]: 美学评分列表，按输入顺序返回
    """
    resp: ImageAestheticScoringResponse
    req = ImageAestheticScoringRequest(image_urls=image_urls, image_infos=image_infos)
    logging.info(f"[image_aesthetic_scoring] req: {req}")
    code, msg, resp = AiCapabilityCli.ImageAestheticScoring(req_object=req)
    logging.info(f"[image_aesthetic_scoring] code: {code}, msg: {msg}, resp: {resp}")
    if not resp or code != 0:
        raise err.WithCodeError(err.ErrCodeAiCapabilityError, f"[image_aesthetic_scoring] failed, code: {code}, msg: {msg}")

    return resp.aesthetic_score or []


def image_auto_crop(
    image_urls: List[str] = None,
    image_infos: List[ImageInfo] = None,
    ratio: float = 0.5625,
    enable_face_det: int = 1,
    enable_saliency_det: int = 1,
    enable_ocr_det: int = 0,
    enable_cartoon_face_det: int = 0,
    keep_most_salient_face: int = 0,
) -> Tuple[List[str], List[ImageInfo]]:
    req = ImageAutoCropRequest(
        image_urls=image_urls,
        image_infos=image_infos,
        aspect_ratio=ratio,
        enable_face_det=enable_face_det,
        enable_saliency_det=enable_saliency_det,
        enable_ocr_det=enable_ocr_det,
        enable_cartoon_face_det=enable_cartoon_face_det,
        keep_most_salient_face=keep_most_salient_face,
    )
    logging.info(f"[image_auto_crop] req: {req}")
    code, msg, resp = AiCapabilityCli.ImageAutoCrop(req_object=req)
    logging.info(f"[image_auto_crop] code: {code}, msg: {msg}, resp: {resp}")
    if not resp or code != 0:
        raise err.WithCodeError(err.ErrCodeAiCapabilityError, f"[image_auto_crop] failed, code: {code}, msg: {msg}")

    return resp.success_image_urls, resp.success_image_infos


def image_super_resolution(
    image_urls: List[str] = None,
    image_infos: List[ImageInfo] = None,
    target_width: int = 720,
    target_height: int = 1280,
    sr_ratio: float = None,
    proportional_denoise: bool = None,
    image_quality: int = 90,
) -> Tuple[List[str], List[bool], List[ImageInfo]]:
    req = ImageSuperResolutionRequest(
        image_urls=image_urls,
        target_width=target_width,
        target_height=target_height,
        image_quality=image_quality,
    )
    logging.info(f"[image_super_resolution] req: {req}")
    code, msg, resp = AiCapabilityCli.ImageSuperResolution(req_object=req)
    logging.info(f"[image_super_resolution] code: {code}, msg: {msg}, resp: {resp}")
    if not resp or code != 0:
        raise err.WithCodeError(err.ErrCodeAiCapabilityError, f"[image_super_resolution] failed, code: {code}, msg: {msg}")

    return resp.success_image_urls, resp.is_do_sr, resp.success_image_infos


if __name__ == "__main__":
    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token  # os.environ['SEC_TOKEN_STRING'] = "toutiao.growth.xenon"
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"
    os.environ["TCE_PSM"] = "ad.creative.image_core_solution"  # toutiao.growth.xenon

    # 人脸检测能力测试
    # url = "https://p16-oec-va.ibyteimg.com/tos-alisg-i-aphluv4xwc-sg/350c903ce5e04fcb98da663aaacf2158~tplv-o3syd03w52-resize-jpeg:1000:1000.image?"
    # print(image_body_face_detect(urls=[url]))

    # # 文案生成能力测试
    # print(generate_text_capability(None, "test"))

    # # 主体分割能力测试
    # url = "https://images.tokopedia.net/img/cache/700/product-1/2018/7/13/1405477/1405477_1379e246-8159-4554-9ae3-86f204d54e05_850_850.jpg"
    # # url = "https://p16-creative-tool-sg.tiktokcdn.com/tos-alisg-i-n2703mo9gi-sg/9b7962c3618a43d29b97c3dd21c9cc45~tplv-n2703mo9gi-webp:538:538.webp"
    # s = time()
    # e = time()
    # print(image_subject_seg(image_urls=[url], only_mask=1, refine_mask=0), f"time = {e - s}")
    # s = time()
    # e = time()
    # print(image_subject_seg(image_urls=[url], refine_mask=1), f'time = {e-s}')
    # s = time()
    # e = time()
    # print(image_subject_seg(image_urls=[url], refine_mask=2), f'time = {e-s}')

    # # OCR能力测试
    # print(text_list)

    # #color_extract能力测试
    # url = "https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20250114336e4113420aab10430ab1a5"
    # print(color_extract(image_urls=[url]))

    # # 文生图能力测试
    # text = "a girl"
    # print(gen_image_by_prompt(text, 720, 1280))

    # ImageParsingSeg Test
    # product: https://sf-tk-sg.ibytedtos.com/obj/ad-creative-sg/resaveImg_1747029672490_3cf3ccb6-9dc5-4361-8db0-ccc1475d2e29
    # bg: https://sf-tk-sg.ibytedtos.com/obj/ad-creative-sg/resaveImg_1747029672459_0d3b1342-12e3-4cd1-9e01-a96ad725f6b3
    # urls = ["https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20250507336eae01822ea48e4c20888f"]
    # print(image_parsing_seg(urls))
    # 人物检测能力测试
    # url = "https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20250122336eb8214640711a4b8d8ccc"
    # url = "https://sf-tk-sg.ibytedtos.com/obj/ad-creative-sg/ai_img_sub_seg_1737481657792_9de922ef-6039-44ca-86cd-3e32e65c7423.png"
    # # url = "https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20250114336e4113420aab10430ab1a5"
    # resp = image_body_face_detect_raw(urls=[url], only_judge_people=True)
    # print(resp)

    print(
        image_super_resolution(
            ["https://sf-tk-sg.ibytedtos.com/obj/ad-creative-sg/ai_img_auto_crop_1751509190365535545_c2c9c48e-78e4-4044-bb44-913517b6277e.jpg"]
        )
    )
