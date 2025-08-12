import copy
from typing import Any, Dict, List, Callable
import traceback
import logging
from retrying import retry

from diffusers.data.byted.tos import url_to_base64, base64_to_md5, get_image_size_mb, resize_image
from diffusers.data.byted.clients.azure_mllm import MLLMClient
from diffusers.data.byted.clients.abase.abase_client import URL_UNDERSTANDING_CACHE_CLIENT
from diffusers.data.byted.decorator import cache, timer
import diffusers.data.byted.errno as err
from diffusers.data.byted.parallel import execute_concurrently

from overpass_ad_creative_image_core_solution.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.capability_thrift import ImageInfo


GPT4oCli = MLLMClient(model_name="gpt-4o-2024-11-20")


IMG_BASIC_CLS_PROMPT = """
# Role  
You are an experienced image annotator with knowledge of multiple languages.

# Goals 
Your task is to annotate an images with corresponding label based on the target product info.

# Input
- Target product info: {product_info}
- An image: 

# Instructions
1. For each image, choose one label from "target_product", "product_user", "irrelevant_person", "other_product", "general_scene", "document", "promotional_image", "low_quality" and "useless". They are defined as follows:
    - target_product: The image directly contains the target product which is consistent with the input product category. It maybe close-ups of the product, template advertisements of the product, or everyday usage scenarios of the product. Particularly: For images of short play covers, when the cover and the product name are consistent, the image is considered to be "target product".
    - product_user: The image is a photo of real persons clearly related to the product. They are using or showcasing the target product. Or the image shows the changes in users before and after using the product, or it highlights the target area of the product's effect on the human body (such as shampoo products, showcasing the hair on a person's head). But simple character portraits need to be excluded.
    - irrelevant_person: The image is a candid photo of a real person, with their face visible. But there is no information related to the target product in the image.
    - other_product: The image displays non-target products which is inconsistent with the input product category. 
    - general_scene: The image shows some ordinary scenes or objects. The image quality is high, but it does not appear to be related to the target product.
    - document: The image's visual focus is on text, which occupies more than 50% of the image space, and serves as the primary source of information or visual emphasis. Even if a product is present, if the main purpose and visual focus of the image are to provide detailed textual information (such as product descriptions, usage instructions, or customer testimonials), then the image should be classified as "document." In this scenario, the product display is secondary, and the text is the main element attracting attention.
    - promotional_image: The picture does not contain any products, but provides some general discount or promotional information in an eye-catching way. It must focus primarily on visual design, such as graphics, icons, or artistic elements, with minimal textual content.
    - low_quality: The image is blurry, of poor quality, or too dark to be recognizable, or looks like it was taken from a random snapshot in a GIF, resulting in meaningless camera movements and motion blur. Alternatively, the image has a prominent video play button, or it was originally a square image but has been extended into a wide image with large black borders or Gaussian blur on both sides.
    - useless: The image shows meaningless cartoon characters, logos, signs, icons, elements unrelated to the target product.
2. When making relevance judgments, extract the object category from the product_info, ignoring information including brand, color and model.

# Output (JSON)
{{
    "cls": "<one_of_the_above_labels>"
}}
"""


@timer
def image_classify(url: str, product_info: dict, source_url: str = None) -> str:
    """Classify a single image based on its relevance to the target product

    Args:
        url (str): Image URL or base64
        product_info (dict): Dictionary containing product details (e.g., category, name, description)
        source_url (str): 原始落地页URL，用于缓存key的生成。不传缓存无法生效
    Returns:
        str: Classification label, One of the following:
        ["target_product", "product_user", "irrelevant_person", "other_product", "general_scene",
         "document", "promotional_image", "low_quality", "useless"]
    """
    logging.info("[image_classify] starts...")

    # incorrect brand and other information will cause confusion in judging relevance, so exclude it
    info_keys = ["productName", "title", "description", "sellingPoint"]
    # filter out None/empty values to avoid confusion in the prompt
    filtered_product_info = {key: value for key, value in product_info.items() if key in info_keys and value and str(value).strip()}
    user_prompt = IMG_BASIC_CLS_PROMPT.format(product_info=filtered_product_info)

    @retry(stop_max_attempt_number=3, wait_fixed=1000)  # 3 tries, 1 s apart
    def _classify_once(url: str, prompt: str) -> str:
        result = GPT4oCli.make_image_json_request(
            "You are an experienced image annotator with knowledge of multiple languages.", prompt, [], [url], 20
        )
        return result["cls"].lower()

    try:
        tag = _classify_once(url, user_prompt)
        logging.info(f"[image_classify] result: {tag}")
        return tag
    except Exception as e:
        logging.error(f"[image_classify] failed after retries: {str(e)}")
        return "invalid_url"


def get_image_classify_cache_func(enable_cache: bool = True, cache_ttl: float = 7, source_url: str = None) -> Callable[[str, dict], str]:
    """
    根据入参返回一个是否带缓存的image_classify函数

    Args:
        enable_cache (bool): 是否使用缓存，默认为True。当前主要缓存gpt打标结果
        cache_ttl (float): 缓存时间，默认为7天
        source_url (str): 原始落地页URL，用于缓存key的生成。不传缓存无法生效
    Returns:
        Callable[[str, dict], str]: 带缓存的image_classify函数
    """
    if not enable_cache or not source_url:
        return image_classify

    def get_key_str(url: str, product_info: dict, source_url: str = None) -> str:
        # base64, 直接构成键值
        if url.startswith("data:image"):
            return url + str(source_url)
        # 否则，先转base64，再构成键值
        else:
            return url_to_base64(url, gpt_prefix=True) + str(source_url)

    image_classify_cache = cache(
        cli=URL_UNDERSTANDING_CACHE_CLIENT,
        key_func=get_key_str,
        version="v0",
        ttl_days=cache_ttl,
        timeout=7,
        interval=1,
        compress=False,
        enable=enable_cache,
    )(image_classify)

    return image_classify_cache


# version 0801
PROMPT = """你是一名专业的图像内容分析师。请分析输入图片是否适合用作"垫图"进行AI重绘或Inpainting处理。
【核心判断原则】
垫图必须是清晰、完整、主体突出的商品展示图，适合进行主体替换或背景更换操作。
垫图的作用是通过Inpainting生成真实拍摄感的照片，所以要考虑画面主体被抠图后Inpainting后的效果真实自然。
而且抠图后，我可能会调整抠图mask大小来适配画面，所以抠图后的主体最好不要有明显的边缘截断。

【严格的适用标准】
✅ 必须满足的条件：
- 商品100%完整展示，无任何边缘截断
- 商品主体与背景边界清晰，易于分离
- 背景简洁统一（纯色/简单渐变），无复杂场景元素
- 图片为标准商品展示图
- 如有模特，必须完整露出且为商品展示目的
- 商品在画面中占主导地位，清晰可辨
- 视角展现商品全貌
- 除了商品主体包装自带的文本（如logo、配料表等）外，文本不能覆盖或遮挡商品主体，但可以出现在背景部分

❌ 严格排除的情况：
- 营销海报、广告设计图
- 商品任何部分被裁切或截断
- **文字、标签、水印等覆盖在商品主体上方**
- **任何纹理背景（大理石、木纹、布料等）**
- **模糊环境背景（可识别出环境元素）**
- **复杂环境背景（家具、装饰、建筑等）**
- 纯人物照片（非商品展示用途）
- 个人定制商品展示（个人姓名、定制文字等）
- 商品与背景融合度高，边界模糊
- 非常规单侧视角展示（如鞋底图、过度俯视图等）
- 摆放过于人工或不自然

【输出格式】
必须是合法 JSON，无注释
{
	"has_product": boolean,
	"product_fully_visible": boolean,
	"product_quantity": "single|multiple|combo|not_applicable",
	"has_product_truncation": boolean,  // 是否有边缘截断（商品任何部分被画面边界截断，包括数据线、项链、配件等细枝末节）
	"complex_surface": boolean,  // 商品表面是否含有密集文字或复杂纹理
	"background_type": "pure_solid|simple_gradient|other",
	"background_separability": "easy|moderate|difficult",
	"product_prominence": "dominant|secondary|minimal|none",
	"has_human": boolean,
	"has_human_truncation": boolean, // 严格关注人物头部、肢体边缘的截断
	"human_body_type": "full_body|upper_body|partial_limbs|not_applicable",
	"is_poster": boolean, // 包括丰富的文本元素、海报设计、拼接图等逻辑
	"text_overlaps_product": boolean,
	"image_clarity": "clear|moderate|blurred",
	"is_clothing_product": boolean,  // 是否为服装产品（仅指上衣、下装、连体服装，不包括袜子、帽子、鞋子、包包等配饰）
	"clothing_display_type": "flat_lay|3D|hanging|model_worn|not_applicable",  // 服装展示类型（仅当is_clothing_product为true时适用）
	"industry_category": "Fashion_Apparel|Beauty_Personal_Care|Home_Furniture|Food_Beverage|Electronics_Appliances|Health_Wellness|Other|Unidentifiable",
	"is_single_sided_view": boolean, // 非常规角度（鞋底、鞋尾、打印机单侧图等过于平面展示无法判断体积和透视关系的商品）
	"viewing_angle": "front|side|top|bottom|three_quarter|multiple_angles|not_applicable",
	"placement_naturalness": "natural|artificial|staged|not_applicable",
	"is_suitable_for_matting": boolean, // 核心字段
	"primary_reason": "详细判断依据",
	"additional_notes": "其他重要说明"
}

【背景类型简化定义】
- pure_solid: 完全单一颜色的背景，无明显纹理、图案或环境元素
- simple_gradient: 连续变化的色调或者阴影背景，无明显纹理、图案或环境元素
- other: 其他所有背景类型（包括渐变、纹理、环境等），拼接图、海报图需要归纳在other

【is_suitable_for_matting关键检测点】
1. **商品完整性**：检查商品边缘是否完全在画面内
2. **背景复杂度**：是否包含环境、家具、装饰等元素
3. **图片用途**：是否为标准商品展示而非使用场景
4. **商品相关性**：如有product_info，检查图片商品是否与描述匹配
5. **个人化程度**：是否包含个人姓名、定制内容等

【特别注意事项】
1. **零容忍截断**：任何商品边缘被裁切都视为不合格
2. **环境场景排除**：室内环境、使用场景一律排除
3. **人物照片排除**：非商品展示的人物照片一律排除
4. **个人定制排除**：包含个人姓名、定制元素的商品排除
5. **商品信息匹配**：有product_info，必须检查相关性

【背景严格判断标准】
- **pure_solid**: 必须是完全单一颜色，无任何可识别的纹理、图案、阴影变化
- simple_gradient: 仅包含简单轻微的颜色渐变，无任何可识别的纹理、图案，文本对背景的割裂作用不强
- other：任何可见的材质纹理、营销海报、广告设计图都归类other，注意海报、广告设计图等富含文本的，背景割裂的都算做other
- background_separability: 仅当图片包含商品主体，且背景为pure_solid或simple_gradient时，background_separability才能为easy，可以理解为is_suitable_for_matting的宽松标准

【服装产品定义】
- **is_clothing_product**: 仅指核心服装类别
  - ✅ 包括：上衣（T恤、衬衫、外套等）、下装（裤子、裙子等）、连体服装（连衣裙、连体裤等）
  - ❌ 不包括：袜子、帽子、鞋子、包包、围巾、手套、首饰等配饰

- **clothing_display_type**: 仅在is_clothing_product为true时填写
  - flat_lay: 平铺展示（衣服平摊在表面上）
  - three_dimensional: 3D渲染图（计算机生成的立体渲染图像）
  - hanging: 悬挂展示（衣服悬挂在衣架上的实拍图）
  - model_worn: 模特穿着展示
  - not_applicable: 非服装产品时使用

【边缘截断判断标准】
- **has_product_truncation**: 商品主要部分是否被画面边界明显截断
  - ✅ 算作截断：商品主体结构被边界切断、关键部件被截断、明显可见截断痕迹
  - ❌ 不算截断：商品主体完整、仅边缘轻微贴近边界但无明显切断、自然边缘接触画面边界
  - 判断重点：是否有明显的"切断"效果，而非像素级的100%完整

【商品表面复杂度】
- **complex_surface**: 商品表面是否有密集文字或复杂纹理
  - ✅ 包括：商品包装上的密集文字、复杂图案、密集标识等
  - ❌ 不包括：简单logo、纯色、简单图案

【要求】
1. human_body_type如果是full_body，全身包括头顶、脚部必须完整出现在画面内
2. 对于全图是海报、拼接图、或者纯文本，以及看不出商品包装主体的，一律排除
3. 美妆类产品，如果商品主体清晰，背景只是大色块分布，主体可以清晰分辨比较明显，可以认为适用于垫图
4. 删除所有海报图类型的图片，包括含有拼接元素的
5. 加强对商品主体是否被截断的判断精准度，不要误判，注意区分在边缘和边缘截断商品主体的差别

请严格按照以上标准进行判断，确保JSON格式正确。
"""

GPT41Cli = MLLMClient(model_name="gpt-4.1-2025-04-14", api_key="QBaJa0nVj9GYH7469SNlqKvhsuX5InrK_GPT_AK")


def label_single_image(image_url: str) -> Dict[str, Any]:
    try:
        res = GPT41Cli.make_image_json_request("", PROMPT, [], [image_url], 500)
        return res
    except err.WithCodeError as e:
        raise err.WithCodeError(e.code, f"failed to label image: {image_url}, error: {e.message}")
    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"failed to label image: {image_url}, exception: {str(e)}, traceback: {error_trace}"
        raise err.WithCodeError(err.ErrCodeInternalError, error_msg)


# TODO: add product relevance filter
@timer
def label_batch_image(
    image_infos: List[ImageInfo], product_info: Dict = {}, label_mode: int = 0, use_cache: bool = False, cache_ttl: float = 0.5, web_url: str = ""
) -> List[ImageInfo]:
    """
    Performs image labeling with optional product relevance filtering.

    For each image, extracted attributes are stored in its `Extra` dictionary, including:
        - Product attributes: presence, visibility, quantity, etc.
        - Background attributes: type, separability, etc.
        - Human attributes: presence, body type, etc.
        - Quality metrics: clarity, placement, matting suitability, etc.

    Args:
        image_infos (List[ImageInfo]): List of images to process
        product_info (Dict): Product metadata used for relevance filtering
        label_mode (int): Mode for labeling:
            - 0: Image labeling only
            - 1: Product relevance filtering + image labeling
        use_cache (bool): Whether to use cached results for labeling
        cache_ttl (int): Time-to-live for cache entries in days

    Returns:
        List[ImageInfo]: Images with updated `Extra` attributes
    """

    try:
        labeled_image_infos = []
        if label_mode == 1:
            args_list = [(image_info.URL, product_info) for image_info in image_infos]
            image_classify = get_image_classify_cache_func(use_cache, cache_ttl, web_url)
            results = execute_concurrently(image_classify, args_list, len(args_list), fail_fast=False, failure_threshold=0.6)
            for image_info, tag in zip(image_infos, results):
                if tag in ["target_product", "product_user"]:
                    if get_image_size_mb(image_info.URL) > 20.0:
                        resized_url = resize_image(image_info.URL, save_mode=1)
                    else:
                        resized_url = image_info.URL
                    if image_info.Extra is None:
                        image_info.Extra = {}
                    image_info.Extra["cls"] = tag
                    image_info.Extra["resized_url"] = resized_url
                    labeled_image_infos.append(image_info)
        else:
            labeled_image_infos = copy.deepcopy(image_infos)

        args_list = []
        for image_info in labeled_image_infos:
            if hasattr(image_info, "Extra") and isinstance(image_info.Extra, dict) and "resized_url" in image_info.Extra:
                args_list.append((image_info.Extra["resized_url"],))
            else:
                args_list.append((image_info.URL,))
        label_single_image_cache = cache(
            cli=URL_UNDERSTANDING_CACHE_CLIENT,
            key_func=lambda image_url: base64_to_md5(url_to_base64(image_url)),  # image MD5
            version="v0",
            ttl_days=cache_ttl,
            enable=use_cache,
            timeout=10,
        )(label_single_image)
        results = execute_concurrently(label_single_image_cache, args_list, len(args_list), fail_fast=False, failure_threshold=0.6)
        for image_info, res in zip(labeled_image_infos, results):
            if res:
                if image_info.Extra is None:
                    image_info.Extra = {}
                image_info.Extra["has_product"] = res.get("has_product", False)
                image_info.Extra["product_fully_visible"] = res.get("product_fully_visible", False)
                image_info.Extra["product_quantity"] = res.get("product_quantity", "")
                image_info.Extra["has_product_truncation"] = res.get("has_product_truncation", False)
                image_info.Extra["complex_surface"] = res.get("complex_surface", False)

                image_info.Extra["background_type"] = res.get("background_type", "False")
                image_info.Extra["background_separability"] = res.get("background_separability", "")
                image_info.Extra["product_prominence"] = res.get("product_prominence", "")

                image_info.Extra["has_human"] = res.get("has_human", False)
                image_info.Extra["has_human_truncation"] = res.get("has_human_truncation", False)
                image_info.Extra["human_body_type"] = res.get("human_body_type", "")

                image_info.Extra["is_poster"] = res.get("is_poster", False)
                image_info.Extra["text_overlaps_product"] = res.get("text_overlaps_product", False)
                image_info.Extra["image_clarity"] = res.get("image_clarity", "")

                image_info.Extra["is_clothing_product"] = res.get("is_clothing_product", False)
                image_info.Extra["clothing_display_type"] = res.get("clothing_display_type", "")
                image_info.Extra["industry_category"] = res.get("industry_category", "")

                image_info.Extra["is_single_sided_view"] = res.get("is_single_sided_view", False)
                image_info.Extra["viewing_angle"] = res.get("viewing_angle", "")
                image_info.Extra["placement_naturalness"] = res.get("placement_naturalness", "")

                image_info.Extra["is_suitable_for_matting"] = res.get("is_suitable_for_matting", False)
                image_info.Extra["primary_reason"] = res.get("primary_reason", "")
                image_info.Extra["additional_notes"] = res.get("additional_notes", "")
    except err.WithCodeError as e:
        raise err.WithCodeError(e.code, f"failed to label batch images: {e.message}")
    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"failed to label batch images: {str(e)}, traceback: {error_trace}"
        raise err.WithCodeError(err.ErrCodeInternalError, error_msg)
    return labeled_image_infos


if __name__ == "__main__":
    import os

    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"
    os.environ["TCE_PSM"] = "ad.creative.image_core_solution"

    # advid: 7019325263603318785, adgroup, 1831664667088930, adid: 1835313206525154
    web_url = (
        "https://sundaysfordogs.com/?utm_campaign=__CAMPAIGN_NAME__&utm_content=__CID_NAME__&utm_medium=cpc&utm_source=tiktok&utm_term=__AID_NAME__"
    )
    product_info = {
        "title": "Fresh Dog Food Made Easy",
        "brand": "Sundays for Dogs",
        "description": "Sundays offers veterinarian-founded, human-grade air-dried dog food made with 100% meat and superfoods with no synthetics. It requires no prep or refrigeration, preserving nutrients and flavor through low and slow air-drying. Convenient, nutritious, and approved by picky eaters, it provides benefits like improved coat softness, energy, and weight at 40% less cost than frozen fresh food.",
        "language": "en",
        "productName": "Sundays Air-Dried Dog Food",
    }
    image_urls = [
        "https://d3rekvgx2f3gtb.cloudfront.net/images/webp/redesign-v4/home-v4/legacy/turkey-piece.webp",
        "https://d3rekvgx2f3gtb.cloudfront.net/images/webp/redesign-v4/home-v4/legacy/testimonial-bg@2x.webp",
        "https://d3rekvgx2f3gtb.cloudfront.net/images/webp/redesign-v4/home-v4/legacy/step-1.webp",
    ]
    # print(label_batch_image([ImageInfo(URL=url) for url in image_urls], label_mode=0, use_cache=True, cache_ttl=0.5))
    print(
        label_batch_image(
            [ImageInfo(URL=url) for url in image_urls], product_info=product_info, label_mode=1, use_cache=True, cache_ttl=0.5, web_url=web_url
        )
    )
