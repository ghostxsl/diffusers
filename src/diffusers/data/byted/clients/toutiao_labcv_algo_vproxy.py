import json
import logging
import hashlib
import traceback
from typing import List, Tuple, Dict
from random import randint
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import byteddps
import bytedlogger
from euler.base_compat_middleware import gdpr_auth_middleware
from overpass_toutiao_labcv_algo_vproxy.clients.rpc.toutiao_labcv_algo_vproxy import ToutiaoLabcvAlgo_VproxyClient
from overpass_toutiao_labcv_algo_vproxy.euler_gen.lab_cv.cv_proxy.idls.idl.labcv.vproxy.common_thrift import AlgoReq, AuthInfo
from overpass_toutiao_labcv_algo_vproxy.euler_gen.lab_cv.cv_proxy.idls.idl.base_thrift import Base

from diffusers.data.byted.config import LabCVAppKey, LabCVAppSecret
import diffusers.data.byted.errno as err
from diffusers.data.byted.tos import url_to_bytes, save_tos
from diffusers.data.byted.middleware import client_logid

bytedlogger.config_default()


LabcvVproxyCli = ToutiaoLabcvAlgo_VproxyClient(cluster="default", transport="ttheader")
LabcvVproxyCli.set_euler_client_middleware(gdpr_auth_middleware)
LabcvVproxyCli.set_euler_client_middleware(client_logid)


def get_auth_info():
    app_key = LabCVAppKey  # "f23f377f8ceb11edbb3e043f72e2b2cc"
    app_secret = LabCVAppSecret  # "f23f37988ceb11edbb3e043f72e2b2cc"
    nonce = randint(0, (1 << 31) - 1)
    timestamp = int(time())
    keys = [str(nonce), str(app_secret), str(timestamp)]
    keys.sort()
    keystr = "".join(keys)
    keystr = keystr.encode("utf-8")  # python3需要这行
    signature = hashlib.sha1(keystr).hexdigest()
    auth_info = AuthInfo(app_key=app_key, timestamp=str(timestamp), nonce=str(nonce), sign=signature)
    return auth_info


def product_classify(image: bytes) -> dict:
    """
    IC商品分类服务
    https://bytedance.larkoffice.com/docx/RnUDdcvRLoaXDvxip1ecXDuonvc
    Parameters:
    - image (bytes): 输入图片二进制数据；
    Returns:
    - dict:
        - prob (float): 识别结果得分
        - category_name (str): 品类中文名
        - category_id (int): 品类id
    """
    req = AlgoReq(req_key="product_classification", binary_data=[image], req_json=json.dumps({}), auth_info=get_auth_info())

    code, msg, resp = LabcvVproxyCli.Process(req_object=req)
    if code != 0:
        raise Exception(f"[product_classify] failed, code: {code}, msg: {msg}")

    result = json.loads(resp.resp_json)["results"][0]

    logging.info(f"[product_classify] success. Res = {result}")
    if result:
        res_dict = {"prob": result[0], "category_name": result[1], "category_id": result[2]}
    else:
        res_dict = {"prob": 0, "category_name": "", "category_id": -1}
        logging.warning("[product_classify] result is emmpty!")

    return res_dict


def product_prompt_optimizer(
    product_category: str, user_prompt: str = "", return_mode: str = "positive_prompt_and_negative_prompt", work_mode: str = "recommended_mode"
) -> dict:
    """
    IC AI商品图自由Prompt
    https://bytedance.larkoffice.com/docx/BrJDdiYNCoAabexrDxOcAiYJnKh
    Parameters:
    - product_category (str): product_classify的识别结果，category_name；
    - user_prompt (str): 用户输入的自定义背景，仅支持英文传入。可以为空字符串''；
    - return_mode (str): ['complete_content', 'positive_prompt_and_negative_prompt', 'positive_prompt']
        此参数决定了返回内容是什么。
    - work_mode (str): ['rewrite_mode', 'recommended_mode', 'quick_recommended_mode','no_rewrite']
        此参数决定了模型工作在什么模式上。
    Returns:
    - dict:
        - Overall Description (str): 用于文生图的positive prompt；
        - Overall Description Non-decorative (str): positive prompt去掉装饰，展示用；
        - Negative Phrase (str): 用于文生图的negative prompt；
    """
    req = AlgoReq(
        req_key="prompt_optimizer_backdrop",
        req_json=json.dumps(
            {"product_category": str(product_category), "user_prompt": user_prompt, "return_mode": return_mode, "work_mode": work_mode}
        ),
        auth_info=get_auth_info(),
    )

    code, msg, resp = LabcvVproxyCli.Process(req_object=req)
    if code != 0:
        raise Exception(f"[product_prompt_optimizer] failed, code: {code}, msg: {msg}")

    result = json.loads(resp.resp_json)
    if result["status_code"] != 1000:
        raise Exception(f"[product_prompt_optimizer] failed, code: {result['status_code']}")
    prompt_dict = result["prompt_dict"]

    return prompt_dict


def backgen_ic_raw(
    image: bytes, preset_scene: str, mode_type: int = 0, prompt: str = "", neg_prompt: str = "", rgb: List[int] = None, num: int = 1
) -> List[bytes]:
    """
    IC AI商品图换背景-V3
    流量接入请提前沟通资源
    https://bytedance.larkoffice.com/docx/HVfadW53WovPEsxhEOEcHNHinXf
    Parameters:
    - image (bytes): 输入图片二进制数据，需要为RGBA格式；
    - mode_type (int): 0 = 预设场景换背景，1 = 自由prompt换背景，2 = 纯色换背景；
    - preset_scene (str): 预设场景的名称，比如'v3_kitchen'，mode_type为0时，需要给该参数赋值；
    - prompt (str): 自由prompt中正向提示词，mode_type为1时，需要给该参数赋值；
    - neg_prompt (str): 自由prompt中负向提示词，mode_type为1时，需要给该参数赋值；
    - rgb (List[int]): 纯色换背景的颜色输入，比如[255, 0, 0]，list中的数字依次代表r、g、b三个数值。mode_type为2时，需要给该参数赋值。
    - num (int): 一次请求生成图片的数量，默认值为1，大并发请提前沟通资源。
    Returns:
    - List[bytes]: 结果图片二进制list，长度为num
    """

    def process_request(image: bytes, preset_scene: str, mode_type: int, prompt: str, neg_prompt: str, rgb: List[int]):
        req = AlgoReq(
            req_key="bg_img_switch_bm",
            binary_data=[image],
            req_json=json.dumps({"mode_type": mode_type, "preset_scene": preset_scene, "prompt": prompt, "neg_prompt": neg_prompt, "rgb": rgb}),
            auth_info=get_auth_info(),
        )

        code, msg, resp = LabcvVproxyCli.Process(req_object=req)
        if code != 0:
            raise Exception(f"[backgen_ic_raw] failed, code: {code}, msg: {msg}")

        return resp.binary_data[0]

    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_request, image, preset_scene, mode_type, prompt, neg_prompt, rgb) for _ in range(num)]
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                traceback.print_exc()
                results.append(None)
                logging.warning(f"[backgen_ic_raw] An error occurred while batch infer: {e}")

    if all(result is None for result in results):
        raise Exception("[backgen_ic_raw] batch infer failed.")

    return results


def image_reframe(
    image_url: str,
    aspect_ratio: float = 0.5625,  # target aspect ratio of reframed image
    enable_face_det: int = 1,  # 0 or 1 indicates whether to enable human face detection, 1 by default
    enable_saliency_det: int = 1,
    enable_ocr_det: int = 0,
    enable_cartoon_face_det: int = 0,
    enable_pet_face_det: int = 0,
    enable_aesthetic_crop: int = 0,
    keep_most_salient_face: int = 0,
) -> Tuple[str, Dict]:
    # https://ic-aip.arcosite.bytedance.com/7689/227240
    image_bytes = url_to_bytes(image_url)
    req = AlgoReq(
        req_key="image_reframe",
        binary_data=[image_bytes],
        req_json=json.dumps(
            {
                "aspect_ratio": aspect_ratio,
                "enable_face_det": enable_face_det,
                "enable_saliency_det": enable_saliency_det,
                "enable_ocr_det": enable_ocr_det,
                "enable_cartoon_face_det": enable_cartoon_face_det,
                "enable_pet_face_det": enable_pet_face_det,
                "enable_aesthetic_crop": enable_aesthetic_crop,
                "keep_most_salient_face": keep_most_salient_face,
            }
        ),
        Base=Base(Caller="ad.creative.image_core_solution"),
    )
    code, msg, resp = LabcvVproxyCli.Process(req_object=req)
    if code != 0:
        raise Exception(f"[image_reframe] failed, code: {code}, msg: {msg}")

    resp_url = save_tos(resp.binary_data[0], folder_name="image_reframe")
    resp_dict = json.loads(resp.resp_json)

    return resp_url, resp_dict


def img2img_inpainting(image: bytes, mask: bytes) -> bytes:
    # https://ic-aip.arcosite.bytedance.com/7689/53oqgn32
    req = AlgoReq(
        req_key="img2img_inpainting",
        binary_data=[image, mask],
        req_json=json.dumps(
            {
                "max_height": 1024,
                "max_width": 1024,
            }
        ),
        auth_info=get_auth_info(),
    )

    code, msg, resp = LabcvVproxyCli.Process(req_object=req)

    if code != 0:
        raise Exception(f"[img2img_inpainting] failed, code: {code}, msg: {msg}")

    return resp.binary_data[0]


def img2img_outpainting(
    image: bytes, mask: bytes, max_height=1920, max_width=1920, custom_prompt="", top=None, bottom=None, left=None, right=None
) -> bytes:
    try:
        req = AlgoReq(
            req_key="img2img_outpainting",
            binary_data=[image, mask],
            req_json=json.dumps(
                {
                    "max_height": max_height,
                    "max_width": max_width,
                    "custom_prompt": custom_prompt,
                    "top": top,
                    "bottom": bottom,
                    "left": left,
                    "right": right,
                }
            ),
            auth_info=get_auth_info(),
        )

        code, msg, resp = LabcvVproxyCli.Process(req_object=req)
    except Exception as e:
        logging.error(f"[img2img_outpainting] failed, code: {code}, msg: {msg}, error: {e}")
        raise err.WithCodeError(err.ErrCodeAigcImageError, f"[img2img_outpainting] failed, code: {code}, msg: {msg}")
    if code != 0:
        raise err.WithCodeError(err.ErrCodeAigcImageError, f"[img2img_outpainting] failed, code: {code}, msg: {msg}")

    return resp.binary_data[0]


def image_edit(image: bytes, prompt: str) -> bytes:
    """使用seed edit v1.6渲染"""

    req_json_dict = {
        "prompt": prompt,
        "req_schedule_conf": "poster",
        "use_pre_llm": True,
        "model_version": "seededit_v1.6",
        "seed_vlm_version": "test_1.5_pro_32k_250115",
    }

    req = AlgoReq(
        req_key="seededit_v16",
        binary_data=[image],
        req_json=json.dumps(req_json_dict),
        auth_info=get_auth_info(),
    )

    code, msg, resp = LabcvVproxyCli.Process(req_object=req)

    if code != 0:
        raise Exception(f"[edit_image] failed, code: {code}, msg: {msg}")

    return resp.binary_data[0]


def call_dreamposter(
    image_byte: bytes,
    prompt: str,
    is_pe=True,
    negative_prompt="nsfw",
    condition_timestep_scale=0.7,
    face_aug=False,
    width=1024,
    height=1024,
    steps=25,
    guidance=2.5,
) -> bytes:
    """seed dreamposter接口"""
    if is_pe:
        pe_req_json_dict = {
            "prompt": prompt,
        }
        pe_req = AlgoReq(
            req_key="dreamposter_pe",
            binary_data=[image_byte],
            req_json=json.dumps(pe_req_json_dict),
            auth_info=get_auth_info(),
        )

        code, msg, resp = LabcvVproxyCli.Process(req_object=pe_req)

        if code != 0:
            raise Exception(f"[dreamposter_pe] failed, code: {code}, msg: {msg}")

        output_dict_str = resp.resp_json
        output_dict = json.loads(output_dict_str)
        print(f"output_dict: {output_dict}")  # 记得删除
        prompt = output_dict.get("output_dict", {}).get("pe_result", prompt)
        print(f"prompt_PE: {prompt}")  # 记得删除

    req_json_dict = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "condition_timestep_scale": condition_timestep_scale,
        "face_aug": face_aug,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": guidance,
    }
    req = AlgoReq(
        req_key="img2img_seed3_dreamposter_jm",
        binary_data=[image_byte],
        req_json=json.dumps(req_json_dict),
        auth_info=get_auth_info(),
    )

    code, msg, resp = LabcvVproxyCli.Process(req_object=req)

    if code != 0:
        raise Exception(f"[img2img_seed3_dreamposter_jm] failed, code: {code}, msg: {msg}")

    return resp.binary_data[0]


def call_seed_text_to_image(prompt: str, schedule_conf="std", use_pe=True, model_version="general_v3.0_18b") -> bytes:
    # use seed3.0 text to image https://bytedance.larkoffice.com/docx/WWyFdrrajoGHGex26pEcBJJanwe
    # auth_info
    auth_info = get_auth_info()

    # req_json
    req_json_dict = {
        "prompt": prompt,  # "a pizza with text pizza on image",
        "req_schedule_conf": schedule_conf,  # "poster"/std,
        "use_pre_llm": use_pe,  # 开启prompt优化
        "model_version": model_version,
    }
    req_json = json.dumps(req_json_dict)

    # client = euler.Client(VisionService, "tcp://[fdbd:dc02:22:188::218]:10842", timeout=30)  # 对方在CN部署，只能先这么凑活
    req = AlgoReq(req_key="high_aes_general_v30l", req_json=req_json, auth_info=auth_info)
    logging.info(f"req:{req}")
    code, msg, seed3Resp = LabcvVproxyCli.Process(req_object=req)
    if code != 0:
        raise err.WithCodeError(err.ErrCodeAigcImageError, f"[seed3 t2i] failed, code: {code}, msg: {msg}")
    return seed3Resp.binary_data[0]


def image_parsing(image_byte: bytes) -> Tuple[dict, dict]:
    # return template_json, source_dict
    # source_dict::: bg/product/bg_prompt
    req = AlgoReq(
        req_key="tce_poster_parser_v2",
        binary_data=[image_byte],
        req_json=json.dumps(
            {
                "is_return_prompt": False,
                "is_v2": True,
            }
        ),
        auth_info=get_auth_info(),
    )
    code, msg, resp = LabcvVproxyCli.Process(req_object=req)
    if code != 0 or resp is None or resp.resp_json is None:
        raise Exception(f"[image_parsing] failed, code: {code}, msg: {msg}")
    for item_list in json.loads(resp.resp_json).get("templates", []):
        templ = item_list.get("template", "")
        sources = item_list.get("image_resources", {})
        # get 1 only
        return json.loads(templ) if templ else {}, {
            "bg": sources.get("s_bg.png"),
            "product": sources.get("s_prod.png"),
            "bg_prompt": item_list.get("bg_prompt"),
            "bg_layer_id": item_list.get("bg_layer_id"),
            "prod_layer_id": item_list.get("prod_layer_id"),
        }
    return {}, {}


def text2design(image_byte: bytes, prompt: str, produce_mode: str = "matting"):
    # return template_json, source_dict
    # source_dict::: bg/product/bg_prompt
    req_json_str = json.dumps(
        {
            "produce_mode": produce_mode,  # textol, matting, nomatting, resize
            "canvas_size": [1024, 1024],
            # "canvas_size": [720, 1280],  #临时改为9:16
            "prompt": prompt,
            "retri_temp_num": 1,
            "is_simplify": False,
        }
    )
    logging.info(f"\n\n请求参数: {req_json_str}\n\n")

    req = AlgoReq(
        req_key="text2design_pipeline_simple_new",
        binary_data=[image_byte],
        req_json=req_json_str,
        auth_info=get_auth_info(),
    )
    code, msg, resp = LabcvVproxyCli.Process(req_object=req)
    logging.info(f"返回结果 code:{code}, msg:{msg}")
    if code != 0 or resp is None or resp.resp_json is None:
        raise Exception(f"[image_parsing] failed, code: {code}, msg: {msg}")

    return resp.binary_data[0], resp.resp_json


def text2design_pe(image_byte: bytes, prompt="generate a poster", prompt_style="poster"):
    # return template_json, source_dict
    # source_dict::: bg/product/bg_prompt
    req_json_str = json.dumps(
        {
            "produce_mode": "pe",
            "canvas_size": [1024, 1024],
            "prompt": prompt,
            "retri_temp_num": 1,
            "prompt_style": prompt_style,
            "pe_mode": "t2d",  # t2d, ftd
        }
    )

    print(f"\n\n请求参数: {req_json_str}\n\n")

    req = AlgoReq(
        req_key="text2design_pipeline_simple_new",
        binary_data=[image_byte],
        req_json=req_json_str,
        auth_info=get_auth_info(),
    )
    code, msg, resp = LabcvVproxyCli.Process(req_object=req)
    print(f"返回结果 code:{code}, msg:{msg}")
    if code != 0 or resp is None or resp.resp_json is None:
        raise Exception(f"[image_parsing] failed, code: {code}, msg: {msg}")

    try:
        resp_json_object = json.loads(resp.resp_json)
        pe_prompt = None
        pe_prompt = resp_json_object.get("results").get("templates")[0].get("prompt")
    except Exception as e:
        print(f"[text2design_pe] failed, code: {code}, msg: {msg}, exception: {e}")
        raise Exception(f"[text2design_pe] failed, code: {code}, msg: {msg}, exception: {e}")

    return pe_prompt


def aigc_high_aesthetic_etta_text2img(
    prompt: str,
    model_version: str = "etta_v2.1_L",
    req_schedule_conf: str = "etta_pe",
    negative_prompt: str = "nsfw, nude, smooth skin, unblemished skin, mole, low resolution, blurry, worst quality, mutated hands and fingers, poorly drawn face, bad anatomy, distorted hands, limbless, 国旗, national flag.",
    seed: int = -1,
    scale: float = 5.5,
    guidance_rescale_factor: float = 0.5,
    use_pre_llm: bool = True,
    ddim_steps: int = 16,
    width: int = 512,
    height: int = 512,
    use_sr: bool = False,
    sr_seed: int = -1,
    sr_strength: float = 0.4,
    sr_scale: float = 3.5,
    sr_steps: int = 10,
    cluster: str = "default",
) -> bytes:
    """
    AIGC-高美感2.1-9B-ETTA 生图服务
    基于文本提示词生成图像

    Parameters:
    - prompt (str): 用于生成图像的提示词
    - model_version (str): 模型版本名称，默认为"etta_v2.1_L"
    - req_schedule_conf (str): 调度参数，默认为"etta_pe"
    - negative_prompt (str): 用于不想要生成效果的提示词
    - seed (int): 随机种子，-1为随机种子；其他为指定随机种子，取值范围[0, 2^31-1)
    - scale (float): 影响文本描述的程度，取值范围[1, 10]
    - guidance_rescale_factor (float): 影响画面的亮度和文本相应度
    - use_pre_llm (bool): 是否开启PE，如果手动调用PE，则需要设置成False
    - ddim_steps (int): 生成图像的步数，取值范围[1-200]，推荐取值范围[1,50]，耗时与步数成正比
    - width (int): 生成图像的宽（超分前大小），取值范围[512, 1024]
    - height (int): 生成图像的高（超分前大小），取值范围[512, 1024]
    - use_sr (bool): True：文生图+AIGC超分；False：文生图
    - sr_seed (int): 超分模型随机种子，-1为不随机种子；其他为指定随机种子，当use_sr开启时有效
    - sr_strength (float): 只在超分模型生效，取值范围[0.0, 1.0]
    - sr_scale (float): 在超分模型上，影响文本描述的程度，取值范围[1, 10]
    - sr_steps (int): 超分模型生成图像的步数，取值范围[1-200]，推荐取值范围[1,50]，耗时与步数成正比
    - cluster (str): 集群名称，默认为"default"，可选值包括"即梦"、"CapCut"等

    Returns:
    - bytes: 生成的图像二进制数据
    """
    # 构建请求参数
    req_json_dict = {
        "prompt": prompt,
        "model_version": model_version,
        "req_schedule_conf": req_schedule_conf,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "scale": scale,
        "guidance_rescale_factor": guidance_rescale_factor,
        "use_pre_llm": use_pre_llm,
        "ddim_steps": ddim_steps,
        "width": width,
        "height": height,
        "use_sr": use_sr,
        "sr_seed": sr_seed,
        "sr_strength": sr_strength,
        "sr_scale": sr_scale,
        "sr_steps": sr_steps,
    }

    # 创建请求
    req = AlgoReq(
        req_key="high_aes_scheduler_svr_etta_9b_v2.0",  # 根据集群选择不同的req_key
        req_json=json.dumps(req_json_dict),
        auth_info=get_auth_info(),
    )

    # 发送请求
    logging.info(f"[aigc_high_aesthetic_etta_text2img] Request: {req_json_dict}")
    code, msg, resp = LabcvVproxyCli.Process(req_object=req)

    # 处理响应
    if code != 0:
        error_msg = f"[aigc_high_aesthetic_etta_text2img] failed, code: {code}, msg: {msg}"
        logging.error(error_msg)
        raise err.WithCodeError(err.ErrCodeAigcImageError, error_msg)

    # 返回生成的图像二进制数据
    logging.info(f"[aigc_high_aesthetic_etta_text2img] success. request_id: {json.loads(resp.resp_json).get('request_id', '')}")
    return resp.binary_data[0]


def aigc_high_aesthetic_etta_rephreaser(prompt: str, fallback_to_prompt: bool = True, cluster: str = "default") -> str:
    """
    AIGC-高美感2.1-9B-ETTA Rephreaser服务
    文本改写/增强服务

    Parameters:
    - prompt (str): 需要改写/增强的原始文本
    - fallback_to_prompt (bool): 当推理失败的时候返回原始prompt，默认True
    - cluster (str): 集群名称，默认为"default"，可选值包括"即梦"、"CapCut"等

    Returns:
    - str: 改写/增强后的文本
    """
    # 构建请求参数
    req_json_dict = {"prompt": prompt, "fallback_to_prompt": fallback_to_prompt}

    # 创建请求
    req = AlgoReq(
        req_key="high_aes_rephraser_etta_9b",  # 根据集群选择不同的req_key
        req_json=json.dumps(req_json_dict),
        auth_info=get_auth_info(),
    )

    # 发送请求
    logging.info(f"[aigc_high_aesthetic_etta_rephreaser] Request: {req_json_dict}")
    code, msg, resp = LabcvVproxyCli.Process(req_object=req)

    # 处理响应
    if code != 0:
        error_msg = f"[aigc_high_aesthetic_etta_rephreaser] failed, code: {code}, msg: {msg}"
        logging.error(error_msg)
        if fallback_to_prompt:
            logging.warning(f"[aigc_high_aesthetic_etta_rephreaser] Fallback to original prompt: {prompt}")
            return prompt
        raise err.WithCodeError(err.ErrCodeAigcImageError, error_msg)

    # 从响应中提取结果
    try:
        result = json.loads(resp.resp_json).get("rephaser_result", "")
        if not result and fallback_to_prompt:
            logging.warning(f"[aigc_high_aesthetic_etta_rephreaser] Empty result, fallback to original prompt: {prompt}")
            return prompt
        return result
    except Exception as e:
        error_msg = f"[aigc_high_aesthetic_etta_rephreaser] Failed to parse response: {e}"
        logging.error(error_msg)
        if fallback_to_prompt:
            return prompt
        raise Exception(error_msg)


if __name__ == "__main__":
    import os

    # TCE ACL鉴权
    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"
    os.environ["TCE_PSM"] = "ad.creative.image_core_solution"

    url = "https://images.ctfassets.net/a3qyhfznts9y/40UkIjB5kRJegBg6AsTQgc/255f3dc5f6c6061152283ad877bab72e/Thermo_Pinpad.png"
    print(image_reframe(url))

    # # 测试商品分类服务 product_classify
    # from biz.common.network.downloader import url_to_bytes
    # url = 'https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20241202336e9dac4aa8467a4ab3ac36'
    # img_bytes = url_to_bytes(url)
    # print('product_classify result: ', product_classify(img_bytes))

    # # 测试商品图自由Prompt
    # resp =  product_prompt_optimizer(
    #     product_category = '香水',
    #     user_prompt = '',
    #     return_mode = 'positive_prompt_and_negative_prompt',
    #     work_mode = 'rewrite_mode'
    # )
    # print(resp)

    # # 测试换背景服务
    # from biz.common.network.downloader import url_to_bytes, bytes_to_image

    # url = "https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20241202336e9dac4aa8467a4ab3ac36"
    # img_bytes = url_to_bytes(url)
    # result = backgen_ic_raw(image=img_bytes, mode_type=0, preset_scene="v3_Leaves", prompt="", neg_prompt="", num=1)
    # print(len(result), type(result[0]))
    # bytes_to_image(result[0]).save("ic_backgen_test.png")

    # 测试抹除服务

    # url = "https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20250114336ee9375f08db0241bb9c5e"
    # image_bytes = image_to_bytes(url_to_image(url))
    # seg_resp = image_subject_seg(image_urls=None, image_infos=[ImageInfo(URL=url)])
    # image_url = seg_resp.success_image_infos[0].URL
    # image_pil = url_to_image(image_url)
    # mask_pil = get_alpha_channel(image_pil)
    # mask_bytes = image_to_bytes(mask_pil)
    # inpaint_bytes = img2img_inpainting(image_bytes, mask_bytes)
    # print(save_tos(inpaint_bytes, make_key()))

    # 测试parsing服务
    # url = "https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20250507336e59275bdb440644e3ada0"
    # image_bytes = image_to_bytes(url_to_image(url))
    # template_json, source_dict = image_parsing(image_bytes)
    # print(template_json, source_dict)

    # import requests
    # # 下载product_url到本地
    # if source_dict.get("product"):
    #     product_resp = requests.get(source_dict.get("product"))
    #     with open("product.png", "wb") as f:
    #         f.write(product_resp.content)
    # # 下载product_url到本地
    # if source_dict.get("bg"):
    #     bg_resp = requests.get(source_dict.get("bg"))
    #     with open("bg.png", "wb") as f:
    #         f.write(bg_resp.content)
    # print("1")

    # 测试seed3.0
    # respImgByte = call_seed_text_to_image(
    #     prompt="Anime-style illustration of a cute tabby cat with big, expressive eyes, sitting upright. The cat is holding a plastic cup with a straw, appearing to drink from it. The background is a simple, light-colored wall, emphasizing the cat's playful and curious expression. The perspective is at eye level, capturing the cat's face and upper body prominently in the frame, with the cup occupying a significant portion of the foreground. The cat's ears are perked up, and its gaze is directed slightly upwards, adding to its endearing demeanor.",
    #     schedule_conf="std",
    # )
    # with open("product.png", "wb") as f:
    #     f.write(respImgByte)
    # print("1")
    # # 测试AIGC-高美感2.1-9B-ETTA生图服务
    # img_bytes = aigc_high_aesthetic_etta_text2img(
    #     prompt="Desgin big-character poster with promotional text 'Your Gateway to Dubai’s Finest – Hotels.com Delivers Unmatched Comfort! ' for travel industry. The image should be catching eyes and the text should be clear and easy to read. Do not add not given text in the image.",
    #     width=1280,
    #     height=720,
    # )
    # with open("etta_text2img_result.png", "wb") as f:
    #     f.write(img_bytes)
    # print("AIGC-高美感2.1-9B-ETTA生图服务测试完成")
    # # 测试AIGC-高美感2.1-9B-ETTA Rephreaser服务
    # enhanced_text = aigc_high_aesthetic_etta_rephreaser(
    #     prompt="This is a poster for a smart pet feeder during Black Friday."
    # )
    # print("AIGC-高美感2.1-9B-ETTA Rephreaser服务测试结果:")
    # print(enhanced_text)

    from biz.common.network.downloader import url_to_bytes
    from biz.infra.clients.tos import save_tos, make_key

    image_bytes = url_to_bytes("https://sf16-muse-va.ibytedtos.com/obj/ad-creative/1750388052.8052356.png?t=1751534954725")
    prompt = "Add text at the top of the picture: New Spring Style, and add a circular discount mark 80% in the lower right corner。*Text cannot be overlaid on the product*"
    # prompt = "给商品增加三个文字，顶部展示：80%优惠，左下角展示：巨大折扣，右下角展示：新品"

    print(f"len:{len(image_bytes)}")

    res_bytes = call_dreamposter(image_byte=image_bytes, prompt=prompt, is_pe=True)
    res_url = save_tos(res_bytes, make_key())
    print(res_url)
