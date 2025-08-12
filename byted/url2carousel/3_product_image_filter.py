import os
import json
import logging
import concurrent.futures
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image

import diffusers.data.byted.errno as err
from diffusers.data.byted.label import label_batch_image
from diffusers.data.byted.tos import url_to_image_simple, save_tos, image_to_bytes, make_key
from diffusers.data.byted.parallel import execute_concurrently, execute_concurrently_mul_func
from diffusers.data.byted.clients.creative_ai_capability import image_body_face_detect_raw, image_subject_seg
from diffusers.data.byted.clients.ad_aigc_imagegen import calculate_solid_color
from diffusers.data.byted.clients.creative_ai_laplace_gateway import clip_g_image

from overpass_ad_creative_ai_capabilities.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.capability_thrift import ImageInfo


def fetch_image_data(url) -> Dict:
    image = url_to_image_simple(url)
    resize_image = image.resize((512, 512))
    image_bytes = image_to_bytes(resize_image)
    clip_data = clip_g_image(image_bytes)
    image_data = {"url": url, "feature": clip_data["feature"], "resolution": image.size[0] * image.size[1]}
    return image_data


def dedup_within_images_by_clip_g(image_url_list, similarity_threshold=0.94, max_workers=4) -> Tuple[List[str], List[int]]:
    image_data = [None] * len(image_url_list)
    # 使用线程池并发请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(fetch_image_data, url): i for i, url in enumerate(image_url_list)}
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                data = future.result()
                image_data[index] = data
            except Exception as exc:
                url = image_url_list[index]
                logging.error(f"Image {url} generated an exception: {exc}")

    unique_images = []
    unique_indices = []
    for i, current_image in enumerate(image_data):
        if current_image:  # skip None values
            is_unique = True
            for unique_image in unique_images:
                similarity = calculate_similarity(current_image["feature"], unique_image["feature"])
                # print(f"similarity: {similarity}")
                if similarity >= similarity_threshold:
                    if current_image["resolution"] > unique_image["resolution"]:
                        unique_images.remove(unique_image)
                        unique_images.append(current_image)
                    is_unique = False
                    break
            if is_unique:
                unique_images.append(current_image)
                unique_indices.append(i)

    return [image["url"] for image in unique_images], unique_indices


def get_alpha_channel(image_pil: Image.Image) -> Image.Image:
    """
    提取给定PIL图像对象的alpha通道字节值。
    Parameters:
    - image_pil (Image.Image): 一个PIL图像对象。
    Returns:
    - Alpha通道的PIL图像对象。
    """
    # 检查图像模式是否包含alpha通道
    if image_pil.mode not in ("RGBA", "LA"):
        # 如果没有alpha通道，抛出一个错误
        raise ValueError("The image does not have an alpha channel. Check the input or set need_mask = 1")

    # 分离出alpha通道
    alpha_channel = image_pil.split()[-1]  # 获取最后一个通道即alpha通道

    # 创建一个新的图像来保存alpha通道
    # 对于单通道图像，使用 'L' 模式
    alpha_image = Image.new("L", image_pil.size)
    alpha_image.putdata(alpha_channel.getdata())

    return alpha_image


def analyze_mask(mask_pil: Image.Image, is_human: bool, tolerance: int = 2, line_min_length_ratio: float = 0.1) -> dict:
    """
    判断mask是否截断，&判断mask边距
    Parameters:
    - mask_pil (Image.Image): The mask of the image.
    - is_human (bool): Whether the image contains a human. If True, extra truncation detection logic is applied.
    - tolerance (int): 计算被截断边缘的长度，往内推tolerance个pixel进行计算。避免抠图最边缘像素的渐变、不连贯等影响造成计算错误
    - straight_line_th (float): Minimum ratio of continuous foreground pixels along an edge to be considered a straight line.

    Returns:
    - dict: e.g.
        {
            "truncated_detail": {
                "top": False,
                "bottom": False,
                "left": False,
                "right": False
            },
            "distance": {
                "top": 10,
                "bottom": 100,
                "left": 10,
                "right": 100
            },
            "is_truncated": False
        }
    """
    mask_np = np.array(mask_pil.convert("L"))
    mask_np = np.where(mask_np > 127, 255, 0)

    # Calculate distances to the edges
    def find_edge_distances(mask):
        top_distance = np.argmax(mask.sum(axis=1) > 0)
        bottom_distance = mask.shape[0] - np.argmax(mask[::-1].sum(axis=1) > 0) - 1
        left_distance = np.argmax(mask.sum(axis=0) > 0)
        right_distance = mask.shape[1] - np.argmax(mask[:, ::-1].sum(axis=0) > 0) - 1
        return top_distance, bottom_distance, left_distance, right_distance

    top_distance, bottom_distance, left_distance, right_distance = find_edge_distances(mask_np)
    top_truncated_raw = top_distance == 0
    bottom_truncated_raw = bottom_distance == mask_np.shape[0] - 1
    left_truncated_raw = left_distance == 0
    right_truncated_raw = right_distance == mask_np.shape[1] - 1

    # Extract the edge pixel arrays
    top_edge = mask_np[min(top_distance + tolerance, mask_np.shape[0] - 1), :]
    bottom_edge = mask_np[max(bottom_distance - tolerance, 0), :]
    left_edge = mask_np[:, min(left_distance + tolerance, mask_np.shape[1] - 1)]
    right_edge = mask_np[:, max(right_distance - tolerance, 0)]

    #计算wh_ratio
    wh_ratio = (right_distance - left_distance) / (bottom_distance - top_distance)

    # 计算edge线的长度占比
    def straight_line_ratio(edge_pixels):
        # Create a binary mask of the edge pixels
        edge_mask = np.where(edge_pixels > 0, 1, 0)
        # Sum along the possible line direction
        line_strength_ratio = edge_mask.sum() / len(edge_pixels)
        return line_strength_ratio

    top_edge_ratio = straight_line_ratio(top_edge)
    bottom_edge_ratio = straight_line_ratio(bottom_edge)
    left_edge_ratio = straight_line_ratio(left_edge)
    right_edge_ratio = straight_line_ratio(right_edge)

    # 为人体时的特殊逻辑
    if is_human:
        # Determine if each edge is a straight line
        top_truncated = top_truncated_raw or top_edge_ratio>=line_min_length_ratio
        bottom_truncated = bottom_truncated_raw or bottom_edge_ratio>=line_min_length_ratio
        left_truncated = left_truncated_raw or left_edge_ratio>=line_min_length_ratio
        right_truncated = right_truncated_raw or right_edge_ratio>=line_min_length_ratio
    else:
        top_truncated = top_truncated_raw
        bottom_truncated = bottom_truncated_raw
        left_truncated = left_truncated_raw
        right_truncated = right_truncated_raw

    # Return the results
    res_dict = {
        "truncated_detail": {
            "top": bool(top_truncated),
            "bottom": bool(bottom_truncated),
            "left": bool(left_truncated),
            "right": bool(right_truncated),
        },
        # 基于图像边缘判断是否被截断的原始逻辑
        "truncated_detail_raw": {
            "top": bool(top_truncated_raw),
            "bottom": bool(bottom_truncated_raw),
            "left": bool(left_truncated_raw),
            "right": bool(right_truncated_raw),
        },
        "distance": {"top": int(top_distance), "bottom": int(bottom_distance), "left": int(left_distance), "right": int(right_distance)},
        "edge_ratio": {
            "top": float(top_edge_ratio),
            "bottom": float(bottom_edge_ratio),
            "left": float(left_edge_ratio),
            "right": float(right_edge_ratio),
        },
        "is_truncated": bool(top_truncated or bottom_truncated or left_truncated or right_truncated),
        "is_truncated_raw": bool(top_truncated_raw or bottom_truncated_raw or left_truncated_raw or right_truncated_raw),
        "subject_wh_ratio": float(wh_ratio),
    }

    return res_dict


def truncated_judge(url: str, tolerance: int = 2, line_min_length_ratio: float = 0.1) -> dict:
    """
    判断图片是否被截断
    Parameters:
    - url (str): The URL of the image.
    - tolerance (int): 判断是否为人体被截断的图片时，边缘往内推tolerance的距离来判断是否截断
    - line_min_length_ratio (float): Minimum ratio of continuous foreground pixels along an edge to be considered a straight line.
    Returns:
    - dict: e.g.
        {
            "truncated_detail": {
                "top": False,
                "bottom": False,
                "left": False,
                "right": False
            },
            "distance": {
                "top": 10,
                "bottom": 100,
                "left": 10,
                "right": 100
            },
            "is_truncated": False,
            "is_human": False
        }
    """
    try:
        human_resp = image_body_face_detect_raw(urls=[url])
        is_human = human_resp.with_people[0]
        if is_human:
            is_human = human_resp.detected_items[0][0].score > 0.8 or human_resp.detected_items[0][1].score > 0.8

        seg_resp = image_subject_seg(image_urls=[url], refine_mask=2)
        image_url = seg_resp.success_image_urls[0]
        image_pil = url_to_image_simple(image_url)
        mask_pil = get_alpha_channel(image_pil)
        truncated_res = analyze_mask(mask_pil, is_human, tolerance, line_min_length_ratio)
        truncated_res.update({'is_human': is_human})
    except err.WithCodeError as e:
        #无前景/背景情况下，视为截断
        if e.code == err.ErrCodeNoBackgroundImage:
            truncated_res = {
                "truncated_detail": {
                    "top": True,
                    "bottom": True,
                    "left": True,
                    "right": True
                },
                "distance": {
                    "top": 0,
                    "bottom": 0,
                    "left": 0,
                    "right": 0
                },
                "is_truncated": True,
                "is_human": is_human
            }
        else:
            truncated_res = {}

    top = truncated_res.get('distance', {}).get('top', 0)
    bottom = truncated_res.get('distance', {}).get('bottom', image_pil.height)
    left = truncated_res.get('distance', {}).get('left', 0)
    right = truncated_res.get('distance', {}).get('right', image_pil.width)
    subject_image = image_pil.crop([left, top, right, bottom])
    subject_image_url = save_tos(image_to_bytes(subject_image), make_key())
    truncated_res.update({'subject_image_url': subject_image_url})

    return truncated_res


def truncated_batch_judge(image_url_list: List[str]) -> List[dict]:
    """
    批量判断图片是否被截断
    :param image_url_list: 图片url列表
    :return: 截断信息dict列表
    """
    truncated_res_list = execute_concurrently(truncated_judge, args_list = [(url, ) for url in image_url_list], max_workers=20)
    return truncated_res_list


def white_batch_judge(image_url_list: List[str]) -> List[bool]:
    """
    批量判断图片是否为白图
    :param image_url_list: 图片url列表
    :return: 是否白图bool list
    """
    white_resp_list = execute_concurrently(calculate_solid_color, args_list = [(url, ) for url in image_url_list], max_workers=20)
    white_bool_list = [res.if_solid for res in white_resp_list]
    return white_bool_list


def input_image_tagging(image_infos: List[ImageInfo]) -> List[ImageInfo]:
    """
    对所有相关性图片做处理，得到其标签，截断信息，长宽比
    :param image_infos: 图片信息列表
    :return: 处理后的图片List，标签信息放在Extra中
    """
    res_dict = execute_concurrently_mul_func(
        {
            label_batch_image: [image_infos, {}, 0, True, 30],
            truncated_batch_judge: [[v.URL for v in image_infos]],
            white_batch_judge: [[v.URL for v in image_infos]]
        }
    )
    label_res = res_dict['label_batch_image'][0]
    truncated_res = res_dict['truncated_batch_judge'][0]
    white_res = res_dict['white_batch_judge'][0]
    res_list = []
    for image_info,tuncated_info,is_white in zip(label_res, truncated_res, white_res):
        if image_info.Extra is None:
            image_info.Extra = {}
        tuncated_info = tuncated_info or {}
        image_info.Extra["truncated_res"] = json.dumps(tuncated_info)
        image_info.Extra["is_white"] = json.dumps(is_white)
        res_list.append(image_info)

    return res_list


def input_image_filter(image_infos: List[ImageInfo]) -> List[ImageInfo]:
    """
    按打标结果对输入图片做过滤，得到最终的图片列表
    """
    filter_res = []
    for image_info in image_infos:
        if image_info.Extra is None:
            continue

        truncated_res = json.loads(image_info.Extra["truncated_res"])
        #is_white = json.loads(image_info.Extra["is_white"])

        #适合抠图的
        is_suitable_for_matting = json.loads(image_info.Extra["is_suitable_for_matting"])
        if not is_suitable_for_matting:
            continue

        # 上面被截断的
        top_truncated = truncated_res['truncated_detail_raw']['top']
        if top_truncated:
            continue

        # 下面被截断，且宽长比比较大的，过滤
        subject_wh_ratio = truncated_res.get("subject_wh_ratio", 1)
        if subject_wh_ratio>=0.7: # 1280*(1-0.3) / 720*(1-0.07*2)
            continue

        filter_res.append(image_info)

    return filter_res


def input_image_sort(image_infos: List[ImageInfo]) -> List[ImageInfo]:
    """
    对输入图片做排序，得到最终的图片列表
    """
    # 按标签排序，有人的放前面
    image_infos = sorted(image_infos, key=lambda x: json.loads(x.Extra["truncated_res"]).get("is_human", False), reverse=True)
    return image_infos


def input_image_dedup(image_infos: List[ImageInfo]) -> List[ImageInfo]:
    """
    对输入图片抠完主体后的部分做一道去重
    """
    subject_image_url_list = [json.loads(image_info.Extra["truncated_res"]).get("subject_image_url", "") for image_info in image_infos]
    if subject_image_url_list is None or len(subject_image_url_list) < 1:
        return image_infos
    _, indices = dedup_within_images_by_clip_g(subject_image_url_list, similarity_threshold=0.98)
    dedup_image_infos = [image_infos[i] for i in indices]
    return dedup_image_infos


def input_image_process_pipeline(image_infos: List[ImageInfo]) -> List[ImageInfo]:
    """
    输入图片处理pipeline
    """
    image_infos = input_image_tagging(image_infos)
    image_infos = input_image_filter(image_infos)
    image_infos = input_image_dedup(image_infos)
    image_infos = input_image_sort(image_infos)
    return image_infos


if __name__=='__main__':
    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"
    os.environ["TCE_PSM"] = "ad.creative.image_core_solution"

    # single
    url1 = 'https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20250711336e39bc88d6e6204d77bebf?t=1755791357459'
    url2 = 'https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20250711336edb711809dae64cd6a81d?t=1755791357460'
    image_infos = [ImageInfo(URL=url1), ImageInfo(URL=url2), ImageInfo(URL=url1)]
    res = input_image_process_pipeline(image_infos)
    print(res)
