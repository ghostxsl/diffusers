from typing import List, Tuple
import json
import msgpack
import logging
import itertools
from PIL import Image, ImageDraw

from diffusers.data.byted.tos import save_tos
from diffusers.data.byted.middleware import client_logid, calc_runtime_middleware
from diffusers.data.byted.clients.creative_ai_capability import image_body_face_detect
from diffusers.data.byted.tos import url_to_image_simple, image_to_bytes
from diffusers.data.byted.errno import WithCodeError, ErrCodeAiCapabilityError

from euler.base_compat_middleware import gdpr_auth_middleware
from overpass_ad_creative_ai_laplace_gateway.clients.rpc.ad_creative_ai_laplace_gateway import AdCreativeAi_Laplace_GatewayClient


AdCreativeAi_Laplace_GatewayCli = AdCreativeAi_Laplace_GatewayClient(idc="my", cluster="default", transport="ttheader", timeout=3000)
AdCreativeAi_Laplace_GatewayCli.set_euler_client_middleware(gdpr_auth_middleware)
AdCreativeAi_Laplace_GatewayCli.set_euler_client_middleware(client_logid)
AdCreativeAi_Laplace_GatewayCli.set_euler_client_middleware(calc_runtime_middleware)


def draw_bounding_boxes(image: Image.Image, bboxes: List[List[int]]) -> Image.Image:
    """
    Draws bounding boxes on an image.

    Args:
        image: The input PIL Image.
        bboxes: A list of bounding boxes, each defined as [x, y, w, h].

    Returns:
        A new PIL Image with the bounding boxes drawn on it.
    """
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    for bbox in bboxes:
        x, y, w, h = bbox
        shape = [(x, y), (x + w, y + h)]
        draw.rectangle(shape, outline="red", width=2)
    return image_copy


def clip_g_image(image_bytes: bytes) -> dict:
    """
    调用clip_g_image服务，获取图片的描述和特征
    Parameters:
        image_bytes (bytes): 图片bytes
    Returns:
        dict: 描述和特征，
            description: 图片的描述
            feature: 图片的特征
    """

    code, msg, response = AdCreativeAi_Laplace_GatewayCli.generricMethod(
        model_name="clip_g_image",
        psm="ad.creative.clip_g_image",
        idc="my2",
        cluster="Bernard-Prod",
        timeout=10,
        input_bytes_lists={"images": [image_bytes]},
    )

    if code != 0:
        raise Exception(f"[clip_g_image] failed, code: {code}, msg: {msg}")

    description = json.loads(response.output_bytes_lists["descriptions"][0])
    feature = json.loads(response.output_bytes_lists["features"][0])

    logging.info(f"Caption by CLIP Success! Res = {description}")
    res = {"description": description, "feature": feature}

    return res


def vsa_quality_score_predict(image_bytes: List[bytes], pids: List[str]) -> List[dict]:
    """
    调用vsa_quality_score_predict服务，获取图片的质量分
    Parameters:
        image_bytes (bytes): 图片bytes
    Returns:
        dict: [{'ctr': 0.01874211883544922, 'quality_score': 0.035050418227910995, 'low_quality_score': 93.9079360961914}]
    """
    code, msg, response = AdCreativeAi_Laplace_GatewayCli.generricMethod(
        model_name="image_ctr_prediction",
        psm="ad.creative.image_ctr_prediction",
        idc="maliva",
        cluster="default",
        timeout=30,
        input_bytes_lists={
            "images": image_bytes,
            "pids": [bytes(pid, encoding="utf-8") for pid in pids],
            "business": [b"VSA"] * len(image_bytes),
        },
        # input_str_lists={"image_urls": image_urls, "product_ids": pids},
    )
    if code != 0:
        raise Exception(f"[vsa_quality_score_predict] failed, code: {code}, msg: {msg}")
    score_dict = response.output_float_lists
    logging.info(f"Quality Score by VSA Success! Res = {score_dict}")
    # ctr low_qs qs
    return [
        {
            "ctr": ctr,
            "quality_score": qs,
            "low_quality_score": low_qs,  # 是低质量的概率
        }
        for ctr, low_qs, qs in zip(score_dict["ctr"], score_dict["low_qs"], score_dict["qs"])
    ]


def smarttext_layout_predict(image_urls: List[str], slogans: List[str], ocr_bboxes: List[str] = None, face_threshold=0.30) -> List[dict]:
    """
    调用smarttext_layout_predict服务，获取图片的描述和特征
    Parameters:
        image_urls (list[str]): 图片urls, 只支持长度为1
        slogans (list[str]): 文字卖点, 只支持长度为1
    Returns:
        list[dict]: 文本框坐标、文本内容和分数
    """
    # 人脸检测
    detect_items = image_body_face_detect(image_urls, max_obj_num=10)
    face_bboxes = []
    for item in detect_items:
        if item.eng_name == "body":
            continue
        face_bboxes.append({"x1": item.box.x1, "y1": item.box.y1, "x2": item.box.x2, "y2": item.box.y2})

    if ocr_bboxes is None:
        ocr_bboxes = ["[]"] * len(image_urls)
    image = url_to_image_simple(image_urls[0])
    if image is None:
        logging.info("图片加载失败或无有效图片数据")
        return [None]
    width, height = image.size

    def slogan_split(slogan):
        slogan = slogan.replace("\n", " ").strip()
        words = slogan.split()
        results = [slogan]

        max_lines = min(len(words), 4)
        for n_lines in range(2, max_lines + 1):
            # 分割点的所有组合（n_lines-1个分割点）
            best_split = None
            best_maxlen = float("inf")
            for split_points in itertools.combinations(range(1, len(words)), n_lines - 1):
                lines = []
                prev = 0
                for sp in split_points + (len(words),):
                    line_words = words[prev:sp]
                    line = " ".join(line_words)
                    lines.append(line)
                    prev = sp
                maxlen = max(len(line) for line in lines)
                minlen = min(len(line) for line in lines)
                # 优先最大行最短，如果一样则最小行最长
                if (maxlen < best_maxlen) or (maxlen == best_maxlen and minlen > min(len(line) for line in best_split or [""])):
                    best_split = lines
                    best_maxlen = maxlen
            results.append("\n".join(best_split))
        return results

    def face_overlap(text_area, face_area):
        y1, y2 = face_area[1], face_area[3]
        face_area[1] = (y2 + 3 * y1) // 4
        face_area[3] = (4 * y2 + y1) // 5

        x_left = max(text_area[0], face_area[0])
        y_top = max(text_area[1], face_area[1])
        x_right = min(text_area[2], face_area[2])
        y_bottom = min(text_area[3], face_area[3])

        if x_right <= x_left or y_bottom <= y_top:  # 人脸关键区域和文本区域没有重叠
            return False
        return True

    if len(face_bboxes) > 0:
        splited_slogans = slogan_split(slogans[0])
    else:
        splited_slogans = [slogans[0]]

    for slogan in splited_slogans:
        request = {
            "urls": [url.encode("utf-8") for url in image_urls],
            "slogans": [slogan.encode("utf-8")],
            "ocr_bboxes": [ocr_bbox.encode("utf-8") for ocr_bbox in ocr_bboxes],
        }
        code, msg, response = AdCreativeAi_Laplace_GatewayCli.generricMethod(
            model_name="SmartText",
            psm="ad.creative.layout_smarttext_1p",
            idc="my",
            cluster="Bernard-Prod",
            timeout=30,
            input_bytes_lists=request,
        )
        if code != 0:
            raise Exception(f"[smarttext_layout_predict] failed, code: {code}, msg: {msg}")

        result = []
        result = eval(response.output_bytes_lists["result"][0].decode("utf-8"))
        if result is None:
            continue
        else:
            result.pop("idx")
            result.pop("tl_cnt")
            result["slogan"] = slogan
            flag = True
            for face_bbox in face_bboxes:
                face_area = [face_bbox["x1"] * width, face_bbox["y1"] * height, face_bbox["x2"] * width, face_bbox["y2"] * height]
                text_area = [result["xl"], result["yl"], result["xr"], result["yr"]]
                if face_overlap(text_area, face_area):
                    flag = False
                    break
            if flag:
                return [result]

    logging.info("[smarttext_layout_predict] Layout by SmartText Failed! Not enough room for slogan!")
    return [None]


# poster llama, mode class to pos, 返回bbox xywh，pixel wise
def poster_llama_layout_c_to_ps(image_url: str, text_list: List[str], category_list=None, is_preview=False) -> Tuple[List[List[float]], str]:
    if category_list is None:
        category_list = []
    if len(text_list) != len(category_list):
        logging.error("text_list and category_list must have the same length, set to text")
        category_list = ["Text"] * len(text_list)

        # raise WithCodeError(ErrCodeParamError, "text_list and category_list must have the same length")

    img = url_to_image_simple(image_url)
    imgBytes = image_to_bytes(img)
    width = img.width
    height = img.height
    elem_templ = """<rect data-category="<Category>", x="<M>", y="<M>", width="<M>", height="<M>"/> """
    inst = """I want to generate layout in poster design format. The layout should avoid the main subject of the picture as much as possible. Please generate the layout html according to the image I provide (in html format). 
            Text: <text_prompt> 
            ###bbox html: <body> <svg width="513" height="750"> <pos_prompt> </svg> </body> <MID>"""

    pos_prompt = ""
    text_prompt = ""
    for idx in range(len(text_list)):
        text_prompt += f"{text_list[idx]}" + (" & " if idx != len(text_list) - 1 else "")
        pos_prompt += elem_templ.replace("<Category>", category_list[idx])
    inst = inst.replace("<pos_prompt>", pos_prompt).replace("<text_prompt>", text_prompt).encode("utf-8")
    request = {
        # url 和 binary 二选一
        # "img_urls": [b"https://sf16-muse-va.ibytedtos.com/obj/ad-creative/1749556786.773304.png"],
        "binary_imgs": [imgBytes],
        "insts": [inst],
    }
    code, msg, response = AdCreativeAi_Laplace_GatewayCli.generricMethod(
        model_name="LayoutLlama",
        psm="ad.creative.layout_llama",
        idc="maliva",
        cluster="Bernard-Prod",
        timeout=60,
        input_bytes_lists=request,
    )

    if code != 0:
        raise Exception(f"failed, code: {code}, msg: {msg}")

    posterLLamaRes = eval(response.output_bytes_lists["results"][0].decode("utf-8"))
    print(posterLLamaRes)
    if posterLLamaRes is None:
        raise WithCodeError(ErrCodeAiCapabilityError, "posterLLamaRes is None")
    respBBox = []
    Texts = posterLLamaRes["Text"]
    Underlays = posterLLamaRes["Underlay"]
    Overlays = posterLLamaRes["Embellishment"]
    Logos = posterLLamaRes["Logo"]
    HighLights = posterLLamaRes["Highlighted text"]
    # todo [tzw text poster] 这里会有个matching问题... 先简单用长度matching了，也无视类型了
    box_gather = [
        [int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width - bbox[0] * width), int(bbox[3] * height - bbox[1] * height)]
        for bbox in (Texts + Underlays + Overlays + Logos + HighLights)
    ]
    lengths = [len(s) for s in text_list]
    sorted_indices = sorted(range(len(text_list)), key=lambda i: lengths[i])
    respBBox = [box_gather[i] for i in sorted_indices]
    logging.info(f"poster_llama_layout_c_to_ps resp: {respBBox}")
    preview_painted_url = ""
    if is_preview:
        painted_img = draw_bounding_boxes(img, respBBox)
        preview_painted_url = save_tos(image_to_bytes(painted_img), None, "poster_llama_layout_preview")
    return respBBox, preview_painted_url


# poster llama, mode class to pos, 返回bbox xywh，pixel wise
def poster_llama_layout_cs_to_p(
    image_url: str,
    text_list: List[str],
    category_list=None,
    size_list=None,
    is_preview=False,
    is_series=False,
) -> Tuple[List[List[float]], str]:
    if category_list is None:
        category_list = []
    if size_list is None:
        size_list = []
    if len(text_list) != len(category_list):
        logging.error("text_list and category_list must have the same length, set to text")
        category_list = ["Text"] * len(text_list)
    if len(text_list) != len(size_list):
        logging.error("text_list and category_list must have the same length, set to text")
        size_list = []
        for idx, text in enumerate(text_list):
            w = 0.025 * len(text) if idx != 0 else 0.035 * len(text)  # *1.2 title is bigger.
            h = (int(w) + 1) * 0.055 * 1.1 if idx != 0 else (int(w) + 1) * 0.055 * 1.2
            size_list.append([int(min(w, 0.95) * 513), int(min(h, 0.5) * 750)])  # w,h ratio
        # raise WithCodeError(ErrCodeParamError, "text_list and category_list must have the same length")

    img = url_to_image_simple(image_url)
    imgBytes = image_to_bytes(img)
    width = img.width
    height = img.height
    elem_templ = """<rect data-category="<Category>", x="<M>", y="<M>", width="<WIDTH_V>", height="<HEIGHT_V>"/> """
    inst_base = """I want to generate layout in poster design format. The layout should avoid the main subject of the picture as much as possible. Please generate the layout html according to the image I provide (in html format). 
            Text: <text_prompt> 
            ###bbox html: <body> <svg width="513" height="750"> <pos_prompt> </svg> </body> <MID>"""
    respBBox = []
    subj_pos = []
    # subj_pos = [[0, 550, 720, 750]]
    if is_series:
        pos_prompts = []
        text_prompt = ""

        def check_overlap(bbox_list, overlap_threshold=0.15):
            """检查bbox列表中是否有任何两个bbox重叠面积超过阈值"""
            for i in range(len(bbox_list)):
                for j in range(i + 1, len(bbox_list)):
                    box1 = bbox_list[i]
                    box2 = bbox_list[j]

                    # 计算相交区域
                    x_left = max(box1[0], box2[0])
                    y_top = max(box1[1], box2[1])
                    x_right = min(box1[0] + box1[2], box2[0] + box2[2])
                    y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])

                    if x_right <= x_left or y_bottom <= y_top:
                        continue  # 没有重叠

                    # 计算重叠面积
                    overlap_area = (x_right - x_left) * (y_bottom - y_top)
                    box1_area = box1[2] * box1[3]
                    box2_area = box2[2] * box2[3]

                    # 计算重叠比例（取两个box中较小的面积作为基准）
                    min_area = min(box1_area, box2_area)
                    overlap_ratio = overlap_area / min_area

                    if overlap_ratio > overlap_threshold:
                        return True
            return False

        for idx in range(len(text_list)):
            text_prompt += (" & " if idx != 0 else "") + f"{text_list[idx]}"
            pos_prompts.append(
                (
                    elem_templ.replace("<Category>", category_list[idx])
                    .replace("WIDTH_V", str(size_list[idx][0]))
                    .replace("HEIGHT_V", str(size_list[idx][1]))
                )
            )
            inst = inst_base.replace("<pos_prompt>", "".join(pos_prompts)).replace("<text_prompt>", text_prompt).encode("utf-8")
            request = {
                "binary_imgs": [imgBytes],
                "insts": [inst],
            }
            # 第一次尝试
            code, msg, response = AdCreativeAi_Laplace_GatewayCli.generricMethod(
                model_name="LayoutLlama",
                psm="ad.creative.layout_llama",
                idc="maliva",
                cluster="Bernard-Prod",
                timeout=60,
                input_bytes_lists=request,
            )
            if code != 0:
                raise Exception(f"failed, code: {code}, msg: {msg}")
            posterLLamaRes = eval(response.output_bytes_lists["results"][0].decode("utf-8"))
            if posterLLamaRes is None:
                raise WithCodeError(ErrCodeAiCapabilityError, "posterLLamaRes is None")
            # 获取生成的box
            new_boxes = [
                [int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width - bbox[0] * width), int(bbox[3] * height - bbox[1] * height)]
                for bbox in (
                    posterLLamaRes["Text"]
                    + posterLLamaRes["Underlay"]
                    + posterLLamaRes["Embellishment"]
                    + posterLLamaRes["Logo"]
                    + posterLLamaRes["Highlighted text"]
                )
            ]
            # 检查重叠
            if idx > 0 and check_overlap(new_boxes + subj_pos):
                # 第二次尝试
                code, msg, response = AdCreativeAi_Laplace_GatewayCli.generricMethod(
                    model_name="LayoutLlama",
                    psm="ad.creative.layout_llama",
                    idc="maliva",
                    cluster="Bernard-Prod",
                    timeout=60,
                    input_bytes_lists=request,
                )

                if code != 0:
                    raise Exception(f"failed, code: {code}, msg: {msg}")

                posterLLamaRes = eval(response.output_bytes_lists["results"][0].decode("utf-8"))
                if posterLLamaRes is None:
                    raise WithCodeError(ErrCodeAiCapabilityError, "posterLLamaRes is None")
                new_boxes = [
                    [int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width - bbox[0] * width), int(bbox[3] * height - bbox[1] * height)]
                    for bbox in (
                        posterLLamaRes["Text"]
                        + posterLLamaRes["Underlay"]
                        + posterLLamaRes["Embellishment"]
                        + posterLLamaRes["Logo"]
                        + posterLLamaRes["Highlighted text"]
                    )
                ]

                # 如果第二次还有重叠，则跳过这个文本
                if check_overlap(new_boxes + subj_pos):
                    logging.warning(f"Text '{text_list[idx]}' could not be placed without overlapping, break")
                    break

            # 添加成功的box
            respBBox = new_boxes
    else:
        pos_prompt = ""
        text_prompt = ""
        for idx in range(len(text_list)):
            text_prompt += f"{text_list[idx]}" + (" & " if idx != len(text_list) - 1 else "")
            pos_prompt += (
                elem_templ.replace("<Category>", category_list[idx])
                .replace("WIDTH_V", str(size_list[idx][0]))
                .replace("HEIGHT_V", str(size_list[idx][1]))
            )
        inst = inst_base.replace("<pos_prompt>", pos_prompt).replace("<text_prompt>", text_prompt).encode("utf-8")
        print(f"poster_llama_layout_cs_to_p inst: {inst}")
        request = {
            # url 和 binary 二选一
            # "img_urls": [b"https://sf16-muse-va.ibytedtos.com/obj/ad-creative/1749556786.773304.png"],
            "binary_imgs": [imgBytes],
            "insts": [inst],
        }
        code, msg, response = AdCreativeAi_Laplace_GatewayCli.generricMethod(
            model_name="LayoutLlama",
            psm="ad.creative.layout_llama",
            idc="maliva",
            cluster="Bernard-Prod",
            timeout=60,
            input_bytes_lists=request,
        )

        if code != 0:
            raise Exception(f"failed, code: {code}, msg: {msg}")

        posterLLamaRes = eval(response.output_bytes_lists["results"][0].decode("utf-8"))
        print(posterLLamaRes)
        if posterLLamaRes is None:
            raise WithCodeError(ErrCodeAiCapabilityError, "posterLLamaRes is None")
        Texts = posterLLamaRes["Text"]
        Underlays = posterLLamaRes["Underlay"]
        Overlays = posterLLamaRes["Embellishment"]
        Logos = posterLLamaRes["Logo"]
        HighLights = posterLLamaRes["Highlighted text"]
        # todo [tzw text poster] 这里会有个matching问题... 先简单用长度matching了，也无视类型了
        box_gather = [
            [int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width - bbox[0] * width), int(bbox[3] * height - bbox[1] * height)]
            for bbox in (Texts + Underlays + Overlays + Logos + HighLights)
        ]
        lengths = [len(s) for s in text_list]
        sorted_indices = sorted(range(len(text_list)), key=lambda i: lengths[i])
        box_gather.sort(key=lambda x: (x[2], x[3]))
        respBBox = [None] * len(text_list)
        for box_idx, i in enumerate(sorted_indices):
            respBBox[i] = box_gather[box_idx]
    # respBBox = [box_gather[i] for i in sorted_indices]
    logging.info(f"poster_llama_layout_c_to_ps resp: {respBBox}")
    preview_painted_url = ""
    if is_preview:
        painted_img = draw_bounding_boxes(img, respBBox)
        preview_painted_url = save_tos(image_to_bytes(painted_img), None, "poster_llama_layout_preview")
    return respBBox, preview_painted_url


def carousel_reward_llm(
    image_bytes_list: List[bytes],
    title,
    deliver_country_code,
    deliver_region_name,
    l1_vertical_tag="",
    l2_vertical_tag="",
    l3_vertical_tag="",
    l4_vertical_tag="",
    is_smb="",
    creative_id="",
    model_name="carousal_reward_llm",
) -> dict:
    """
    # https://bytedance.us.larkoffice.com/docx/Jc60dRuTQoV4kuxliqsutn4ksRc
    模型参数跟 @Haotian Zhang 对齐；
    实验入参跟 @万发东 对齐。
    """

    code, msg, response = AdCreativeAi_Laplace_GatewayCli.generricMethod(
        model_name=model_name,
        psm="tiktok.brand_ads.carousel_reward_llm",
        idc="my2",
        cluster="default",
        timeout=30,
        input_bytes_lists={
            "images": [msgpack.packb(image_bytes_list)],
            "creative_id": [creative_id.encode()],
            "deliver_country_code": [deliver_country_code.encode()],
            "deliver_region_name": [deliver_region_name.encode()],
            "l1_vertical_tag": [l1_vertical_tag.encode()],
            "l2_vertical_tag": [l2_vertical_tag.encode()],
            "l3_vertical_tag": [l3_vertical_tag.encode()],
            "l4_vertical_tag": [l4_vertical_tag.encode()],
            "is_smb": [is_smb.encode()],
            "title": [title.encode()],
        },
    )

    if code != 0:
        raise Exception(f"[carousel_reward_llm] failed, code: {code}, msg: {msg}")

    res_dict = {"ctr_pred": response.output_float_lists["ctr_pred"][0]}
    logging.info(f"[carousel_reward_llm] Success. Res = {res_dict}")

    return res_dict


if __name__ == "__main__":
    import os

    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"
    os.environ["TCE_PSM"] = "ad.creative.image_core_solution"

    # url = "https://img.fantaskycdn.com/157752aef65bb35e1d545c21ddf2c30a.jpeg"
    # from biz.common.network.downloader import url_to_bytes

    # image_bytes = url_to_bytes(url)
    # res = clip_g_image(image_bytes)
    # print(res)

    # print(smarttext_layout_predict(["https://p16-oec-va.ibyteimg.com/tos-maliva-i-o3syd03w52-us/a2420e3d8d81420e86863ad8fc5e40f2~tplv-o3syd03w52-resize-jpeg:1000:1000.image?"], ["Best Selling Product"]))
    # exit(0)

    # from biz.common.image.image_process import draw_polygons, concatenate_images
    #
    # urls = [
    #     "https://p16-oec-va.ibyteimg.com/tos-maliva-i-o3syd03w52-us/425346dcaaa44ba283c5fcbc414c81d9~tplv-o3syd03w52-resize-jpeg:1000:1000.image?",
    #     "https://p16-oec-va.ibyteimg.com/tos-alisg-i-aphluv4xwc-sg/ba77ae63d6bd4a4f92222491994e9ec0~tplv-o3syd03w52-resize-jpeg:1000:1000.image?",
    #     "https://p16-oec-va.ibyteimg.com/tos-alisg-i-aphluv4xwc-sg/7353f3ed15d0432aa4ce893db14e659b~tplv-o3syd03w52-resize-jpeg:1000:1000.image?",
    #     "https://p16-oec-va.ibyteimg.com/tos-alisg-i-aphluv4xwc-sg/bb83715f16574a3fbc8f46ef9fea55c1~tplv-o3syd03w52-resize-jpeg:1000:1000.image?",
    #     "https://p16-oec-va.ibyteimg.com/tos-alisg-i-aphluv4xwc-sg/c913993bc2c44b5184057983e222413c~tplv-o3syd03w52-resize-jpeg:1000:1000.image?",
    # ]
    # pre_results = [
    #     "https://p16-lp-sg.ibyteimg.com/tos-alisg-i-375lmtcpo0-sg/316fa22628324662ab71ab6babed3a93~tplv-375lmtcpo0-image.png",
    #     "https://p16-lp-sg.ibyteimg.com/tos-alisg-i-375lmtcpo0-sg/1bcec1d5adba4790ba7b2bf951abd1ad~tplv-375lmtcpo0-image.png",
    #     "https://p16-lp-sg.ibyteimg.com/tos-alisg-i-375lmtcpo0-sg/b3c4d4a865334b5b94bfa065f4cadf9e~tplv-375lmtcpo0-image.png",
    #     "https://p16-lp-sg.ibyteimg.com/tos-alisg-i-375lmtcpo0-sg/1685047063294f8982e59a3aa9a35fcd~tplv-375lmtcpo0-image.png",
    #     "https://p16-lp-sg.ibyteimg.com/tos-alisg-i-375lmtcpo0-sg/368234cc6f1647c2a6dad85b3c9f3015~tplv-375lmtcpo0-image.png",
    # ]
    # slogans = ["Tak Panas", "Kain kapas bersulam", "Butang marmar unik", "Waistband Elastik", "Gaya Elegan Harian"]
    # images = []
    # for url, slogan, pre_result in zip(urls, slogans, pre_results):
    #     result = smarttext_layout_predict([url], [slogan])[0]
    #     image0 = url_to_image_simple(url)
    #     image1 = url_to_image_simple(pre_result)
    #     x1, y1, x2, y2 = int(result["xl"]), int(result["yl"]), int(result["xr"]), int(result["yr"])
    #     image2 = draw_polygons(image0, [[[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]], [result["slogan"]])
    #     images.append(image0)
    #     images.append(image1)
    #     images.append(image2)
    # print(concatenate_images(images))

    # # poster_llama
    # resp, preview_url = poster_llama_layout_cs_to_p(
    #     "https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20250707336ee84738bf20bc49229940",
    #     [
    #         "Test Title Test Title",
    #         "Test Title Test Title Test Title Test Title",
    #         "Test sellingpoint Test sellingpoint Test sellingpoint",
    #         "Test sellingpoint",
    #         "Test description Test description Test description Test description Test description Test description Test description",
    #         "Shopping now",
    #     ],
    #     is_preview=True,
    # )
    # print(resp)
    # print(preview_url)

    from biz.common.network.downloader import url_to_bytes

    url = "https://img.fantaskycdn.com/157752aef65bb35e1d545c21ddf2c30a.jpeg"
    image_bytes = url_to_bytes(url)
    title = "Si est\u00e1s interesado"
    deliver_country_code = "CO"
    deliver_region_name = "LATAM USA"

    res = carousel_reward_llm([image_bytes], title=title, deliver_country_code=deliver_country_code, deliver_region_name=deliver_region_name)
    print(res)
