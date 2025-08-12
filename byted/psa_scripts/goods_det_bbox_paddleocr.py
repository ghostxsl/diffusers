# @author: wilson.xu.
import os
os.environ["TCE_PSM"] = "ad.creative.image_core_solution"
import argparse
import json
from tqdm import tqdm
from PIL import Image, ImageDraw
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.utils import load_file, json_save, load_csv_or_xlsx_to_dict
from diffusers.data.outer_vos_tools import encode_pil_bytes, load_or_download_image
from diffusers.data.byted.tos import save_tos
from biz.infra.clients.rpc.creative_ai_capability import image_subject_seg
from overpass_ad_creative_qwen_image_v1.clients.rpc.ad_creative_qwen_image_v1 import AdCreativeQwen_Image_V1Client
from overpass_ad_creative_image_core_solution.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.capability_thrift import ImageInfo


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="catalogbg20260315.csv", type=str)
    parser.add_argument(
        "--output_file", default="bbox_catalogbg20260315.json", type=str)
    parser.add_argument(
        "--idc", default="sg1", type=str)
    parser.add_argument(
        "--max_workers", default=1, type=int)

    args = parser.parse_args()
    return args


def draw_bbox(image: Image.Image, ocr_result: dict) -> Image.Image:
    """
    在图片上绘制OCR检测框
    :param image: 输入的PIL图片
    :param ocr_result: OCR返回的结果字典
    :return: 绘制好框的PIL图片
    """
    # 复制图片，避免修改原图
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # 读取框数据
    rec_boxes = ocr_result.get("rec_boxes", [])  # 红色矩形框
    rec_polys = ocr_result.get("rec_polys", [])  # 蓝色旋转框

    # ==================== 绘制红色矩形框 rec_boxes ====================
    red_color = (255, 0, 0)  # 红色 RGB
    for box in rec_boxes:
        x1, y1, x2, y2 = box
        # 绘制矩形：左上角(x1,y1)，右下角(x2,y2)
        draw.rectangle([x1, y1, x2, y2], outline=red_color, width=2)

    # ==================== 绘制蓝色旋转框 rec_polys ====================
    blue_color = (0, 0, 255)  # 蓝色 RGB
    for poly in rec_polys:
        # poly 是 4个点：[[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        # 把点转成 tuple 格式，方便绘制多边形
        points = [(p[0], p[1]) for p in poly]
        # 绘制闭合多边形
        draw.polygon(points, outline=blue_color, width=2)

    return img


def get_subject_image(image: Image.Image) -> Image.Image:
    """
    实现主体抠图+最小外接矩形裁剪，最终输出RGB格式（白色背景）
    :param image: 输入原图（PIL Image格式）
    :return: 裁剪后的主体图（RGB格式，无透明通道）
    """
    # 1. 调用接口获取主体mask（你原有代码，无修改）
    mask_url = (
        image_subject_seg(
            image_urls=[],
            image_infos=[ImageInfo(Binary=encode_pil_bytes(image, to_string=False))],
            only_mask=1,
            refine_mask=2,
        )
        .success_image_infos[0]
        .URL
    )
    mask = load_or_download_image(mask_url)

    # 2. Mask二值化（阈值127）
    mask = mask.convert("L")
    mask = mask.point(lambda x: 255 if x > 127 else 0)

    # ===================== 核心抠图步骤（必须用RGBA） =====================
    # 转RGBA是为了添加透明通道，实现背景透明（中间步骤，无影响）
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    # 用mask做透明通道，完成抠图（背景变透明）
    image.putalpha(mask)

    # 3. 计算最小外接矩形
    bbox = mask.getbbox()
    if bbox is None:
        # 无主体时，直接返回原图转RGB
        return image.convert("RGB")

    # 4. 裁剪抠图结果（此时还是RGBA透明格式）
    cropped_subject = image.crop(bbox)

    # ===================== 关键：转为RGB格式（白色背景） =====================
    # 创建白色背景图（尺寸和裁剪后的图一致）
    white_bg = Image.new("RGB", cropped_subject.size, (255, 255, 255))
    # 将透明的抠图粘贴到白色背景上
    white_bg.paste(cropped_subject, mask=cropped_subject.split()[-1])

    # 最终返回：纯RGB格式（无透明通道，背景白色）
    return white_bg


def ocr_image_text_detection(image, text=""):
    code, msg, resp = client.AiModel(
        request_body=json.dumps(
            {
                "image_bytes": encode_pil_bytes(image),
                "check_text": text,
            }
        )
    )
    if code == 0:
        result = json.loads(resp.result_body)
        if result["StatusCode"] == 0:
            return result
        else:
            raise Exception(f"OCR detection error, code: {result['StatusCode']}, msg: {result['StatusMessage']}")
    else:
        raise Exception(f"Service[OCR] error, code: {code}, msg: {msg}")


def get_ocr_and_subject(image_url):
    image = load_or_download_image(image_url)
    subject_image = get_subject_image(image)
    ocr_result = ocr_image_text_detection(subject_image)["ocr_result"]
    return ocr_result, subject_image, image


def send_request(item):
    ori_ocr, ori_subject, ori_img = get_ocr_and_subject(item["url_1"])
    ori_draw = draw_bbox(ori_subject, ori_ocr)
    ori_draw_url = save_tos(encode_pil_bytes(ori_draw, False), headers={"Content-Type": "image/jpeg"})

    gen_ocr, gen_subject, gen_img = get_ocr_and_subject(item["url_2"])
    gen_draw = draw_bbox(gen_subject, gen_ocr)
    gen_draw_url = save_tos(encode_pil_bytes(gen_draw, False), headers={"Content-Type": "image/jpeg"})

    item["ori_draw_url"] = ori_draw_url
    item["gen_draw_url"] = gen_draw_url

    return item


def main(data, dst, max_workers):
    results = []
    error_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(send_request, item): item for item in data}
        with tqdm(total=len(data)) as pbar:
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    res_item = future.result()
                    pbar.update(1)  # Update progress bar

                    results.append(res_item)
                    if len(results) % 100 == 0:
                        json_save(results, dst)
                except Exception as e:
                    print(f"An error occurred for {e}")
                    error_results.append({"error_reason": str(e)})

    json_save(results, dst)
    print(len(error_results))


if __name__ == "__main__":
    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"
    os.environ["TCE_PSM"] = "ad.creative.image_core_solution"

    args = parse_args()
    psm = "ad.creative.psa"
    cluster = "ocr"
    print(f"psm: {psm}, cluster: {cluster}, idc: {args.idc}, max_workers: {args.max_workers}")
    client = AdCreativeQwen_Image_V1Client(psm=psm, cluster=cluster, idc=args.idc, transport="ttheader")

    data = load_csv_or_xlsx_to_dict(args.input_file)
    main(data, args.output_file, args.max_workers)

    print('Done!')
