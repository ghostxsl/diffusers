# @author: wilson.xu.
import argparse
import concurrent
import json
from typing import List
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from diffusers.data.byted.clients.azure_mllm import MLLMClient
from diffusers.data.outer_vos_tools import load_or_download_image, encode_pil_bytes
from diffusers.data.byted.tos import save_tos
from diffusers.data.utils import json_save, load_file


gpt_client = MLLMClient(model_name="gpt-5-mini-2025-08-07", api_key="7e5RK9vuv5NTXU07CosK9uLotGpltSpD_GPT_AK")
gemini_client = MLLMClient(model_name="gemini-2.5-flash", api_key="3H7OHTGn3JwHvlZHP9JUJzb850gp3TGR_GPT_AK")


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="gemini_0121_psa_ingress_filtering_3.json", type=str)
    parser.add_argument(
        "--output_file", default="bbox_gemini_0121_psa_ingress_filtering_3.json", type=str)
    parser.add_argument(
        "--num_workers", default=1, type=int)

    args = parser.parse_args()
    return args


def draw_normalized_bboxes(
    image: Image.Image,
    bboxes: List[List[float]],
    color: tuple = (255, 0, 0),  # 红色框
    width: int = 3,  # 边框粗细
    scale_factor: float = 1.2,  # 缩放系数，1.0为原大小
) -> Image.Image:
    """
    在 PIL 图像上绘制多个归一化坐标的 bounding box，并返回一张新图像。

    此函数支持对每个 bbox 进行居中缩放，并自动将缩放后的框裁剪到图像边界内。

    Args:
        image: 输入的 PIL 图像。
        bboxes: 包含多个归一化坐标的列表, e.g., [[box1], [box2], ...]。
        color: 边框颜色，默认红色 (255,0,0)。
        width: 边框粗细。
        scale_factor: Bbox 的宽高缩放系数。大于1.0放大，小于1.0缩小。默认1.0。

    Returns:
        一个画好了所有框的新的 PIL.Image.Image 对象。
    """
    # 复制原图，避免修改原始图像
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # 获取图像真实宽高（只需获取一次）
    img_w, img_h = img.size

    # 如果缩放系数为1，则无需进行复杂的计算
    if scale_factor == 1.0:
        for bbox in bboxes:
            if not bbox or len(bbox) != 4:
                continue
            x_min_norm, y_min_norm, x_max_norm, y_max_norm = bbox
            x_min = int(x_min_norm * img_w)
            y_min = int(y_min_norm * img_h)
            x_max = int(x_max_norm * img_w)
            y_max = int(y_max_norm * img_h)
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)
        return img

    # --- 处理缩放和裁剪的核心逻辑 ---
    for bbox in bboxes:
        if not bbox or len(bbox) != 4:
            continue

        # 1. 获取原始归一化坐标
        x_min_norm, y_min_norm, x_max_norm, y_max_norm = bbox

        # 2. 计算原始的中心点和宽高 (在归一化空间中)
        width_norm = x_max_norm - x_min_norm
        height_norm = y_max_norm - y_min_norm
        center_x_norm = x_min_norm + width_norm / 2
        center_y_norm = y_min_norm + height_norm / 2

        # 3. 计算缩放后的新宽高
        new_width_norm = width_norm * scale_factor
        new_height_norm = height_norm * scale_factor

        # 4. 根据新宽高和中心点，计算新的四角坐标
        scaled_x_min = center_x_norm - new_width_norm / 2
        scaled_y_min = center_y_norm - new_height_norm / 2
        scaled_x_max = center_x_norm + new_width_norm / 2
        scaled_y_max = center_y_norm + new_height_norm / 2

        # 5. 【关键】裁剪(clip)坐标，确保其在 [0.0, 1.0] 范围内
        final_x_min_norm = max(0.0, scaled_x_min)
        final_y_min_norm = max(0.0, scaled_y_min)
        final_x_max_norm = min(1.0, scaled_x_max)
        final_y_max_norm = min(1.0, scaled_y_max)

        # 6. 将最终的归一化坐标转换为像素坐标
        x_min_px = int(final_x_min_norm * img_w)
        y_min_px = int(final_y_min_norm * img_h)
        x_max_px = int(final_x_max_norm * img_w)
        y_max_px = int(final_y_max_norm * img_h)

        # 7. 绘制矩形框
        draw.rectangle([x_min_px, y_min_px, x_max_px, y_max_px], outline=color, width=width)

    return img


prompt_det_text_bbox = """# 角色与核心任务
你是一个顶级的视觉语义定位专家。你的唯一任务是：根据用户提供的【目标文本列表】，在图片中为每一个目标文本找到其对应的视觉区域，并输出该区域的边界框（bbox）。

你的核心价值在于，即使图片中的文本渲染不完美（如乱码、变形、错字），你也能通过语义、位置、字体样式等线索，推断出最可能的对应区域。

# 核心规则
1.  **模糊定位（最高优先级）**: 必须为【目标文本列表】中的 **每一个** 文本找到最可能的 bbox。不要因为文字渲染错误而放弃定位。
2.  **定长顺序输出（铁律）**: 输出的 JSON 数组长度和顺序，必须与输入的【目标文本列表】完全一致。

# 输入说明
1.  待检测图像：用户上传的 AI 生成海报图。
2.  目标文本列表（有序）：{selling_points}

# 坐标规范
-   **坐标系**: 图像归一化坐标系，左上角为 (0,0)，右下角为 (1,1)。
-   **bbox 格式**: `[x_min, y_min, x_max, y_max]`。

# 输出要求 (铁律)
1.  **仅输出 JSON**: 禁止任何解释性文字，确保输出可以直接被程序解析。
2.  **输出格式**: 一个 JSON 数组，每个元素对应一个目标文本，格式如下：
    ```json
    {{
      "target_text": "原样复制输入列表中的对应文本",
      "status": "定位结果枚举：success (成功定位) / inferred (推断定位) / failed (无法定位)",
      "bbox": "若 status 为 success 或 inferred, 输出 [x_min, y_min, x_max, y_max]; 若为 failed, 输出 []"
    }}
"""


def send_request(item):
    image_url = item['gen_url'][0]
    img = load_or_download_image(image_url)
    selling_points = json.loads(item["selling_points"])
    meta_prompt = prompt_det_text_bbox.format(selling_points=f"[\"{selling_points[0]}\", \"{selling_points[1]}\"]")

    res_json1 = gemini_client.make_image_json_request("", meta_prompt, [], [image_url], max_tokens=4000, timeout=60)
    gen_img1 = draw_normalized_bboxes(img.copy(), [a['bbox'] for a in res_json1 if a['status'] != 'failed'])
    draw_url1 = save_tos(encode_pil_bytes(gen_img1, False), headers={"Content-Type": "image/jpeg"})

    res_json2 = gpt_client.make_image_json_request("", meta_prompt, [], [image_url], max_tokens=4000, timeout=60)
    gen_img2 = draw_normalized_bboxes(img.copy(), [a["bbox"] for a in res_json2 if a["status"] != "failed"])
    draw_url2 = save_tos(encode_pil_bytes(gen_img2, False), headers={"Content-Type": "image/jpeg"})

    item['draw_url'] = [draw_url1, draw_url2]

    return item


def main(data, dst, max_workers):
    results = []
    error_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(send_request, item): item for item in data}
        with tqdm(total=len(data)) as pbar:
            for future in concurrent.futures.as_completed(future_to_url):
                item = future_to_url[future]
                try:
                    res_item = future.result()
                    pbar.update(1)  # Update progress bar
                    results.append(res_item)
                    if len(results) % 10 == 0:
                        json_save(results, dst)

                except Exception as e:
                    print(f"Error: {str(e)}")
                    error_results.append({'image_item': item, 'error_reason': str(e)})

    json_save(results, dst)
    print(f"error num: {len(error_results)}")


if __name__ == "__main__":
    args = parse_args()
    print(f"input_file: {args.input_file}, num_workers: {args.num_workers}")

    data = load_file(args.input_file)
    main(data, args.output_file, args.num_workers)

    print('Done!')
