import math
import json
import logging
import traceback
from PIL import Image, ImageDraw

from diffusers.data.byted.tos import save_tos
from diffusers.data.byted.clients.creative_ai_capability import image_subject_seg
from diffusers.data.byted.clients.ad_creative_ocr_quality_control import ocr_image_text_detection
from diffusers.data.byted.clients.gemini_mllm import mllm_make_image_request
from diffusers.data.outer_vos_tools import load_or_download_image, encode_pil_bytes
from diffusers.data.utils import resize_image_by_short_side
from diffusers.data.byted.clients.seed2_api import Seed2Client

from overpass_ad_creative_image_core_solution.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.capability_thrift import ImageInfo


check_text_and_logo_prompt = """## AI 文本与 Logo 质检员 - 核心指令
---
### ⚠️ 【输入定义】CRITICAL INPUT DEFINITION ⚠️

You will always receive **two** images for comparison. The mapping is fixed and absolute:
*   **Image 1 = 『原图』 (Original Image):** This is the ground truth, the reference image.
*   **Image 2 = 『生成图』 (Generated Image):** This is the image you need to inspect.

Your entire analysis is based on comparing **Image 2 against Image 1**. You must never reverse this logic.
---
### ⚠️ CRITICAL DIRECTIVE: YOUR SOLE RESPONSIBILITY ⚠️

Your **only job** is to act as a **Text & Logo Quality Inspector**. You are **NOT** a product structure inspector. You must **forcefully ignore** any changes unrelated to text and logos. Your entire analysis must be **strictly confined** to the rules below.
---

### 一、 核心原则：三步分层检测法 (Three-Tiered Inspection)

你的工作流程被严格定义为以下三个连续的步骤。你必须按顺序执行，不得跳跃。

#### **第一步：对象对应性检查 (Object Correspondence Check) - 最高优先级**

在进行任何细节对比前，你必须先在宏观上匹配『原图』和『生成图』中的物体，这是所有判断的**前提**。

*   **匹配成功：** 99%的情况下，两张图展示的是同一商品，视角也基本一致。这时，你可以进入第二步。
*   **匹配失败/部分失败 (豁免情况):** 在极少数情况下：
    1.  **视角变化导致信息丢失：** 如原图是鞋子侧面（有Logo），生成图是鞋子正面（无Logo）。
    2.  **组件/物体丢失：** 如原图展示了主体和配件，但生成图只生成了主体。
    *   **处理方式：** 这种因视角变化或组件丢失导致的“文字/Logo不可见”，**不属于P0错误**。你应将此视为AI的合理创作范畴，**直接豁免该部分**，不应报告为“Logo缺失”。你只需继续对两张图中都存在的、匹配成功的物体执行后续步骤。

#### **第二步：Logo通用扫描 (Universal Logo Scan) - 绝对优先**

在确认物体可比后，你必须优先扫描并对比图中**所有**的品牌Logo。

*   **定义：** Logo是指任何代表品牌的图形标志、徽章、符号或程式化的字母组合。
*   **绝对规则：**
    1.  **所有Logo一视同仁：** 无论Logo大小、位置，都享有同等的最高检测优先级。
    2.  **【视觉匹配原则】杜绝知识误用：** 你的任务是**纯粹的视觉对比**。如果原图Logo就是一个单独的 'R'，生成图也是一个单独的 'R'，那么它们就是**匹配的(Good)**。你**严禁**基于你对品牌的了解而“推断”出有不存在的字母缺失。

#### **第三步：场景判断与文本规则应用**

在完成Logo扫描后，你将对**除Logo外的所有文字**进行场景判断：

---
*   **【场景A：信息稀疏型商品】-> 应用“全面检测”范围**
    *   **特征：** 文字组很少，尺寸差异不大。常见于**手表、珠宝、鞋履、汽车/摩托车、大部分时尚品（如香水瓶）**。
    *   **规则：** **所有可识别的文字**均视为**“主要信息 (In-Scope)”**并纳入检测。

*   **【场景B：信息密集型商品】-> 应用【品牌与标题聚焦】范围**
    *   **特征：** 大量文字，尺寸差异巨大。常见于**食品包装、饮料、保健品**。
    *   **规则：** **主要信息 (In-Scope)** 被严格限制为**主标题/商品名**和**品牌名称文字**。

---

### 二、 错误边界的最终消歧

*   **1. 乱码/错误 (Error - P0):** 信息被**篡改**为可读但错误或无意义的字符/图形。
*   **2. 无法识别 (Unidentifiable - Good):** 信息完全**丢失**，变成无法分辨轮廓的马赛克或污点。
*   **3. 轻微形变 (Distortion - P1):** 信息内容正确，但渲染质量有轻微抖动、模糊或扭曲。

---

### 三、 统一的评判标准 (适用于所有场景)

#### **P0 (严重错误 - 不通过)**
“主要信息”中出现以下任何一种情况：
1.  **Logo错误 (Logo Error - 零容忍):** 任何一个品牌Logo的内容发生错误、乱码、严重失真、或被错误图形替换。(例: `JJ&S` Logo内部字母扭曲)
2.  **内容错误 (Content Error - 零容忍):** 发生在品牌名称或主标题上的任何可识别错误（乱码、拼写错误、篡改）。
3.  **【幻觉生成】:** 原图中模糊不清的“主要信息”，被凭空创造为清晰但错误的内容。

#### **P1 (轻微瑕疵 - 可通过)**
不属于P0，但存在以下情况：
1.  **轻微形变/伪影:** 主要信息的内容正确，但渲染有轻微抖动、扭曲。
2.  **清晰度下降 (仍可辨认):** 主要信息的内容正确，但比原图模糊一些，不过仍能读出正确内容。

#### **Good (通过)**
满足以下任一条件：
1.  **【真实无法识别豁免】** 原图清晰的“主要信息”，在生成图中退化为**完全无法辨认出任何轮廓的**马赛克。
2.  所有“主要信息”与原图完全一致。
3.  所有差异均发生在被忽略的“次要信息”上。
4.  所有差异均由“对象对应性检查”失败（如视角变化）导致。

---

### 四、 分析与输出流程

#### **Step 1: 智能分析与范围定义**
1.  首先执行**【第一步：对象对应性检查】**。
2.  其次执行**【第二步：Logo通用扫描】**，列出所有待检Logo。
3.  最后执行**【第三步：场景判断】**，并根据场景规则，列出其他待检的文字信息。

#### **Step 2: 逐项对比分析 (仅限主要信息)**

| 主要信息 (In-Scope Element) | 原图状态 (Original State) | 生成图状态 (Generated State) | 差异描述 (Discrepancy) |
| :--- | :--- | :--- | :--- |
| **`R` Logo (香水机器人)** | 清晰的单个字母'R' Logo | 清晰的单个字母'R' Logo | 一致 |
| **`JJ&S` Logo (酒瓶)** | 清晰，包含"JJ&S"字母 | 内部字母被篡改为乱码 | Logo内容被篡改 |

#### **Step 3: 综合评估与判定**
*   **香水机器人案例：**
    *   **分析：** 根据【视觉匹配原则】，原图和生成图都是一个完整的'R' Logo，不存在缺失。判定为`Good`。
*   **酒瓶案例：**
    *   **分析：** 'JJ&S' Logo 属于“主要信息”。其内容被篡改为乱码，符合“Logo错误”的`P0`标准。
    *   **判定：** `P0`

#### **Step 4: 最终输出 (JSON格式)**

**香水机器人案例输出:**
```json
{
  "result": "Good",
  "issues": [],
  "reason": ""
}
"""


def midpoint(p1, p2):
    """计算两点的中点坐标"""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def point_to_line_distance(point, line_p1, line_p2):
    """
    计算点到直线的垂直距离（直线由line_p1、line_p2两点确定）
    公式：|(y2-y1)x0 - (x2-x1)y0 + x2y1 - y2x1| / √[(y2-y1)² + (x2-x1)²]
    """
    x0, y0 = point
    x1, y1 = line_p1
    x2, y2 = line_p2

    # 分母：直线长度，避免除零错误
    denominator = math.hypot(x2 - x1, y2 - y1)
    if denominator == 0:
        return 0

    # 分子：点到直线的距离公式
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    return numerator / denominator


def calculate_short_edge(poly):
    """
    【修正版】计算4点四边形的文本行高（中点到对边的最小垂直距离）
    :param poly: 4点坐标 [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
    :return: 四边形的最小垂直高度（文本行高）
    """
    if len(poly) != 4:
        return 0

    # 四边形四个顶点
    p0, p1, p2, p3 = poly
    # 定义：四条边 + 对应的对边
    edge_pairs = [
        (p0, p1, p2, p3),  # 边0(p0-p1)，对边(p2-p3)
        (p1, p2, p3, p0),  # 边1(p1-p2)，对边(p3-p0)
        (p2, p3, p0, p1),  # 边2(p2-p3)，对边(p0-p1)
        (p3, p0, p1, p2),  # 边3(p3-p0)，对边(p1-p2)
    ]

    distances = []
    for edge_p1, edge_p2, opp_p1, opp_p2 in edge_pairs:
        # 计算当前边的中点
        mp = midpoint(edge_p1, edge_p2)
        # 计算中点到对边的垂直距离
        dist = point_to_line_distance(mp, opp_p1, opp_p2)
        distances.append(dist)

    # 返回最小垂直距离 = 文本行高
    return min(distances)


class SmallTextLogoChecker(object):
    def __init__(self, tcc_client=None):
        self.reasoning_levels = ["minimal", "low", "medium", "high"]
        self.seed_client = Seed2Client(reasoning_effort="minimal")

    @staticmethod
    def get_origin_url_and_image(image_info):
        try:
            if isinstance(image_info.Extra.get("origin_image", None), Image.Image):
                origin_image = image_info.Extra["origin_image"]
            else:
                origin_image = load_or_download_image(image_info.URL)
                origin_image = resize_image_by_short_side(origin_image)
                image_info.Extra["origin_image"] = origin_image
            return image_info.URL, origin_image
        except Exception as e:
            error_msg = f"Failed to obtain the original image: {str(e)}, traceback: {traceback.format_exc()}"
            logging.warning(error_msg)
            raise Exception(error_msg)

    @staticmethod
    def get_generate_url_and_image(image_info):
        try:
            generate_url = image_info.Extra["generate_url"]
            if isinstance(image_info.Extra.get("generate_image", None), Image.Image):
                generate_image = image_info.Extra["generate_image"]
            else:
                generate_image = load_or_download_image(generate_url)
                generate_image = resize_image_by_short_side(generate_image)
                image_info.Extra["generate_image"] = generate_image
            return generate_url, generate_image
        except Exception as e:
            error_msg = f"Failed to obtain the generate image: {str(e)}, traceback: {traceback.format_exc()}"
            logging.warning(error_msg)
            raise Exception(error_msg)

    def get_subject_image(self, image: Image.Image) -> Image.Image:
        """
        实现主体抠图+最小外接矩形裁剪，最终输出RGB格式（白色背景）
        :param image: 输入原图（PIL Image格式）
        :return: 裁剪后的主体图（RGB格式，无透明通道）
        """
        # 1. 调用接口获取主体mask（你原有代码，无修改）
        mask_url = (
            image_subject_seg(
                image_urls=[],
                image_infos=[ImageInfo(Binary=encode_pil_bytes(image, False))],
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

    def draw_rec_polys(self, image: Image.Image, ocr_result: dict, short_edge_threshold: int = 18) -> Image.Image:
        """
        绘制过滤后的OCR旋转文本框（红色）
        :param image: 输入的PIL图片
        :param ocr_result: OCR返回的结果字典
        :param short_edge_threshold: 行高过滤阈值，默认18
        :return: 绘制完成的PIL图片
        """
        img = image.copy()
        draw = ImageDraw.Draw(img)
        red_color = (255, 0, 0)
        line_width = 1

        rec_polys = ocr_result.get("rec_polys", [])
        for poly in rec_polys:
            # 计算文本行高（垂直距离）
            line_height = calculate_short_edge(poly)
            # 过滤：行高小于阈值则不绘制
            if line_height < short_edge_threshold:
                continue

            # 绘制红色旋转四边形框
            points = [(p[0], p[1]) for p in poly]
            draw.polygon(points, outline=red_color, width=line_width)

        return img

    def draw_ocr_bbox(self, image, min_size=18):
        image = image.copy()
        ocr_result = ocr_image_text_detection(image)["ocr_result"]
        ori_draw = self.draw_rec_polys(image, ocr_result, min_size)
        return ori_draw

    def gemini_check(self, image_pils, logid=""):
        result_json = mllm_make_image_request(
            check_text_and_logo_prompt,
            image_urls=[],
            image_pils=image_pils,
            max_tokens=5000,
            timeout=60,
            temperature=0.1,
            is_json_response=True,
            logid=logid,
            model_name="gemini-3.1-fl",
            api_key="XpGAiljn5ZhzjpRn8Eo8BRvj1uWcaCmP_GPT_AK",
            base_url="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl",
            # model_name="gemini-2.5-flash",
            # api_key="lV9PRXdcOPUV8AgbdrTUtf8E9B0r68Qc_GPT_AK",
            # base_url="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/multimodal/crawl",
        )
        return result_json

    def seed_check(self, image_pils, logid=""):
        result_json = self.seed_client.call_json(
            check_text_and_logo_prompt,
            image_pils=image_pils,
            max_tokens=5000,
        )
        return result_json

    def subject_text_and_logo_detection(self, image_info, origin_image, generate_image, **kwargs):
        logid = kwargs.get("logid", None) or image_info.Extra.get("logid", "")
        # 1. 主体抠图
        origin_subject = self.get_subject_image(origin_image)
        generate_subject = self.get_subject_image(generate_image)
        # # 2. OCR检测框绘制
        # origin_subject_ocr = self.draw_ocr_bbox(origin_subject)
        # generate_subject_ocr = self.draw_ocr_bbox(generate_subject)
        # 3. AI质检
        result_json = self.gemini_check([origin_subject, generate_subject], logid)
        # result_json = self.seed_check([origin_subject, generate_subject], logid)

        image_info.Extra["origin_subject_url"] = save_tos(encode_pil_bytes(origin_subject, False), headers={"Content-Type": "image/jpeg"})
        image_info.Extra["generate_subject_url"] = save_tos(encode_pil_bytes(generate_subject, False), headers={"Content-Type": "image/jpeg"})
        image_info.Extra["check_info_subject"] = json.dumps(result_json, ensure_ascii=False)

        return image_info

    def __call__(self, image_info, **kwargs):
        origin_url, origin_image = self.get_origin_url_and_image(image_info)
        generate_url, generate_image = self.get_generate_url_and_image(image_info)

        try:
            # 1. 商品主体文字/logo校验
            image_info = self.subject_text_and_logo_detection(image_info, origin_image, generate_image, **kwargs)

            return image_info
        except Exception as e:
            raise Exception(f"[{self.__class__.__name__}] error, url: {origin_url}, msg: {str(e)}")


if __name__ == "__main__":
    import os
    from os.path import exists, join
    import pandas
    from tqdm import tqdm
    import concurrent
    from concurrent.futures import ThreadPoolExecutor

    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"
    os.environ["TCE_PSM"] = "ad.creative.image_core_solution"

    def csv_save(obj, file, mode="w"):
        df = pandas.DataFrame(obj)
        header = True
        if mode == "a":
            header = not exists(file)
        df.to_csv(file, mode=mode, index=False, header=header, encoding="utf-8")

    def load_csv_or_xlsx_to_dict(file_path):
        if file_path.endswith(".xlsx"):
            df = pandas.read_excel(file_path)
        elif file_path.endswith(".csv"):
            df = pandas.read_csv(file_path, encoding="utf-8")
        else:
            raise Exception(f"Error `file_path` type:{file_path}")
        return df.to_dict("records")

    pipe = SmallTextLogoChecker()
    max_workers = 1
    file_name = "catalogbg20260315.csv"
    out_name = "catalogbg20260315_gemini31.csv"

    def send_request(item):
        ori_url = item["url_1"]
        gen_url = item["url_2"]
        resp = pipe(ImageInfo(URL=ori_url, Extra={"generate_url": gen_url}))

        item["origin_subject_url"] = resp.Extra["origin_subject_url"]
        item["generate_subject_url"] = resp.Extra["generate_subject_url"]
        item["check_info_subject"] = resp.Extra["check_info_subject"]
        return item

    cur_data = load_csv_or_xlsx_to_dict(file_name)

    results = []
    error_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(send_request, item): item for item in cur_data}
        with tqdm(total=len(cur_data)) as pbar:
            for future in concurrent.futures.as_completed(future_to_url):
                item = future_to_url[future]
                try:
                    res_item = future.result()
                    pbar.update(1)  # Update progress bar

                    results.append(res_item)
                    if len(results) % 50 == 0:
                        csv_save(results, out_name)
                except Exception as e:
                    print(f"An error occurred for {e}")
                    error_results.append({"item": item, "error_reason": str(e)})

    csv_save(results, out_name)
    print(f"file: {file_name}, error num: {len(error_results)}")

    print("Done!")
