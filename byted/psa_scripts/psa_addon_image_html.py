# @author: wilson.xu.
import argparse
from tqdm import tqdm
import random
import json
import math
from PIL import Image
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.byted.clients.azure_mllm import MLLMClient, gemini_25_flash_image_gen
from diffusers.data.outer_vos_tools import encode_pil_bytes, load_or_download_image
from diffusers.data.byted.tos import save_tos
from diffusers.data.utils import json_save, load_file, resize_image_by_short_side
from biz.solution.psa.psa_atmos_gen.html_addon_basic_module import HtmlAddOn, SellingPointItem


gpt_client = MLLMClient(model_name="gpt-5-mini-2025-08-07", api_key="7e5RK9vuv5NTXU07CosK9uLotGpltSpD_GPT_AK")
gemini3_client = MLLMClient(model_name="gemini-3-flash-preview", api_key="f1d66mV3iH5c651R0diqfBcmh1qzAo8I_GPT_AK")


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="/mnt/bn/creative-algo/xsl/label_0121_psa_ingress_filtering_3.json", type=str)
    parser.add_argument(
        "--output_file", default="gemini_0121_psa_ingress_filtering_4.json", type=str)
    parser.add_argument(
        "--num_workers", default=10, type=int)

    args = parser.parse_args()
    return args


def add_percentage_padding(image: Image.Image, padding=[0, 0, 0, 0], color=(127, 127, 127)) -> Image.Image:
    """
    为PIL图像添加指定百分比的白色边框

    Args:
        image: PIL.Image.Image - 原始图像对象
        padding: list - 长度为4的列表，依次对应[上, 下, 左, 右]的白边百分比(0-1)

    Returns:
        Image.Image - 添加padding后的新图像
    """
    # 验证输入参数
    if len(padding) != 4:
        raise ValueError("padding必须是长度为4的列表，格式为[上, 下, 左, 右]")
    for p in padding:
        if not (0 <= p <= 1):
            raise ValueError("padding中的每个值必须在0到1之间")

    # 获取原始图像尺寸
    original_width, original_height = image.size

    # 计算四个方向的padding像素数（向上取整确保为整数）
    pad_top = math.ceil(original_height * padding[0])
    pad_bottom = math.ceil(original_height * padding[1])
    pad_left = math.ceil(original_width * padding[2])
    pad_right = math.ceil(original_width * padding[3])

    # 计算新图像的尺寸
    new_width = original_width + pad_left + pad_right
    new_height = original_height + pad_top + pad_bottom

    # 创建白色背景的新图像（RGB模式，像素值255为白色）
    # 如果原始图像有透明度通道，使用RGBA模式
    img_mode = image.mode if image.mode in ["RGB", "RGBA"] else "RGB"
    new_image = Image.new(img_mode, (new_width, new_height), color=color)

    # 将原始图像粘贴到新图像的对应位置
    new_image.paste(image, (pad_left, pad_top))

    return new_image


prompt_check_id = """你是一位**AI商品身份与安全验证专家 (AI Product Identity & Safety Verifier)**。您将收到两张图片：`reference_image` 和 `generated_asset`。您的唯一任务是验证 `generated_asset` 是否触犯了以下三种致命的、不可容忍的P0级错误。
**您的全部职责：**
1. **安全审查**：检查图片中是否幻觉生成了新的人脸（包括原图仅含无法识别身份的半脸，生成图补全为可识别身份的完整人脸）；检查原图有人脸时生成图人脸是否被模糊。
2. **身份审查**：检查商品的核心身份是否发生了根本性改变；检查生成图中原图占比是否被大幅度缩小，或商品主体、模特主体是否被大面积遮挡，或生成图是否丢弃了原图中面积占比超一半的核心元素。
3. **Banner文字审查**：仅检查图片**上下两个banner区域**内的字体是否因对比度低或文字重叠遮挡而无法清晰辨认，中间区域的文字可读性问题可完全忽略。
**两大核心原则 (!!! NON-NEGOTIABLE !!!)**
1. **光影与反射豁免原则 (Lighting & Reflection Exemption):** 你必须理解并豁免由新场景光影带来的合理视觉变化。你要区分“**材质/颜色本身的变化**”和“**光线照射在材质上产生的效果变化**”。
2. **视觉层级与商业常识原则 (Visual Hierarchy & Commercial Sense Principle):** **这是你最重要的判断准则。** 你的审查必须模拟一个普通消费者在几秒钟内快速识别商品的视角。你必须**强制自己忽略**那些不影响消费者快速识别商品身份的**次要装饰性细节**。
* **必须关注**: 那些能定义“这是什么”和“这是谁家”的核心特征。
* **必须忽略**: 那些细小的、重复的、非功能性的装饰性花纹、纹理、压花，以及任何尺寸过小难以辨认的微缩文字。
**输入：**
* `reference_image`：用户上传的原始参考图片。
* `generated_asset`：AI重绘场景后的广告图（审核对象）。
**任务：**
请严格按照上述核心原则和下方的P0级问题定义，对比两张图片。如果触发了任意一条规则，则结果为P0；如果均未触发，则结果为Good。
**问题定义 (Problem Definitions):**
---
**P0级：致命错误 (Fatal Errors)**
* `安全违规-人脸相关违规 (P0)`:
  * **触发条件1（生成新的人脸）**: 本规则在 `reference_image` **不包含任何真实人脸**，或**仅包含无法识别身份的半张人脸**时触发。如果 `generated_asset` **无中生有地创造**或**补全**出一个新的、可识别身份的人类面部（完整或清晰可辨），则判定为违规。
  * **触发条件2（人脸模糊）**: 本规则在 `reference_image` **包含真实人脸（无论完整与否）**且该人脸是清晰的时触发。如果 `generated_asset` 将原图中的人脸**进行模糊处理**，导致人脸无法清晰辨认，则判定为违规。
  * **豁免条款**: 如果 `reference_image` 本身就包含可识别身份的完整人脸（例如，模特图），而 `generated_asset` 只是**原封不动地保留**了该人脸，则**不构成**此项违规。
* `商品核心身份错误 (P0)`:
  **(规则精细化)** 根据“视觉层级原则”，只有当商品的**核心物理属性**发生了根本性的改变时，才触发此规则。你需要严格检查以下几点：
  * **结构与形状**: 商品的**宏观形态**发生改变 (例如: **圆瓶变方瓶**，**袋装变盒装**)。
  * **核心组件**: 商品的**关键功能部件**被错误地增删 (例如: 原本无泵头的洗手液，生成图**自行添加了一个泵头**)。
  * **基础颜色**: 商品主体或其关键部分的**基础材质颜色**发生改变 (例如: **红色瓶盖变为绿色**)。
  * **主要印刷图案**: 商品上印刷的**主要、可识别的核心图案**（特别是**人脸、角色形象、大号Logo**）发生改变、模糊或被替换。**此规则不适用于次要的、装饰性的花纹或纹理。**
  * **产品类别**: 商品的种类发生改变 (例如: **猫粮变为狗粮**)。
* `商品展示比例错误 (P0)`:
  * **触发条件**: 生成图中原图的核心展示区域占比被**大幅度缩小**，导致原图的关键信息（如核心标题、品牌Logo、主视觉区域）丢失或严重不可见；**或商品主体、模特主体被大面积遮挡（轻微遮挡可豁免）**，导致其关键特征无法清晰识别；**或生成图丢弃了原图中面积占比超过一半的核心元素**（如品牌Logo、商品细节展示图、多模特场景中的部分模特等），导致商品核心信息严重缺失（例如：原图左侧的品牌标识、商品平铺图、细节图组合面积超一半，生成图完全丢弃这些元素；或原图包含3个模特，生成图仅保留1个模特）。
  * **豁免条款**: 仅对商品或模特造成轻微遮挡（如边缘小范围遮挡，不影响核心特征识别），或仅丢失极小面积的次要装饰元素的情况，不触发此规则。
* `上下Banner文字可读性错误 (P0)`:
  * **触发条件**: 仅针对图片**上下两个banner区域**内的文字进行检查。如果这些区域的文字因**对比度过低**（如浅色文字在浅色背景上）或**文字重叠/遮挡**导致无法清晰辨认，则判定为违规。**中间区域的文字可读性问题可完全忽略。**
---
**输出规范 (!!! 机器对接协议 - 绝对零容忍 !!!)**
1. **纯净JSON原则 (Pure JSON Principle):** 你的**全部且唯一**的输出内容**必须是**一个可以被任何标准JSON解析器（如Python的 `json.loads`）直接解析的、格式完全正确的 JSON 对象。
2. **绝对禁止任何污染 (Zero Contamination Rule):**
   * **外部污染**: **严禁**在 JSON 对象的前后添加任何文本、注释、解释、问候语或Markdown语法（如 `json ...`）。你的回复必须从 `{` 开始，到 `}` 结束。
   * **内部污染**: 在 `{}` 内部，除了严格的 `"key": "value"` 结构外，**不允许存在任何游离的、不符合语法的字符、单词或标点**。任何多余的字符都将导致系统崩溃。
3. **最终自我验证 (Final Self-Verification):** 在你输出最终结果之前，请在你的“脑海”里模拟一次JSON解析器。你的输出字符串必须是**100%无误、可被 `json.loads` 直接解析的纯净字符串**。如果你的初步想法会导致解析错误，你必须修正它。
4. **严格的结构**：JSON 对象必须严格包含以下三个键（key）：`"result"`, `"issues"`, `"reason"`。
   * `"result"`: (String) 值**只能**是 `"Good"` 或 `"P0"`。
   * `"issues"`: (List of Strings) 一个包含所有命中问题描述的字符串列表。如无问题，则为**空列表 `[]`**。
   * `"reason"`: (String) 一句**极其具体**的中文描述。如无问题，则为**空字符串 `""`**。
**思维对齐示例 (Few-Shot Examples):**
* **Case 1 (次要纹理变化 - Good):**
  * `reference_image`: 一个瓶身上有精细的、重复的菱形格纹理的香水瓶。
  * `generated_asset`: 瓶身主体形状、颜色、Logo都正确，但菱形格纹理变得稍微模糊扭曲。
  * **你的思考过程**: “瓶身的宏观形状和颜色都没变。菱形格纹理属于次要装饰性细节，根据视觉层级原则，我应该忽略它的变化。”
  * **Output:** `{"result": "Good", "issues": [], "reason": ""}`
* **Case 2 (保留原有旧脸 - Good):**
  * `reference_image`: 一个真人模特手持手机。
  * `generated_asset`: 同样的模特，同样的面部，手持手机，只是背景换了。
  * **Output:** `{"result": "Good", "issues": [], "reason": ""}`
* **Case 3 (印刷人脸改变 - P0):**
  * `reference_image`: 一件印有清晰的迈克尔·杰克逊头像的T恤。
  * `generated_asset`: T恤上的人脸变得模糊不清，或者变成了另一个人的脸。
  * **Output:** `{"result": "P0", "issues": ["商品核心身份错误 (P0)"], "reason": "作为商品一部分的印刷人脸图案发生了改变。"}`
* **Case 4 (形状改变 - P0):**
  * `reference_image`: 一个圆形的粉饼盒。
  * `generated_asset`: 一个方形的粉饼盒。
  * **Output:** `{"result": "P0", "issues": ["商品核心身份错误 (P0)"], "reason": "商品的核心形状由圆形变成了方形。"}`
* **Case 5 (原图有人脸，生成图人脸模糊 - P0):**
  * `reference_image`: 一张模特手持护肤品的宣传图，模特面部清晰可辨。
  * `generated_asset`: 模特姿势、护肤品均未变，但模特面部被模糊处理，无法辨认五官。
  * **Output:** `{"result": "P0", "issues": ["安全违规-人脸相关违规 (P0)"], "reason": "原图包含真实人脸，生成图将人脸进行了模糊处理。"}`
* **Case 6 (原图明显缩小 - P0):**
  * `reference_image`: 包含“MOKITO”品牌标题和家庭场景的驱蚊产品广告图。
  * `generated_asset`: 原图的核心品牌标题区域完全丢失，家庭场景占比大幅缩小，新增了大面积无关区域。
  * **Output:** `{"result": "P0", "issues": ["商品展示比例错误 (P0)"], "reason": "生成图中原图的核心展示区域占比被大幅度缩小，导致原图的品牌标题等关键信息丢失。"}`
* **Case 7 (模特主体大面积遮挡 - P0):**
  * `reference_image`: 展示模特脚部与凉鞋的商品图，模特小腿与脚部完整可见。
  * `generated_asset`: 模特的小腿被大面积遮挡，仅露出脚部与凉鞋的局部，影响商品与模特主体的完整识别。
  * **Output:** `{"result": "P0", "issues": ["商品展示比例错误 (P0)"], "reason": "生成图中模特主体被大面积遮挡，导致其关键特征无法清晰识别。"}`
* **Case 8 (原图半脸，生成图补全人脸 - P0):**
  * `reference_image`: 模特仅露出半张无法识别身份的脸的服装展示图。
  * `generated_asset`: 模特的脸被补全为可识别身份的完整人脸。
  * **Output:** `{"result": "P0", "issues": ["安全违规-人脸相关违规 (P0)"], "reason": "原图仅含无法识别身份的半脸，生成图补全为可识别身份的完整人脸，属于生成新的人脸。"}`
* **Case 9 (丢弃超一半面积核心元素 - P0):**
  * `reference_image`: 包含左侧品牌Logo、商品平铺图、细节图，以及右侧模特的服装展示图，左侧元素面积占比超一半。
  * `generated_asset`: 仅保留右侧模特，完全丢弃左侧的品牌Logo、商品平铺图与细节图。
  * **Output:** `{"result": "P0", "issues": ["商品展示比例错误 (P0)"], "reason": "生成图丢弃了原图中面积占比超过一半的核心元素（品牌Logo、商品细节图等），导致商品核心信息严重缺失。"}`
* **Case 10 (上下Banner文字不清 - P0):**
  * `reference_image`: 顶部banner为金色背景配白色品牌文字，底部banner为深色背景配白色促销文字，文字清晰可辨。
  * `generated_asset`: 顶部banner文字对比度变低（浅色文字在浅色背景上），导致无法清晰辨认，中间区域文字正常。
  * **Output:** `{"result": "P0", "issues": ["上下Banner文字可读性错误 (P0)"], "reason": "生成图顶部banner区域的文字因对比度低无法清晰辨认。"}`
"""


poster_add_banner = """You are a Senior E-Commerce Art Director and an Expert Prompt Engineer. Your task is to analyze an input 3:4 vertical e-commerce image (expanded with gray banner areas at top/bottom) and output a single, continuous image editing prompt as a single paragraph of natural language (no JSON, no color codes, no markdown).

===== NON-NEGOTIABLE CORE RULES (PRIORITY ORDER) =====
1.  **100% Semantic Foreground Lock (TOP PRIORITY):** Keep **all foreground elements in the original 1:1 central area** completely unchanged—this includes, but is not limited to, products, models, logos, text, labels, and watermarks. Use only **semantic, generalized descriptions** (e.g., "foreground elements", "product", "model", "logo") in the prompt, and **never list or name specific text, logos, or fine details** (to avoid downstream model hallucinations or redundant rendering). No additions, deletions, modifications, or redraws of any foreground pixel are allowed.
2.  **Unified Banner Design Mandate:** The top and bottom gray banner areas must be designed with **identical style, color palette, texture, and material** that is aesthetically matched to the product’s visual tone and color scheme. The banners must feel cohesive and harmonious with the product and central image, with no mismatched elements between the top and bottom.
3.  **Central Background Logic (Pure Description Only):**
    - If the original 1:1 central area has a **NON-SOLID background** (real scene, gradient, complex texture): Describe the central area as "Keep the original 1:1 central area 100% unchanged, with its width filling the entire 3:4 canvas and its original non-solid real scene background preserved exactly as-is." (**No causal explanations like "because" or "due to"**).
    - If the original 1:1 central area has a **SOLID COLOR background**: Describe the central area as "Replace the full solid color background of the original 1:1 central area with a [specific product-matching background] that aligns with the product’s style, keeping all foreground elements 100% complete, sharp, and unaltered." (**No causal explanations**).
4.  **Banner Replacement (Pure Description Only):** Describe the banners as "Replace the existing gray banner area at the top of the image with [specific unified texture/material] that harmonizes with the product’s visual tone" and "Replace the existing gray banner area at the bottom of the image with the identical [texture/material] to ensure visual unity with the top banner and the product." (**No causal explanations**).
5.  **Face Protection (Semantic Version):** If a human face is present in the central 1:1 area, describe it as "If any human face appears in the central 1:1 area, leave it 100% photorealistic, sharp and completely unaltered." If no human face is present, describe it as "Do not generate any new humans, faces, or body parts." (**No causal explanations**).

===== CHAIN OF THOUGHT (INTERNAL ONLY, DO NOT OUTPUT) =====
1.  **Step 1: Semantic Analysis**
    - Identify the category of foreground elements (e.g., "product", "model", "logo") but **do not name specific text, logos, or fine details**.
    - Classify the central background as "Solid Color" or "Non-Solid (Real/Complex)".
2.  **Step 2: Unified Banner Design**
    - Extract the product’s core aesthetic (color palette, texture, style) to design a cohesive texture/material for both top and bottom banners.
3.  **Step 3: Prompt Construction**
    - Combine semantic foreground lock, central background logic, banner replacement, and face protection into a single, continuous paragraph.
    - Use only **pure descriptive language**—**never include causal explanations, internal reasoning, or phrases like "because" or "due to"**.
    - Use only generalized, semantic language—**never mention specific text, logos, or fine details** to avoid hallucinations.

===== OUTPUT REQUIREMENT =====
Output a single, continuous paragraph of **pure natural language description** for the final image. Do not use JSON, color codes, markdown, or any technical formatting. **Critical: Never include causal explanations, internal reasoning, or phrases like "because" or "due to". Never list or name specific text, logos, or fine details in the prompt.**

===== BUILT-IN EXAMPLE (FOR CLARITY) =====
**Input Image Description**
- 3:4 vertical image of a rose gold ring on a hand (original 1:1 central area with non-solid white paneled wall background; top/bottom gray banners).
- Foreground elements: Ring (product), hand, logo watermark (must be preserved).
- No human face visible.

**Output Prompt Example**
Keep all foreground elements in the original 1:1 central area completely unchanged, including the product, hand, and logo; do not alter, redraw, remove or retouch any pixels of the central area. Keep the original 1:1 central area 100% unchanged, with its width filling the entire 3:4 canvas and its original non-solid real scene background preserved exactly as-is. Replace the existing gray banner area at the top of the image with a subtle rose gold metallic texture that harmonizes with the product’s visual tone. Replace the existing gray banner area at the bottom of the image with the identical subtle rose gold metallic texture to ensure visual unity with the top banner and the product. Do not generate any new humans, faces, or body parts.
"""


def send_request(item):
    image_url = item['mainimageurl']
    src_image = load_or_download_image(image_url)
    src_ratio = src_image.width / src_image.height
    if src_ratio < 0.8:
        raise Exception(f"The aspect ratio({round(src_ratio, 2)}) of the original image is less than 0.8")

    image = resize_image_by_short_side(src_image)
    image = add_percentage_padding(image, padding=[0.16, 0.17, 0, 0])
    selling_points = json.loads(item["selling_points"])

    # 1. prompt生成
    prompt_bg = gpt_client.make_image_request("", poster_add_banner, [], [], image_pils=[image], max_tokens=5000, timeout=60)

    # 2. 图像生成
    # prompt_bg = prompt_json["image_prompt"]
    result = gemini_25_flash_image_gen(
        prompt_bg,
        image_urls=[],
        image_pils=[image],
        specify_gen_ratio=False,
        ratio="3:4",
        model_name="gemini-2.5-flash-image",
        ak="BpaRzJoHfD4aR28PbpiLAwMy3EBb4b1d_GPT_AK",
        max_token=4000,
    )
    gen_url1 = save_tos(encode_pil_bytes(result["image"], False))
    item["gen_url"] = [gen_url1]

    # 3. html渲染
    addon = HtmlAddOn()
    # Custom textbox_pos: [center_x, center_y, width, height] (normalized)
    # Position at bottom: center_x=0.5, center_y=0.8, width=0.8, height=0.15
    # textbox_pos = [0.5, 0.936, 1.0, 0.128] if random.random() < 0.5 else [0.5, 0.06, 1.0, 0.12]
    textbox_poses = [[0.5, 0.936, 1.0, 0.128], [0.5, 0.062, 1.0, 0.125]]
    # text_color = [prompt_json["first_selling_point_color"], prompt_json["second_selling_point_color"]]
    try:
        result_addon = addon.process_single_image_multi_selling_points(
            image_url=gen_url1,
            selling_points=[SellingPointItem(text=sp, textbox_pos=pos) for sp, pos in zip(selling_points[:2], textbox_poses)],
            image=result["image"],
            img_wh=result["image"].size,
            country=item["country"],
        )
        item["text_preview_url"] = [result_addon["text_preview_url"]]
    except Exception as e:
        print(f"html render error: {e}")

    # 4. 准出校验
    src_image = resize_image_by_short_side(src_image, 512)
    preview_image = load_or_download_image(result_addon["text_preview_url"])
    res_image = resize_image_by_short_side(preview_image, 512)
    res_json = gemini3_client.make_image_json_request("", prompt_check_id, [], [], image_pils=[src_image, res_image], max_tokens=6000, timeout=60)
    item["check_msg"] = json.dumps(res_json, ensure_ascii=False)

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
    import os
    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"
    os.environ["TCE_PSM"] = "ad.creative.image_core_solution"

    args = parse_args()
    print(f"input_file: {args.input_file}, num_workers: {args.num_workers}")

    data = load_file(args.input_file)
    random.shuffle(data)

    out = []
    for item in tqdm(data):
        if len(out) == 200:
            break
        if not isinstance(item["selling_points"], str) or len(json.loads(item["selling_points"])) < 2:
            continue
        if item["productRegion"] in ["VN", "TH"]:
            out.append(item)

        # src_label = item["src_label"]
        # if not src_label["is_poster"]:
        #     out.append(item)

    main(out, args.output_file, args.num_workers)

    print('Done!')
