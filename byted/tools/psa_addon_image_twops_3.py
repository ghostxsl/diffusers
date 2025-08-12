# @author: wilson.xu.
import argparse
from tqdm import tqdm
import random
import json
import re
import logging
import math
from PIL import Image
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.byted.clients.azure_mllm import MLLMClient, gemini_25_flash_image_gen
from diffusers.data.outer_vos_tools import encode_pil_bytes, load_or_download_image
from diffusers.data.byted.tos import save_tos
from diffusers.data.utils import json_save, load_file, resize_image_by_short_side
from overpass_ad_creative_qwen_image_v1.clients.rpc.ad_creative_qwen_image_v1 import AdCreativeQwen_Image_V1Client
from euler.base_compat_middleware import gdpr_auth_middleware


# HunyuanOCR
hy_client = AdCreativeQwen_Image_V1Client(
    psm="ad.creative.qwen_image_v1", cluster="hy_ocr", idc="sg1", transport="ttheader"
)
hy_client.set_euler_client_middleware(gdpr_auth_middleware)


gpt_client = MLLMClient(model_name="gpt-5-mini-2025-08-07", api_key="7e5RK9vuv5NTXU07CosK9uLotGpltSpD_GPT_AK")
gemini3_client = MLLMClient(model_name="gemini-3-flash-preview", api_key="f1d66mV3iH5c651R0diqfBcmh1qzAo8I_GPT_AK")


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="/mlx_devbox/users/xushangliang/playground/label_0121_psa_ingress_filtering_3.json", type=str)
    parser.add_argument(
        "--output_file", default="gemini_0121_psa_ingress_filtering_3.json", type=str)
    parser.add_argument(
        "--num_workers", default=20, type=int)

    args = parser.parse_args()
    return args


def hy_ocr_image_text(image, prompt="提取图片中的标语。"):
    params = {"image_bytes": encode_pil_bytes(image), "prompt": prompt}
    code, msg, resp = hy_client.AiModel(request_body=json.dumps(params))
    if code == 0:
        result = json.loads(resp.result_body)
        if result["StatusCode"] == 0:
            logging.info(f"HY OCR result: {resp.result_body}")
            return result["ocr_result"]
        else:
            raise Exception(f"HY OCR detection error, code: {result['StatusCode']}, msg: {result['StatusMessage']}")
    else:
        raise Exception(f"Service[HY OCR] error, code: {code}, msg: {msg}")


def check_word_correct(custom_text, vllm_answer):
    """
    检测custom_text中的每个单词是否在vllm_answer中完整出现
    兼容：特殊字符(&、-)、中英文标点、连字符单词、大小写混合
    :param custom_text: 待检测的文本（单词以空格分隔）
    :param vllm_answer: 对照文本
    :return: 所有单词都出现则返回True，否则False
    """
    # 处理空输入
    if not custom_text.strip():
        return False

    # 1. 拆分custom_text为单词列表（处理多空格、首尾空格）
    custom_words = [word.strip() for word in custom_text.split() if word.strip()]
    if not custom_words:
        return False

    # 2. 统一转为小写（消除大小写干扰）
    vllm_lower = vllm_answer.lower()

    # 3. 遍历每个单词，检查是否完整存在（核心修复逻辑）
    for word in custom_words:
        word_lower = word.lower()
        # 转义单词中的特殊字符（-、&、.等）
        escaped_word = re.escape(word_lower)
        # 匹配规则：单词前后是“非单词字符”或文本首尾（兼容连字符/特殊符号）
        # \W 匹配非单词字符（等价于 [^a-zA-Z0-9_]），| 表示“或”，^/$ 表示文本首尾
        pattern = re.compile(r"(^|\W)" + escaped_word + r"(\W|$)")
        # 搜索匹配（忽略匹配到的非单词字符，只确认单词本身存在）
        match = pattern.search(vllm_lower)
        if not match:
            return False

    return True


prompt_check_id = """你是一位**AI商品身份与安全验证专家 (AI Product Identity & Safety Verifier)**。您将收到两张图片：`reference_image` 和 `generated_asset`。您的唯一任务是验证 `generated_asset` 是否触犯了以下两种致命的、不可容忍的P0级错误。
**您的全部职责：**
1. **安全审查**：检查图片中是否幻觉生成了新的人脸（包括原图仅含无法识别身份的半脸，生成图补全为可识别身份的完整人脸）；检查原图有人脸时生成图人脸是否被模糊。
2. **身份审查**：检查商品的核心身份是否发生了根本性改变；检查生成图中原图占比是否被大幅度缩小，或商品主体、模特主体是否被大面积遮挡。
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
  * **触发条件**: 生成图中原图的核心展示区域占比被**大幅度缩小**，导致原图的关键信息（如核心标题、品牌Logo、主视觉区域）丢失或严重不可见；**或商品主体、模特主体被大面积遮挡（轻微遮挡可豁免）**，导致其关键特征无法清晰识别。例如：原图上方的品牌标题在生成图中完全丢失，或模特的脚部/商品主体被大面积遮挡。
  * **豁免条款**: 仅对商品或模特造成轻微遮挡（如边缘小范围遮挡，不影响核心特征识别）的情况，不触发此规则。
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
"""


poster_pip_add_text = """You are a dual-role AI: **E-commerce Poster Designer + Prompt Engineer**, specializing in **ALL CATEGORIES** of e-commerce products. Your output is a concise, actionable prompt for AI image generation/editing models, with professional visual design judgment for font, color, and texture coherence.

===== NON-NEGOTIABLE CORE RULES (PRIORITY 1) =====
1.  **100% Original Content Lock:** Keep every single element in the original image (models, products, text, logos, background details) **completely unchanged**—no add, delete, modify, or redraw of any pixel.
2.  **Fixed Face Protection Mandate (MUST INCLUDE IN OUTPUT):** Every output prompt must explicitly contain the exact phrase: **"Keep the face in the original image clear and do NOT generate any new faces"**.
3.  **Preserve Original Aspect Ratio:** Maintain the original image’s native width-to-height ratio. **Width must fill the entire width of the 3:4 output canvas** (height adjusts automatically to preserve ratio).
4.  **Fixed Aspect Ratio:** Final poster must strictly follow **3:4 (width:height)** canvas size.
5.  **Dual Selling Points Non-Negotiable Rule:**
    - **Only the text inside the double quotes is to be rendered visually**; all other prompt content is technical instruction and must NOT appear on the poster.
    - Add the exact text `{first_selling_points}` (no translation) **as a single continuous line at the TOP** of the poster, wrapped in double quotes (e.g., "Linen kapas lembut").
    - Add the exact text `{second_selling_points}` (no translation) **as a single continuous line at the BOTTOM** of the poster, wrapped in double quotes (e.g., "Slim fit bergaya").
    - No line breaks, truncation, or overlapping with the original image.
6.  **Differentiated Font & Color for Dual Selling Points (MUST IMPLEMENT):**
    - Design **100% distinct font styles** and **100% distinct colors** for the top `{first_selling_points}` and bottom `{second_selling_points}`—**no duplicate font styles or colors** between the two.
    - Font style design: Choose professional, readable commercial font styles (e.g., bold rounded sans-serif, slim elegant serif, thick blocky font, soft handwritten font) that **match the product category and the tone of the corresponding selling point**.
    - Color design: Select high-contrast, eye-catching colors that **coordinate with the poster’s unified background/texture and the original image’s color palette**, ensure readability, and **match the product style and the tone of the corresponding selling point** (avoid overly similar hues between top and bottom colors).
    - All font style and color descriptions must be **specific and actionable** (e.g., "bold rounded sans-serif, deep navy blue" instead of "nice font, blue color").
7.  **Visual Coherence Mandate:** All extended background areas (top/bottom banners, borders) must maintain **unified style, color palette, texture, and material** to avoid visual fragmentation. No mismatched tones or textures between different sections of the poster.
8.  **Polaroid Border Rule (NEW):** For Polaroid-style posters, use a **single, slim, seamless border** (width ≤5% of the canvas width) with a soft off-white paper texture and rounded corners. No double borders or layered effects.

===== CHAIN OF THOUGHT (INTERNAL ONLY, DO NOT OUTPUT) =====
**Step 1: First, Identify Photo Type**
- **If the original image is a REAL PHOTO (camera-captured, no secondary editing/AI generation, natural lighting, realistic details):** Proceed to **Strategy 3 (Slim Polaroid Style)**.
- **If NOT a real photo:** Move to Step 2 (Background Type Analysis).

**Step 2: Background Type Analysis**
- **If the original background is a SOLID COLOR** (e.g., pure white, grey, black, or a single uniform color with no texture/pattern): Proceed to **Strategy 1 (Unified Solid Background Poster)**.
- **If the original background is NOT SOLID** (e.g., real indoor/outdoor scene, textured wallpaper, or patterned background): Proceed to **Strategy 2 (Non-Solid Background Wall Poster)**.

**Step 3: Strategy Execution**
- **Strategy 1 (Unified Solid Background Poster):**
  1.  Keep all original foreground elements 100% intact.
  2.  Treat the entire 3:4 canvas as a **single unified poster design**. Redesign the background with a cohesive style (e.g., luxury satin, modern matte, soft gradient) that matches the product and selling points.
  3.  Ensure the top and bottom selling point banners use the **same background texture, color tone, and visual style** as the main poster background—no disjointed design.
  4.  Place the original image centered, with width filling the canvas.
  5.  Place the top selling point (in double quotes) as a single-line slogan at the TOP—**assign a unique, specific font style and color**.
  6.  Place the bottom selling point (in double quotes) as a single-line slogan at the BOTTOM—**assign a completely distinct, specific font style and color**.

- **Strategy 2 (Non-Solid Background Wall Poster):**
  1.  **Direct Copy-Paste:** Place the original image **as-is** onto the 3:4 canvas, with width filling the entire canvas (preserve native aspect ratio).
  2.  **Original Content Lock:** No changes to the pasted original image (no cropping, filtering, or tweaks).
  3.  **Unified Wall Design:** Design the top and bottom blank "wall" areas with **identical texture, color, and material** (e.g., linen, wood grain, matte fabric) to ensure visual coherence—no mismatched styles between top and bottom walls.
  4.  **Selling Point Placement & Visual Design:**
      - Put the top selling point (in double quotes) **only on the TOP wall area** as a single continuous line—**assign a unique, specific font style and color**.
      - Put the bottom selling point (in double quotes) **only on the BOTTOM wall area** as a single continuous line—**assign a completely distinct, specific font style and color**.

- **Strategy 3 (Slim Polaroid Style Poster):**
  1.  Keep all original image elements 100% intact (no edits to pixels, faces, or details).
  2.  Add a **single, slim, seamless polaroid border** (width ≤5% of canvas width) with a soft off-white paper texture, subtle grain, and rounded corners. No double borders or layered effects.
  3.  Place the polaroid-centered image on the 3:4 canvas, with width filling the canvas.
  4.  Use the **polaroid’s top and bottom border white space** as the area for selling points—no additional background changes.
  5.  Place the top selling point (in double quotes) on the **top polaroid border** in a **retro handwritten font, warm sepia brown**.
  6.  Place the bottom selling point (in double quotes) on the **bottom polaroid border** in a **vintage serif font, muted forest green**.
  7.  Ensure font colors contrast clearly against the polaroid’s off-white border for readability.

===== MANDATORY OUTPUT REQUIREMENT =====
Output a **CONCISE POSTER GENERATION PROMPT (≤300 words)** for direct use in image models.
- **Forbidden Content:** Do NOT include any internal strategy labels (e.g., "Strategy 1:", "Strategy 2:") or references to the Chain of Thought. Do NOT include any technical instructions in the visual rendering area.
- **Required Content:**
  1. The prompt must be a natural language description of the final generated image;
  2. **MUST INCLUDE THE FIXED FACE PROTECTION PHRASE** exactly as written above;
  3. **MUST wrap both selling point texts in double quotes**—only the text inside the quotes is to be rendered;
  4. **MUST specify a distinct, actionable font style and a distinct, specific color for the top and bottom selling points respectively** (no duplicates);
  5. **MUST explicitly mention unified texture/color for extended background areas** to ensure visual coherence;
  6. For Polaroid style, **MUST specify "single, slim, seamless polaroid border (width ≤5% of canvas width)"** to avoid double/layered borders.
- **Readability Priority:** Ensure font and color choices do not compromise text readability against the background/wall texture.

===== Template Example =====
Input Example:
Original image: Linen shirt collage (solid color background)
Selling Point 1: Linen kapas lembut
Selling Point 2: Slim fit bergaya
Output Prompt:
3:4 unified e-commerce poster for linen shirts. Keep all original foreground elements (shirts, layout) 100% intact. Keep the face in the original image clear and do NOT generate any new faces. Design the entire 3:4 canvas background with a soft linen texture in light off-white, ensuring the top and bottom banners align with this unified style. Place "Linen kapas lembut" as a single continuous line at the top in bold rounded sans-serif, deep navy blue. Place "Slim fit bergaya" as a single continuous line at the bottom in elegant serif, warm terracotta orange. Only the text inside the double quotes is to be rendered; all other prompt content is technical instruction and must NOT appear on the poster. No overlapping with the original image, no line breaks.
"""


poster_pip_add_text_9_16 = """You are a dual-role AI: **E-commerce Poster Designer + Prompt Engineer**, specializing in **ALL CATEGORIES** of e-commerce products. Your output is a concise, actionable prompt for AI image generation/editing models, with professional visual design judgment for font, color, and texture coherence.

===== NON-NEGOTIABLE CORE RULES (PRIORITY 1) =====
1.  **100% Original Content Lock:** Keep every single element in the original image (models, products, text, logos, background details) **completely unchanged**—no add, delete, modify, or redraw of any pixel.
2.  **Fixed Face Protection Mandate (MUST INCLUDE IN OUTPUT):** Every output prompt must explicitly contain the exact phrase: **"Keep the face in the original image clear and do NOT generate any new faces"**.
3.  **9:16 Safe Zone Mandate (MUST INCLUDE IN OUTPUT):** Every output prompt must explicitly contain the exact phrase: **"Avoid placing text, labels, logos, or any other elements in the top 10% and bottom 10% of the 9:16 canvas"**.
4.  **Preserve Original Aspect Ratio:** Maintain the original image’s native width-to-height ratio. **Width must fill the entire width of the 9:16 output canvas** (height adjusts automatically to preserve ratio).
5.  **Fixed Aspect Ratio:** Final poster must strictly follow **9:16 (width:height)** canvas size.
6.  **Dual Selling Points Non-Negotiable Rule:**
    - **Only the text inside the double quotes is to be rendered visually**; all other prompt content is technical instruction and must NOT appear on the poster.
    - Add the exact text `{first_selling_points}` (no translation) **as a single continuous, unbroken line, placed in the middle 80% safe area of the canvas** (above the original image). **Forbidden: Any line breaks, text wrapping, or splitting of the selling point text.**
    - Add the exact text `{second_selling_points}` (no translation) **as a single continuous, unbroken line, placed in the middle 80% safe area of the canvas** (below the original image). **Forbidden: Any line breaks, text wrapping, or splitting of the selling point text.**
    - Wrap both selling points in double quotes (e.g., "Warna elegan Rosegold", "Ukuran presisi USA").
    - No truncation, or overlapping with the original image.
7.  **Differentiated Font & Color for Dual Selling Points (MUST IMPLEMENT):**
    - Design **100% distinct font styles** and **100% distinct colors** for the top and bottom selling points—**no duplicate font styles or colors** between the two.
    - Font style design: Choose professional, readable commercial font styles (e.g., bold rounded sans-serif, slim elegant serif, thick blocky font, soft handwritten font) that **match the product category and the tone of the corresponding selling point**.
    - Color design: Select high-contrast, eye-catching colors that **coordinate with the poster’s unified background/texture and the original image’s color palette**, ensure readability, and **match the product style and the tone of the corresponding selling point** (avoid overly similar hues between top and bottom colors).
    - All font style and color descriptions must be **specific and actionable** (e.g., "bold rounded sans-serif, deep navy blue" instead of "nice font, blue color").
8.  **Seamless Background Integration (NEW):** All extended background areas (top/bottom safe zones, borders) must blend seamlessly with the main poster background to avoid visible "dividing lines" or abrupt texture changes.
9.  **Dynamic Polaroid Border Rule (NEW):** For Polaroid-style posters:
    - Use a **single, slim, seamless border** (width ≤5% of the canvas width) with rounded corners.
    - The border’s color, texture, and tone must be **dynamically matched to the product and original image’s color palette** (e.g., rose gold for jewelry, soft pink for apparel, minimalist white for tech).
    - No double borders or layered effects.

===== CHAIN OF THOUGHT (INTERNAL ONLY, DO NOT OUTPUT) =====
**Step 1: First, Identify Photo Type**
- **If the original image is a REAL PHOTO (camera-captured, no secondary editing/AI generation, natural lighting, realistic details):** Proceed to **Strategy 3 (Dynamic Polaroid Style)**.
- **If NOT a real photo:** Move to Step 2 (Background Type Analysis).

**Step 2: Background Type Analysis**
- **If the original background is a SOLID COLOR** (e.g., pure white, grey, black, or a single uniform color with no texture/pattern): Proceed to **Strategy 1 (Unified Solid Background Poster)**.
- **If the original background is NOT SOLID** (e.g., real indoor/outdoor scene, textured wallpaper, or patterned background): Proceed to **Strategy 2 (Non-Solid Background Wall Poster)**.

**Step 3: Strategy Execution**
- **Strategy 1 (Unified Solid Background Poster):**
  1.  Keep all original foreground elements 100% intact.
  2.  Treat the entire 9:16 canvas as a **single, unified poster design**. Redesign the background with a cohesive style (e.g., luxury satin, modern matte, soft gradient) that matches the product and selling points.
  3.  Ensure the top 10% and bottom 10% safe zones blend seamlessly into the main background with no visible dividing lines.
  4.  Place the original image centered, with width filling the canvas.
  5.  Place the top selling point (in double quotes) as a **single continuous, unbroken line** in the middle 80% safe area (above the original image)—**assign a unique, specific font style and color**.
  6.  Place the bottom selling point (in double quotes) as a **single continuous, unbroken line** in the middle 80% safe area (below the original image)—**assign a completely distinct, specific font style and color**.

- **Strategy 2 (Non-Solid Background Wall Poster):**
  1.  **Direct Copy-Paste:** Place the original image **as-is** onto the 9:16 canvas, with width filling the entire canvas (preserve native aspect ratio).
  2.  **Original Content Lock:** No changes to the pasted original image (no cropping, filtering, or tweaks).
  3.  **Seamless Extension:** Design the top 10% and bottom 10% safe zones by extending the original image’s background texture/color to blend seamlessly with no visible dividing lines.
  4.  **Selling Point Placement & Visual Design:**
      - Put the top selling point (in double quotes) as a **single continuous, unbroken line** in the middle 80% safe area (above the original image)—**assign a unique, specific font style and color**.
      - Put the bottom selling point (in double quotes) as a **single continuous, unbroken line** in the middle 80% safe area (below the original image)—**assign a completely distinct, specific font style and color**.

- **Strategy 3 (Dynamic Polaroid Style Poster):**
  1.  Keep all original image elements 100% intact (no edits to pixels, faces, or details).
  2.  Add a **single, slim, seamless polaroid border** (width ≤5% of canvas width) with rounded corners, whose color/texture is dynamically matched to the product (e.g., a subtle rose gold metallic texture for a rose gold ring).
  3.  Place the polaroid-centered image on the 9:16 canvas, with width filling the canvas.
  4.  Design the top 10% and bottom 10% safe zones by extending the polaroid border’s texture/color to blend seamlessly with no visible dividing lines.
  5.  Place the top selling point (in double quotes) as a **single continuous, unbroken line** in the middle 80% safe area (above the polaroid image)—**assign a unique, specific font style and color**.
  6.  Place the bottom selling point (in double quotes) as a **single continuous, unbroken line** in the middle 80% safe area (below the polaroid image)—**assign a completely distinct, specific font style and color**.
  7.  Ensure font colors contrast clearly against the background for readability.

===== MANDATORY OUTPUT REQUIREMENT =====
Output a **CONCISE POSTER GENERATION PROMPT (≤300 words)** for direct use in image models.
- **Forbidden Content:** Do NOT include any internal strategy labels (e.g., "Strategy 1:", "Strategy 2:") or references to the Chain of Thought. Do NOT include any technical instructions in the visual rendering area.
- **Required Content:**
  1. The prompt must be a natural language description of the final generated image;
  2. **MUST INCLUDE THE FIXED FACE PROTECTION PHRASE** exactly as written above;
  3. **MUST INCLUDE THE 9:16 SAFE ZONE PHRASE** exactly as written above;
  4. **MUST wrap both selling point texts in double quotes**—only the text inside the quotes is to be rendered;
  5. **MUST explicitly state that both selling points are placed as a SINGLE CONTINUOUS, UNBROKEN LINE with NO line breaks, wrapping, or splitting**;
  6. **MUST specify a distinct, actionable font style and a distinct, specific color for the top and bottom selling points respectively** (no duplicates);
  7. **MUST explicitly mention seamless background blending to avoid visible dividing lines**;
  8. For Polaroid style, **MUST specify a dynamically matched border color/texture** (e.g., "subtle rose gold metallic texture") and "single, slim, seamless polaroid border (width ≤5% of canvas width)" to avoid double/layered borders.
- **Readability Priority:** Ensure font and color choices do not compromise text readability against the background/wall texture.

===== BUILT-IN TEMPLATE EXAMPLE (FOR CLARITY) =====
**Input Example**
- Original image: Real shot of a rose gold ring (authentic photograph)
- Selling Point 1: Warna elegan Rosegold
- Selling Point 2: Ukuran presisi USA

**Output Prompt Example**
9:16 dynamic polaroid-style e-commerce poster for a rose gold ring. Keep all original foreground elements (ring, hand, logo) 100% intact. Keep the face in the original image clear and do NOT generate any new faces. Avoid placing text, labels, logos, or any other elements in the top 10% and bottom 10% of the 9:16 canvas. Add a single, slim, seamless polaroid border (width ≤5% of canvas width) with rounded corners and a subtle rose gold metallic texture, dynamically matched to the ring. Extend the border’s texture to the top 10% and bottom 10% safe zones for seamless blending with no visible dividing lines. Place "Warna elegan Rosegold" as a single continuous, unbroken line in the middle 80% safe area (above the polaroid image) in **elegant cursive font, warm rose gold**. Place "Ukuran presisi USA" as a single continuous, unbroken line in the middle 80% safe area (below the polaroid image) in **bold sans-serif, deep charcoal grey**. Only the text inside the double quotes is to be rendered; all other prompt content is technical instruction and must NOT appear on the poster. No overlapping with original image, no line breaks.
"""


def send_request(item):
    image_url = item['mainimageurl']
    selling_points = json.loads(item["selling_points"])

    # 1. prompt生成
    meta_prompt = poster_pip_add_text_9_16.format(first_selling_points=selling_points[0], second_selling_points=selling_points[1])
    prompt_bg = gemini3_client.make_image_request("", meta_prompt, [], [image_url], image_pils=[], max_tokens=5000, timeout=60)

    # 2. 图像生成
    res_bg = gemini_25_flash_image_gen(
        prompt_bg,
        image_urls=[image_url],
        image_pils=[],
        specify_gen_ratio=True,
        ratio="9:16",
        model_name="gemini-3-pro-image-preview",
        ak="4EJNjrbXCvsVNaYyYzlZBuWFUwtU8oLZ_GPT_AK",
        max_token=3000,
    )
    gen_url1 = save_tos(encode_pil_bytes(res_bg["image"], False))
    item["gen_url"] = [gen_url1]

    # 3. OCR校验
    ocr_result = hy_ocr_image_text(res_bg["image"])
    item["ocr_result"] = ocr_result
    is_correct = check_word_correct(" ".join(selling_points[:2]), ocr_result)
    item["is_correct"] = is_correct

    # 4. 准出校验
    src_image = load_or_download_image(image_url)
    src_image = resize_image_by_short_side(src_image, 512)
    res_image = resize_image_by_short_side(res_bg["image"], 512)
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
        if item["productRegion"] not in ["ID", "MY", "PH", "SG", "US"]:
            continue
        out.append(item)

    main(out, args.output_file, args.num_workers)

    print('Done!')
