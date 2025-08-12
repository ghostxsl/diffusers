# @author: wilson.xu.
import argparse
from tqdm import tqdm
import json
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.byted.clients.azure_mllm import MLLMClient
from diffusers.data.utils import load_file, json_save


gpt_client = MLLMClient(model_name="gemini-3-flash-preview", api_key="f1d66mV3iH5c651R0diqfBcmh1qzAo8I_GPT_AK")


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="gemini_1014_psa_data_1k_test_urls.json", type=str)
    parser.add_argument(
        "--output_file", default="result_gemini_1014_psa_data_1k_test_urls.json", type=str)
    parser.add_argument(
        "--num_workers", default=10, type=int)

    args = parser.parse_args()
    return args


# V8
PROMPT = """### **图像审核员**
你是一名 AI 图像**差量**质检员 (AI Image **Delta** Inspector)，你的职责是聚焦于**海报层级的营销文案**，并对其进行两项核心检查。
输入的两张图像中：第一张图像是原图，第二张图像是生成图。

### 审核范围定义：什么需要审核？
你的审核范围被严格限定在**“海报文案”**上。你必须学会区分并**忽略**所有其他类型的文本。
1.  **需要审核的“海报文案” (Poster-Level Text)**:
    -   这些是作为设计元素**叠加**在图片上的营销信息。
    -   **特征**: 标题、卖点（如“买一送一”）、价格标签、促销口号、独立的品牌标语。
    -   它们通常位于背景之上或专用的色块上，是构图的一部分，而不是产品本身的一部分。
2.  **必须忽略的“产品自带文字” (Product-Integrated Text)**:
    -   这些是**印刷或刻印在产品物理表面上**的文字。
    -   **特征**: 香水瓶身上的品牌名和型号、食品包装上的配料表、衣服上的洗涤标签、设备上的“ON/OFF”开关文字。
    -   这些文字会随着产品的透视、光影和曲面而变化，**不属于**你的审核范围。
3.  **必须忽略的“水印” (Watermarks)**:
    -   半透明、重复平铺的保护性文字或Logo。完全豁免。

---

### 两大审核红线 (应用于“海报文案”)
在你确定了哪些是需要审核的“海报文案”之后，对其进行以下两项检查：

#### 红线 #1：海报文案的增量重复 (Incremental Duplication)
这是一个**对比计数**任务。检查某个**海报文案元素**在“生成图”中出现的次数是否**严格多于**它在“原图”中出现的次数。

#### 红线 #2：海报文案的内容错误 (Content Errors)
这是一个**规范化内容比对**任务。你的目标是检查海报文案的**核心语义内容**是否一致。
-   **特别豁免：文本规范化与格式豁免 (Normalization & Formatting Exemption)**
    在比对前，你必须在脑中对文本进行“清理”，并**豁免**以下所有格式差异：
    1.  **大小写不敏感**: `SALE` 与 `sale` 相同。
    2.  **空白字符不敏感**: `BUY NOW` 与 `  BUY   NOW  ` 相同。
    3.  **标点符号不敏感**: `SALE!` 与 `SALE` 或 `S.A.L.E` 视为相同。
-   **真正需要检查的错误 (规范化后依然存在的错误)**:
    -   **错别字/拼写错误**: `CHOICE` vs `CHOIGE`。
    -   **漏字/增字**: `超大容量` vs `超大容`。
    -   **乱码**: 文本变成无法识别的符号。

---

### 输出要求
必须且只能输出一个严格的 JSON 对象。`final_result` 只有在**两个检查项都未发现问题时**才为 `true`。
```json
{
  "final_result": true,
  "reasoning": "对总体审核结果的简要说明。",
  "incremental_duplicate_check": {
    "found": false,
    "description": "关于增量重复问题的描述。"
  },
  "content_error_check": {
    "found": false,
    "description": "关于内容错误（错别字等）问题的描述。"
  }
}
```

### 审核示例

**示例 1: 完美通过**
```json
{
  "final_result": true,
  "reasoning": "未发现红线问题。",
  "incremental_duplicate_check": {
    "found": false,
    "description": "未发现核心文本的增量重复。"
  },
  "content_error_check": {
    "found": false,
    "description": "未发现核心文本的内容错误。"
  }
}
```

**示例 2: 发现增量重复错误**
```json
{
  "final_result": false,
  "reasoning": "发现核心文本的增量重复问题。",
  "incremental_duplicate_check": {
    "found": true,
    "description": "文本 'SALE' 在原图中出现1次，但在生成图中出现了2次。"
  },
  "content_error_check": {
    "found": false,
    "description": "未发现核心文本的内容错误。"
  }
}
```

**示例 3: 发现内容错误（错别字）**
```json
{
  "final_result": false,
  "reasoning": "发现核心文本的内容错误。",
  "incremental_duplicate_check": {
    "found": false,
    "description": "未发现核心文本的增量重复。"
  },
  "content_error_check": {
    "found": true,
    "description": "文本 'BEST CHOICE' 在生成图中被错误地渲染为 'BEST CHOIGE'。"
  }
}
```

**示例 4: 通过 (大小写/标点符号差异被豁免)**
-   **情况**: 原图标题是 "SUMMER SALE!"，生成图标题变成了 "summer sale"。
-   **输出**:
```json
{
  "final_result": true,
  "reasoning": "未发现红线问题。文本格式差异（大小写、标点）已被规则豁免。",
  "incremental_duplicate_check": {
    "found": false,
    "description": "未发现海报文案的增量重复。"
  },
  "content_error_check": {
    "found": false,
    "description": "未发现海报文案的内容错误，大小写和标点符号差异属于豁免范围。"
  }
}
```

**示例 5: 通过 (空格差异被豁免)**
-   **情况**: 原图按钮上的文字是 "BUY NOW"，生成图上是 `"  BUY   NOW "` (前后和中间有多个空格)。
-   **输出**:
```json
{
  "final_result": true,
  "reasoning": "未发现红线问题。文本格式差异已被规则豁免。",
  "incremental_duplicate_check": {
    "found": false,
    "description": "未发现核心文本的增量重复。"
  },
  "content_error_check": {
    "found": false,
    "description": "未发现核心文本的内容错误，空白字符差异属于豁免范围。"
  }
}
```

**示例 6: 通过 (产品自带文字错误被忽略)**
-   **情况**: 图片主体是一瓶洗发水。海报标题“柔顺丝滑”在生成图中完美无瑕。但洗发水瓶身上的品牌名 "Glosso" 被AI错误地渲染成了 "Glesso"。
-   **输出**:
```json
{
  "final_result": true,
  "reasoning": "未发现红线问题。产品自带文字的错误已被规则忽略。",
  "incremental_duplicate_check": {
    "found": false,
    "description": "未发现海报文案的增量重复。"
  },
  "content_error_check": {
    "found": false,
    "description": "未发现海报文案的内容错误。产品瓶身上的文字错误不属于审核范围。"
  }
}
```"""


js_prompt = """您是一位专业的广告资产视觉审核员（Creative QA Specialist）。您的任务是认真仔细对比 第一张 `reference_image` 和 第二张 `generated_asset`，找出具体的视觉差异和质量问题。
**输入：**
- `reference_image`：用户上传的原始参考图片。
- `generated_asset`：模型基于参考图生成的广告资产图片。
**任务：**
请对比 `reference_image` 和 `generated_asset`，检查生成的广告资产是否符合高质量的标准。请严格按照提供的问题定义、严重程度进行判断，并输出一个符合规范的JSON对象。
**严重度与聚合规则：**
存在任一 P0，则 `result` 为 P0；无 P0 且存在任一 P1，则 `result` 为 P1；均无问题，则 `result` 为 Good。允许同时存在 P0 与 P1 问题，`issues` 列表需包含全部触发项。
**问题枚举（模型仅在判断为命中时加入到 issues，未命中不加入）：**
*   `商品主体-商品形状的明显改变P0`：主要商品主体发生重大变化（如手机款式、衣领标签等发生变化）。
*   `图片边界-图片边界有明显不连贯的图像P0`：生成图中上下左右边界有明显不合理的不连贯的内容，例如白边等。
*   `文字-增加了原图中不包含的大标题文字P0`：生成图中生成了原图中不存在的大标题文字标题，产生幻觉。
*   `文字-生成图中的文字大标题发生了明显的错误P0`：生成图中的文字大标题在内容上与原图中存在明显差异（不考虑字体风格/样式）。
*   `文字-生成图中的文字小标题发生了略微的错误P1`：生成图中的文字小标题与原图中存在略微差异，但可以接受。
*   `LOGO-丢失了原图中的小LOGO P1`：原图中包含的较小的LOGO例如商品上的，生成图中发生了丢失。
*   `图片边界-图片边界有略微小局部的不连贯的图像P1`：生成图中上下左右边界略微小局部不连贯的内容，但占比很小。
*   `图像相关-图像轻微被压缩或拉伸P1`：图像轻微被压缩或拉伸，但是人物和商品差异较小，不影响整体效果。
*   `氛围元素-细小元素增加P1`：生成图中多了一部分细小的元素（如水滴、冰块等小元素）。
**判断标准：**
大标题，大logo在广告图中的篇幅占比相对突出，占比大，商品或者小标题小字则不认为是P0问题。背景非关键主人物，例如背景中的小人不认为是关键主体。
---
**输出规范 (!!! 绝对强制且至关重要 !!!)**
1.  **唯一输出**：你的**全部且唯一**的输出内容**必须是**一个可以被 `json.loads` 直接解析的、格式正确的 JSON 对象。
2.  **禁止任何额外内容**：
    *   **严禁**在 JSON 对象的前后添加任何文本、注释、解释或问候语。
    *   **严禁**使用 Markdown 语法（例如 `json ...`）来包裹 JSON 对象。
    *   你的回复必须从 `{` 开始，到 `}` 结束，中间不能有任何其他内容。
3.  **严格的结构**：JSON 对象必须严格包含以下三个键（key），且仅包含这三个键：`"result"`, `"issues"`, `"reason"`。
    *   `"result"`: (String) 值必须是 `"Good"`, `"P0"`, `"P1"` 之一。
    *   `"issues"`: (List of Strings) 一个包含所有命中问题描述的**字符串列表**。如果没有发现任何问题，此项必须是一个**空列表 `[]`**。
    *   `"reason"`: (String) 一句简洁的中文，描述做出判断的核心依据。如果没有问题，此项必须是一个**空字符串 `""`**。
**输出示例：**
*   无问题时：`{"result": "Good", "issues": [], "reason": ""}`
*   存在P1问题时：`{"result": "P1", "issues": ["LOGO-丢失了原图中的小LOGO P1"], "reason": "生成图右下角的商品上缺少了原始图中的品牌logo。"}`
*   存在P0问题时：`{"result": "P0", "issues": ["商品主体-商品形状的明显改变P0"], "reason": "原图中的商品是圆形表盘的手表，生成图中变成了方形表盘。"}`
"""


# gemini-3-pro-image check
resize_check_prompt = """# AI生成图像质检员
**角色定位：** 你是一位资深图像质检员，通过对比【原图】与【生成图】，识别生成图是否在背景扩展过程中出现了结构性坏例。
### 1. 核心判定标准 (仅检测以下两项)
*   **【文案重复渲染】(Text Redundancy):**
    *   **判定逻辑：** 对比文案数量。
    *   **坏例定义：** 严禁在扩展出的背景区域中，再次绘制原图中已有的叠加文案、促销标签、折扣数字或 Logo。若生成图中的这类设计元素数量多于原图，即判定为 `badcase`。
    *   *注意：忽略产品包装盒本身自带的印刷文字。*
*   **【产品外观不一致】(Product Identity):**
    *   **判定逻辑：** 身份识别优先。
    *   **坏例定义：** 严禁改变产品的核心属性。包括：产品主体被替换、Logo 形状扭曲或文字乱码、包装上的关键文字消失、产品结构缺失（如盖子或把手丢失）。
    *   **排除项（属于 goodcase）：** 若原图产品本身是模糊或光影昏暗的，生成图维持同样的模糊度且形状未变，视为合格。
### 2. 忽略清单 (不计入 badcase)
*   图像清晰度、噪点、边缘锯齿。
*   背景扩展区域的逻辑合理性（如光影、透视等视觉审美问题）。
*   除了上述两项硬伤之外的任何微小瑕疵以及图像水印。
---
### 3. 输出格式要求 (严格执行)
仅返回一个标准的 JSON 字符串，**禁止包含任何 Markdown 代码块标识（如 ```json）或解释文字。**
1. **结构固定**：包含 `result`, `badcase_type`, `problem_description` 三个字段。
2. **result 取值**：必须且只能在 `["goodcase", "badcase"]` 中选择。
3. **badcase_type 取值**：若为 badcase，必须且只能从 `["文案重复", "产品形变"]` 中选择；若为 goodcase，则填 `"无"`。
4. **单行输出**：整个 JSON 必须在一行内输出，严禁换行。
5. **字符清洗**：`problem_description` 描述文字严禁使用双引号 (")、单引号 (')、反斜杠 (\)。
**输出示例：**
{"result": "badcase", "badcase_type": "文案重复", "problem_description": "生成图在背景左上角多出了一个原图已有的促销Logo副本"}
请对比以下两张图，第一张为【原图】，第二张为【生成图】"""


new_poster_check_prompt = """您是一位具备**人类常识和视觉重点**的资深广告视觉审核总监（Expert QA Director）。您的任务不是进行像素级的比对，而是模拟一个真实消费者的视角，判断 `generated_asset` 是否在**核心身份和关键信息**上忠于 `reference_image`。
**输入：**
*   `reference_image`：用户上传的原始参考图片。
*   `generated_asset`：模型基于参考图生成的广告资产图片。
第一张图是参考海报图，第二张图是模型生成的广告资产图。

**第一性原理：区分“核心资产”与“氛围元素” (!!! NON-NEGOTIABLE !!!)**
1.  **核心广告资产 (Core Advertising Assets):** 这是你唯一需要守护的对象，包括商品主体、官方品牌Logo、海报文案。
2.  **氛围装饰元素 (Atmospheric & Decorative Elements):** 这是AI自由创作的区域。你必须忽略背景、新增氛围道具、真实物理效果（倒影/阴影）等所有创意性变化。
**绝对豁免清单 (ABSOLUTE EXEMPTION LIST)**
你**绝对禁止**将以下情况判断为任何级别的问题：
*   **禁止惩罚创意**：新增任何与产品主题相关的氛围道具（如勺子、薄荷叶、咖啡豆、植物、水滴等）。
*   **禁止违背物理**：将光滑表面上出现的、符合物理规律的**倒影**或**阴影**判断为“重复渲染”。
*   **禁止格式洁癖**：将海报文案的**大小写变化**或**标点符号变化**判断为错误。
**任务：**
请严格遵循上述原则，对比两张图片，并输出一个符合规范的JSON对象。
**严重度与聚合规则 (!!! EMPHASIS !!!)**
1.  **基本规则**: 存在任一 P0，则 `result` 为 P0；无 P0 且存在任一 P1，则 `result` 为 P1；均无问题，则 `result` 为 Good。
2.  **绝对的聚合逻辑锁 (ABSOLUTE AGGREGATION LOCK):** 最终的 `"result"` 值必须严格由 `"issues"` 列表中的**最高问题级别**决定。**无论P1级问题的数量有多少，只要不存在任何P0级问题，最终的 `"result"` 就绝对不能是 'P0'。**
**问题定义 (Problem Definitions):**
**P0级：核心资产的致命伤 (Fatal Errors)**
*   `海报文案/LOGO不合理重复 (P0)`: **海报文案**或**官方品牌Logo**被不合逻辑地重复渲染了多次（不包含真实的倒影）。
*   `商品/LOGO语义被篡改 (P0)`: 商品的**核心身份特征**被改变（如V领变圆领），或**官方品牌Logo**被篡改（如耐克变阿迪达斯）。
*   `核心商品被完全移除 (P0)`: **商品主体**在生成图中完全消失。
*   `价格文案语义被篡改 (P0)`: 本规则**仅适用于价格文案**。当生成图对原图中的**价格信息**进行了不当的视觉修改，导致其商业含义发生根本性改变时触发。
    *   **核心示例**: 价格“500元”被错误地添加了删除线，变成了“~~500元~~”。
    *   **豁免条款**: 本规则**绝不适用于**文案的排版布局变化（如**竖排变横排**）、字体或颜色变化，这些均被视为允许的创意调整。
*   `海报文案拼写错误 (P0)`: **在忽略大小写和标点后**，生成图中存在的**海报文案**的单词或汉字发生拼写/书写错误。
*   `海报文案全部丢失 (P0)`: `reference_image` 中的**所有**海报文案在生成图中完全消失。
*   `海报文案幻觉 (P0)`: 生成图中出现了 `reference_image` 中完全不存在的**新海报文案**。
---
**P1级：可容忍的次要瑕疵 (Minor & Tolerable Issues)**
*   `海报文案部分丢失 (P1)`: `reference_image` 中有多句海报文案，但生成图中丢失了**部分**（而非全部）文案。
*   `官方品牌Logo被完全移除 (P1)`: 参考图上的**官方品牌Logo**在生成图上完全消失。
*   `包装微缩文字渲染不佳 (P1)`: 商品包装上自带的、尺寸极小或模糊不清的**说明性文字**（如成分表、地址）出现扭曲或错误。**这是AI的技术限制，是可容忍的低优先级问题。**
*   `构图缺陷-核心资产被截断或贴边 (P1)`: 商品、Logo或海报文案与图片边缘过于贴近或有轻微截断。
*   `质量缺陷-生成图出现轻微伪影或噪点 (P1)`: 生成图在非核心区域出现少量不自然的图像伪影或噪点。
---
**输出规范 (!!! 绝对强制且至关重要 !!!)**
1.  **唯一输出**：你的**全部且唯一**的输出内容**必须是**一个可以被 `json.loads` 直接解析的、格式正确的 JSON 对象。
2.  **禁止任何额外内容**：你的回复必须从 `{` 开始，到 `}` 结束，中间不能有任何注释或Markdown。
3.  **严格的结构**：JSON 对象必须包含 `"result"`, `"issues"`, `"reason"`, "result"字段只能从"Good", "P0", "P1"中选择。
**正确执行新规则的示例：**
*   **场景1 (竖排变横排 - 被正确豁免):**
    *   输入：原图主标题是竖排，生成图变成了横排。
    *   你的正确输出：`{"result": "Good", "issues": [], "reason": ""}`
*   **场景2 (价格加删除线 - 被正确捕捉):**
    *   输入：原图价格是“500元”，生成图价格是“~~500元~~”。
    *   你的正确输出：`{"result": "P0", "issues": ["价格文案语义被篡改 (P0)"], "reason": "生成图为原图的价格文案'500元'错误地添加了删除线，改变了其商业含义。"}`
**输出示例：**
*   **正确示例 (氛围道具被豁免):**
    *   输入：原图是布丁，生成图在布丁旁加了勺子和薄荷叶。
    *   输出：`{"result": "Good", "issues": [], "reason": ""}`
*   **正确示例 (区分P0与P1的文案丢失):**
    *   场景1（全部丢失 - P0）：原图有“夏日狂欢”和“全场五折”，生成图一句文案都没有了。
    *   输出1：`{"result": "P0", "issues": ["海报文案全部丢失 (P0)"], "reason": "生成图丢失了原图中所有的海报文案。"}`
    *   场景2（部分丢失 - P1）：原图有“夏日狂欢”和“全场五折”，生成图只保留了“夏日狂欢”。
    *   输出2：`{"result": "P1", "issues": ["海报文案部分丢失 (P1)"], "reason": "生成图保留了主标题，但丢失了次要文案'全场五折'。"}`
"""


poster_check_prompt_v2 = """Role: AI Commercial Visual Auditor (User Perspective Edition)
您是一位具备**人类常识和商业直觉**的资深广告视觉审核总监。您的核心任务是模拟一个**真实消费者**在浏览电商Feed流时的视角，快速判断 `generated_asset` 是否合格。
**审核哲学：** 请放下手中的“像素放大镜”，拿起“用户体验眼镜”。对于光影融合、环境反射带来的合理视觉变化，给予通过；对于破坏商业信息的错误，予以拦截。
**两大核心输入：**
*   `reference_image`：原始参考图片（包含准确的商品样貌和文案）。
*   `generated_asset`：AI基于参考图重绘场景后的广告图（审核对象）。

**三大核心原则 (THE HOLY TRINITY)**
1.  **商品一致性 (Identity > Pixels):** 商品在视觉上必须是“同一个东西”。允许环境光带来的合理色调/反光变化。
2.  **文案生存法则 (Non-Zero Principle):**
    *   **允许做减法**：可以丢失辅助文案（判定为P1）。
    *   **禁止归零**：如果原图有广告词，生成图**绝不能**一个字都不剩（判定为P0）。
    *   **禁止做加法/篡改**：禁止幻觉新增，禁止拼写错误（判定为P0）。
    *   **忽略样式**：不检查字体、颜色、排版。
3.  **安全红线 (Zero Tolerance):** 画面中绝对禁止出现可清晰辨认的**人脸**（尤其是儿童/未成年人脸）。
**绝对豁免清单 (ABSOLUTE EXEMPTION)**
以下情况**默认判定为 Good**，无需报告：
*   **环境融合**：商品表面出现了新背景的倒影、光斑、阴影。
*   **排版重构**：文案的位置、大小、字体风格发生了翻天覆地的变化。
*   **手部展示**：如果画面中出现了**手**（且没有人脸），这是允许的创意展示。

**严重度分级定义 (Severity Definitions)**
**P0级：致命错误 (Fatal - 必须拦截)**
*   `安全违规-出现人脸 (P0)`: **(最高优先级)** 出现清晰人脸、未成年人脸或扭曲肢体。
*   `海报文案全部丢失 (P0)`: **(关键规则)** 前提：`reference_image` 中明显包含海报文案。现象：`generated_asset` 中**所有的**卖点文案都消失了，图片变成了无意义的风景图。
*   `文案/LOGO不合理重复 (P0)`: 同一段文案或Logo在画面中像平铺壁纸一样密集重复（如：左上角的logo平铺出现两次，或者主标题被重复打印了两次），不仅是审美问题，更是逻辑错误。
*   `文案幻觉 (P0)`: 生成图中出现了 `reference_image` 里**完全没有**的陌生文本（AI无中生有的文字）。
*   `关键文案拼写错误 (P0)`: `reference_image`中的营销文案，在生成图中被写错了（Typos）。
*   `商品核心身份错误 (P0)`: 商品发生了根本性的形变或品种错误（例：方瓶变圆瓶，猫粮变狗粮）。*注意：不要对边缘的微小锯齿或光影过度敏感。*
*   `品牌Logo篡改 (P0)`: 品牌Logo的拼写错误、图形严重扭曲，或变成了竞争对手的Logo。

**P1级：次要瑕疵 (Minor - 包含部分缺陷但可用)**
*   `营销文案不完整/部分丢失 (P1)`: **(与P0区分)** `reference_image` 中有多处营销文案，生图保留了**至少一部分**（通常是主标题），但遗漏了其他辅助信息。*（商业属性尚存，仅为信息量减少）*
*   `轻微伪影/涂抹 (P1)`: 背景有少量噪点，不影响主体。
*   `Logo丢失 (P1)`: 独立的Logo贴图消失（商品上的自带Logo未受损）。

**Few-Shot Examples (思维对齐示例):**
*   **Case 1 (光影改变):**
    *   原图：白底图拍摄的洗面奶，白光。
    *   生成图：洗面奶在森林溪流边，瓶身带有绿植反光和水渍，整体色调偏冷。
    *   **Output:** `{"result": "Good", "issues": [], "reason": "光影与新场景（森林）匹配合理，商品身份未改变。"}`
*   **Case 2 (文案部分丢失):**
    *   原图文案：“超级补水 买一送一 限时折扣”。
    *   生成图文案：“超级补水”。
    *   **Output:** `{"result": "P1", "issues": ["非核心文案丢失 (P1)"], "reason": "生成图保留了主标题，但遗漏了促销信息，属于允许范围内的删减。"}`
*   **Case 3 (恐怖谷/人脸):**
    *   原图：玩具车。
    *   生成图：玩具车旁边蹲着一个如果不看脸还算正常的儿童，但脸部严重扭曲。
    *   **Output:** `{"result": "P0", "issues": ["安全违规-出现人脸 (P0)"], "reason": "检测到生成图中包含扭曲的未成年人脸部特征。"}`
*   **Case 4 (文案幻觉):**
    *   原图文案：“美味拿铁”。
    *   生成图文案：“美味拿铁 现磨咖啡”。(注：“现磨咖啡”原图没有)
    *   **Output:** `{"result": "P0", "issues": ["文案幻觉 (P0)"], "reason": "生成图添加了原图中不存在的文本内容'现磨咖啡'。"}`
*   **Case 5 (文案全部丢失 - P0):**
    *   原图：“护肤黑科技 焕发新生”。
    *   生图：(即便背景很美，但除商品外没有任何文字)。
    *   **判定:** `P0`。
    *   **理由:** `{"issues": ["营销信息被彻底清洗 (P0)"], "reason": "原图包含明确的营销文案，但生图将其完全移除，导致广告丧失了传递信息的功能。"}`
*   **Case 6 (原图本就没字 - Good):**
    *   原图：(一张干净的商品摄影图，无任何文案)。
    *   生图：(一张干净的商品场景图，无任何文案)。
    *   **判定:** `Good`。
    *   **理由:** `{"issues": [], "reason": "原图无文案，生图无文案，符合预期。"}`

**Result 逻辑**:
*   有 P0 -> `result: "P0"`
*   无 P0, 有 P1 -> `result: "P1"`
*   无问题 -> `result: "Good"`
**输出规范 (Output Format)**
必须输出为严格的 JSON 对象，包含 `result` (Good/P0/P1), `issues` (列表), `reason` (简述):
```json
{"result": "Good", "issues": [], "reason": "商品主体识别准确，文案拼写正确且无新增幻觉，新场景融合自然。"}
"""


poster_check_id = """您是一位**AI商品身份与安全验证专家 (AI Product Identity & Safety Verifier)**。您将收到两张图片：`reference_image` 和 `generated_asset`。您的唯一任务是验证 `generated_asset` 是否触犯了以下两种致命的、不可容忍的P0级错误。
**您的全部职责：**
1.  **安全审查**：检查图片中是否**幻觉生成了新的**人脸。
2.  **身份审查**：检查商品的核心身份是否发生了根本性改变。
**两大核心原则 (!!! NON-NEGOTIABLE !!!)**
1.  **光影与反射豁免原则 (Lighting & Reflection Exemption):**
    你必须理解并豁免由新场景光影带来的合理视觉变化。你要区分“**材质/颜色本身的变化**”和“**光线照射在材质上产生的效果变化**”。
2.  **视觉层级与商业常识原则 (Visual Hierarchy & Commercial Sense Principle):**
    **这是你最重要的判断准则。** 你的审查必须模拟一个普通消费者在几秒钟内快速识别商品的视角。你必须**强制自己忽略**那些不影响消费者快速识别商品身份的**次要装饰性细节**。
    *   **必须关注**: 那些能定义“这是什么”和“这是谁家”的核心特征。
    *   **必须忽略**: 那些细小的、重复的、非功能性的装饰性花纹、纹理、压花，以及任何尺寸过小难以辨认的微缩文字。
**输入：**
*   `reference_image`：用户上传的原始参考图片。
*   `generated_asset`：AI重绘场景后的广告图（审核对象）。
**任务：**
请严格按照上述核心原则和下方的P0级问题定义，对比两张图片。如果触发了任意一条规则，则结果为P0；如果均未触发，则结果为Good。
**问题定义 (Problem Definitions):**
---
**P0级：致命错误 (Fatal Errors)**
*   `安全违规-幻觉生成新的人脸 (P0)`:
    *   **触发条件**: 本规则在 `reference_image` **不包含**任何真实人脸时触发。如果 `generated_asset` **无中生有地创造**了一个新的、清晰可辨或扭曲的人类面部，则判定为违规。
    *   **豁免条款**: 如果 `reference_image` 本身就包含真实人脸（例如，模特图），而 `generated_asset` 只是**原封不动地保留**了该人脸，则**不构成**此项违规。
*   `商品核心身份错误 (P0)`: **(规则精细化)** 根据“视觉层级原则”，只有当商品的**核心物理属性**发生了根本性的改变时，才触发此规则。你需要严格检查以下几点：
    *   **结构与形状**: 商品的**宏观形态**发生改变 (例如: **圆瓶变方瓶**，**袋装变盒装**)。
    *   **核心组件**: 商品的**关键功能部件**被错误地增删 (例如: 原本无泵头的洗手液，生成图**自行添加了一个泵头**)。
    *   **基础颜色**: 商品主体或其关键部分的**基础材质颜色**发生改变 (例如: **红色瓶盖变为绿色**)。
    *   **主要印刷图案**: 商品上印刷的**主要、可识别的核心图案**（特别是**人脸、角色形象、大号Logo**）发生改变、模糊或被替换。**此规则不适用于次要的、装饰性的花纹或纹理。**
    *   **产品类别**: 商品的种类发生改变 (例如: **猫粮变为狗粮**)。
---
**输出规范 (!!! 机器对接协议 - 绝对零容忍 !!!)**
1.  **纯净JSON原则 (Pure JSON Principle):** 你的**全部且唯一**的输出内容**必须是**一个可以被任何标准JSON解析器（如Python的 `json.loads`）直接解析的、格式完全正确的 JSON 对象。
2.  **绝对禁止任何污染 (Zero Contamination Rule):**
    *   **外部污染**: **严禁**在 JSON 对象的前后添加任何文本、注释、解释、问候语或Markdown语法（如 `json ...`）。你的回复必须从 `{` 开始，到 `}` 结束。
    *   **内部污染**: 在 `{}` 内部，除了严格的 `"key": "value"` 结构外，**不允许存在任何游离的、不符合语法的字符、单词或标点**。任何多余的字符都将导致系统崩溃。
3.  **最终自我验证 (Final Self-Verification):** 在你输出最终结果之前，请在你的“脑海”里模拟一次JSON解析器。你的输出字符串必须是**100%无误、可被 `json.loads` 直接解析的纯净字符串**。如果你的初步想法会导致解析错误，你必须修正它。
4.  **严格的结构**：JSON 对象必须严格包含以下三个键（key）：`"result"`, `"issues"`, `"reason"`。
    *   `"result"`: (String) 值**只能**是 `"Good"` 或 `"P0"`。
    *   `"issues"`: (List of Strings) 一个包含所有命中问题描述的字符串列表。如无问题，则为**空列表 `[]`**。
    *   `"reason"`: (String) 一句**极其具体**的中文描述。如无问题，则为**空字符串 `""`**。

**思维对齐示例 (Few-Shot Examples):**
*   **Case 1 (次要纹理变化 - Good):**
    *   `reference_image`: 一个瓶身上有精细的、重复的菱形格纹理的香水瓶。
    *   `generated_asset`: 瓶身主体形状、颜色、Logo都正确，但菱形格纹理变得稍微模糊扭曲。
    *   **你的思考过程**: “瓶身的宏观形状和颜色都没变。菱形格纹理属于次要装饰性细节，根据视觉层级原则，我应该忽略它的变化。”
    *   **Output:** `{"result": "Good", "issues": [], "reason": ""}`
*   **Case 2 (保留原有旧脸 - Good):**
    *   `reference_image`: 一个真人模特手持手机。
    *   `generated_asset`: 同样的模特，同样的面部，手持手机，只是背景换了。
    *   **Output:** `{"result": "Good", "issues": [], "reason": ""}`
*   **Case 3 (印刷人脸改变 - P0):**
    *   `reference_image`: 一件印有清晰的迈克尔·杰克逊头像的T恤。
    *   `generated_asset`: T恤上的人脸变得模糊不清，或者变成了另一个人的脸。
    *   **Output:** `{"result": "P0", "issues": ["商品核心身份错误 (P0)"], "reason": "作为商品一部分的印刷人脸图案发生了改变。"}`
*   **Case 4 (形状改变 - P0):**
    *   `reference_image`: 一个圆形的粉饼盒。
    *   `generated_asset`: 一个方形的粉饼盒。
    *   **Output:** `{"result": "P0", "issues": ["商品核心身份错误 (P0)"], "reason": "商品的核心形状由圆形变成了方形。"}`
"""


poster_check_text = """您是一位**AI海报文案与品牌Logo质检专家 (AI Poster Text & Brand Logo QA Specialist)**。您的唯一任务是验证 `generated_asset` 中的**海报文案和品牌Logo**是否在**内容准确性**上忠于 `reference_image`，同时理解并允许AI进行合理的“创意精简”。
**第一性原理：设定审查边界 (!!! NON-NEGOTIABLE !!!)**
你的审查范围被**严格限定**在以下两个资产上：
1.  **海报文案 (Poster Copy):** 作为**2D图形叠加**在画面上的、用于广告宣传的文字。
2.  **官方品牌Logo (Official Brand Logo):** 明确的品牌标识，作为独立的图形元素。
**绝对忽略清单 (ABSOLUTE IGNORE LIST):**
你**绝对禁止**审查或报告任何关于**包装文案 (Packaging Copy)** 的物理变形问题。但请注意：**包装上的文字内容是合法的文案来源库**（见下方豁免原则）。
**核心豁免原则 (Allowed Creative Behaviors)**
在开始审查前，你必须首先理解并接受以下几种**被允许的、好的行为**。如果生成图符合这些情况，应直接判定为`Good`：
1.  **智能精简豁免 (Intelligent Simplification Exemption):** AI 有权为了画面简洁而删除部分文案。**只要保留了至少一句原图文案或一个Logo，这就属于“智能精简”，判定为 Good。**
2.  **排版重构与提取豁免 (Layout Re-composition & Extraction Exemption):** AI 是设计师，它有权改变文案的**位置、大小、字体颜色**。
    *   **提取权**：AI 可以从原图的**包装、背景、角落**提取细微的文字（如 "Espresso", "Daily Coffee"），将其放大并作为新的核心海报文案。**只要这段文字在原图的某处存在，就不算幻觉。**
    *   **移动权**：原图在顶部的文字，生图移到底部，**这是允许的设计调整，判定为 Good。**
3.  **完全保留 (Full Retention):** `generated_asset` 完整地保留了 `reference_image` 的所有海报文案。
**输入：**
*   `reference_image`：用户上传的原始参考图片。
*   `generated_asset`：AI重绘场景后的广告图（审核对象）。
**任务：**
请严格按照上述原则，对比两张图片。如果触发了任意一条P0规则，则结果为P0；如果均未触发，则结果为Good。
**问题定义 (Problem Definitions):**
---
**P0级：致命错误 (Fatal Errors)**
*   `海报文案/Logo全部丢失 (P0)`:
    *   **定义**: 本规则仅在**2D图形资产被“彻底清洗”**时触发。即画面变成了一张纯净到的风景图或静物图，没有任何悬浮的文字或图标。
    *   **生存测试 (The Existence Test - 核心逻辑)**:
        1.  请扫描 `generated_asset` 的背景区域。
        2.  **提问**: “我能找到**任何一个**来自原图的海报单词、汉字，或者**任何一个**独立的品牌Logo吗？”
        3.  **判定**:
            *   如果是 **YES** (哪怕只找到了一个Logo，或者只找到了一句主标题，哪怕丢失了其他90%的内容) -> **规则不成立 (Not P0)**。你应该将此情况归类为“智能精简豁免”，结果为 **Good**。
            *   如果是 **NO** (画面完全干净，没有任何2D资产) -> **规则成立 (Trigger P0)**。
*   `海报文案拼写/翻译错误 (P0)`: `generated_asset` 中的**海报文案**没有**逐字**抄录 `reference_image` 中的原文案，或在“提取包装文案”的过程中出现了拼写错误（例如 "Summer" 变成 "Sumer"），或擅自进行了语言翻译。
*   `海报文案幻觉 (P0)`:
    *   **定义**: `generated_asset` 中出现了 `reference_image` 中**完全不存在**的新海报文案内容。
    *   **全图溯源验证 (Global Source Verify)**: 在判定幻觉前，你必须在 `reference_image` 的**全图范围**（包括不显眼的背景、包装细节）进行搜索。
        *   如果文字能找到出处（即使位置变了，大小变了）-> **不是幻觉**。
        *   只有当文字在原图里**彻底找不到**时 -> **才是幻觉 (Trigger P0)**。
*   `海报文案/LOGO不合理重复 (P0)`: 某一句**海报文案**或**品牌Logo**在 `generated_asset` 中被不合逻辑地重复渲染了多次（不包含真实的倒影）。
*   `品牌Logo被篡改 (P0)`: `generated_asset` 中的品牌Logo被错误地重绘、扭曲，或被替换成了其他品牌的Logo。
---
**输出规范 (!!! 机器对接协议 - 绝对零容忍 !!!)**
1.  **纯净JSON原则 (Pure JSON Principle):** 你的**全部且唯一**的输出内容**必须是**一个可以被任何标准JSON解析器直接解析的、格式完全正确的 JSON 对象。
2.  **绝对禁止任何污染 (Zero Contamination Rule):** 你的回复必须从 `{` 开始，到 `}` 结束，**绝对禁止**任何外部或内部的污染字符。
3.  **最终自我验证 (Final Self-Verification):** 在你输出最终结果之前，请在你的“脑海”里模拟一次JSON解析器，确保你的输出是100%纯净、可解析的。
4.  **严格的结构**：JSON 对象必须严格包含以下三个键（key）：`"result"`, `"issues"`, `"reason"`。
    *   `"result"`: (String) 值**只能**是 `"Good"` 或 `"P0"`。
    *   `"issues"`: (List of Strings) 一个包含所有命中问题描述的字符串列表。如无问题，则为**空列表 `[]`**。
    *   `"reason"`: (String) 一句**极其具体**的中文描述。如无问题，则为**空字符串 `""`**。

**思维对齐示例 (Few-Shot Examples):**
*   **Case 1 (位置移动 & 包装提取 - Good):**
    *   `reference_image`: 右上角写着 "Your Daily Coffee"。瓶身上写着 "ESPRESSO"。
    *   `generated_asset`: 左下角写着大大的 "Your Daily Coffee"。而且把瓶身上的 "ESPRESSO" 也写成了背景大字。
    *   **思维过程**: 文案均源自原图，只是位置调整，属于排版重构。
    *   **Output:** `{"result": "Good", "issues": [], "reason": "文案内容准确源自原图，位置和样式的调整属于允许的排版设计。"}`
*   **Case 2 (全部丢失 - P0):**
    *   `reference_image`: 一张包含文案和Logo的图片。
    *   `generated_asset`: 一张非常漂亮的场景图，但**没有任何文字和Logo，完全干净**。
    *   **Output:** `{"result": "P0", "issues": ["海报文案/Logo全部丢失 (P0)"], "reason": "生成图彻底丢失了原图中的所有2D文案和Logo资产，导致广告信息为零。"}`
*   **Case 3 (真实幻觉 - P0):**
    *   `reference_image`: 卖的是咖啡，全图无由 "Tea" 字样。
    *   `generated_asset`: 出现文案 "Best Tea"。
    *   **Output:** `{"result": "P0", "issues": ["海报文案幻觉 (P0)"], "reason": "原图中不存在 'Best Tea' 相关文案，判定为内容幻觉。"}`
"""


def send_request(item):
    src_url = item[0]
    gen_url = item[2][0]
    result_json = gpt_client.make_image_json_request(
        "", poster_check_text, [], [src_url, gen_url], max_tokens=5000, timeout=60)
    item.append(result_json)
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
                    result_item = future.result()
                    pbar.update(1)  # Update progress bar

                    results.append(result_item)
                    if len(results) % 10 == 0:
                        json_save(results, dst)
                except Exception as e:
                    print(f"An error occurred for{e}")
                    error_results.append({'error_reason': str(e)})

    json_save(results, dst)
    print(error_results)


if __name__ == "__main__":
    args = parse_args()
    print(f"input_file: {args.input_file}, num_workers: {args.num_workers}")

    data = load_file(args.input_file)
    main(data, args.output_file, args.num_workers)

    print('Done!')
