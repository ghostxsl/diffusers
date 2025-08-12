# @author: wilson.xu.
import argparse
from tqdm import tqdm
import json
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.byted.clients.azure_mllm import MLLMClient
from diffusers.data.utils import load_file, json_save, load_csv_or_xlsx_to_dict
from diffusers.data.outer_vos_tools import load_or_download_image


gpt_client = MLLMClient(model_name="gemini-2.5-flash", api_key="3H7OHTGn3JwHvlZHP9JUJzb850gp3TGR_GPT_AK")


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default="seedream5_poster_redraw_label_answer.csv", type=str)
    parser.add_argument(
        "--output_file", default="result_seedream5_poster_redraw_label_answer.json", type=str)
    parser.add_argument(
        "--num_workers", default=10, type=int)

    args = parser.parse_args()
    return args


physics_sanity_check_prompt = """你是一位拥有设计、摄影和物理三重知识的AI图像质检员。你的任务是判断输入的电商海报图，在其描绘的**物理场景**中，是否存在违反物理常识的现象。你必须严格区分场景类型、设计元素、拍摄视角，并理解**画面截断**的构图手法。

### 质检流程 (Inspection Flow)

#### 步骤 1: 场景与视角综合分析 (Scene & Perspective Analysis)

首要任务是判断图像的“背景环境”和“拍摄视角”。

*   **A. 抽象/概念性背景:**
    *   **特征:** 图像主体漂浮在纯色、渐变色、光效或非现实背景前。
    *   **动作:** **判定为 `Good`**。主体的悬浮是可接受的设计风格。

*   **B. 物理环境:**
    *   **特征:** 图像主体被放置在可识别的现实场景或平面上（如桌面、房间、户外）。
    *   **此时，你必须进一步分析拍摄视角：**

        *   **B1. 俯视/平铺视角 (Top-Down / Flat Lay):**
            *   **特征:** 像是从物体正上方垂直向下拍摄，物体平铺在表面上。
            *   **规则:** 物体“看似悬浮”或“看似站立”通常是平放的视觉效果。
            *   **动作:** 只要符合平铺逻辑，**判定为 `Good`**。

        *   **B2. 标准/倾斜视角 (Standard / Angled):**
            *   **特征:** 具有正常的3D空间感、透视和景深。
            *   **动作:** **进入“步骤 3”**，对此类场景进行最严格的物理规律检查。

#### 步骤 2: 设计元素排查 (Design Element Exclusion)

在所有判断之前，必须识别并忽略以下设计元素：
*   文本、图标、Logo、UI元素、色板。
*   图中图/附图 (Picture-in-Picture)。

#### 步骤 3: 物理规律检查 (Physics Checks)

**此步骤仅在步骤1判定为“B2. 标准/倾斜视角”时严格应用。**

1.  **稳定与平衡 (Stability & Balance):** 在有3D透视感的场景中，薄片状物体是否在没有倚靠物的情况下以窄边不合理地站立？

2.  **重力与支撑 (Gravity & Support):**
    *   **核心定义:** 一个**真正**违反物理的“悬浮”，必须满足一个**关键条件**：**该物体必须完全可见于画面之内，但又没有任何可见的支撑。**
    *   **豁免规则 (Frame Truncation Exception):** 在判断一个物体是否悬浮前，必须检查它是否被画框边缘（上、下、左、右）截断。
        *   **如果物体被截断：** 你必须假定其支撑结构（如手、手臂、支架、物体的其余部分）存在于画面之外。**这不属于物理错误**。
        *   **如果物体未被截断（完全可见）：** 此时才应用悬浮检查。它是否有支撑？如果没有，这才是真正的物理错误。

### 输出格式

你必须严格以以下JSON格式返回你的质检结果。不要在JSON前后添加任何额外的解释或文字。

```json
{
  "result": "Good | P0",
  "issues": [],
  "reason": "极其具体的一句话中文总结"
}

### 字段说明
"result": (String) 如果发现任何违反物理规律的问题，值必须为 "P0"。如果完全没有问题，值为 "Good"。
"issues": (List of Strings) 一个字符串列表，包含所有命中的问题。可用的问题描述包括：
- "Object floating without support" (物体无支撑漂浮)
- "Unstable object standing without support" (不稳定物体无支撑站立)
- "Objects stacked without contact" (物体隔空堆叠)
如果无问题，此列表必须为空列表 []。
"reason": (String) 用一句极其具体的中文来描述最核心的问题。如果存在多个问题，请概括。例如：“手机和配件在空中漂浮。” 或 “蓝色碗悬浮在粉色碗上方。”。如果无问题，此字符串必须为空字符串 ""。
"""


def send_request(item):
    urls = json.loads(item["图片信息-图片原图"])
    gen_img = load_or_download_image(urls["gen_img_url_0"])
    result_check_json = gpt_client.make_image_json_request(
        "", physics_sanity_check_prompt, image_urls=[], image_pils=[gen_img], max_tokens=4000, timeout=60)
    item["check_result"] = result_check_json

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
                    print(f"An error occurred for {e}")
                    error_results.append({"item": item, "error_reason": str(e)})

    json_save(results, dst)
    print(len(error_results))


if __name__ == "__main__":
    args = parse_args()
    print(f"input_file: {args.input_file}, num_workers: {args.num_workers}")

    data = load_csv_or_xlsx_to_dict(args.input_file)
    main(data, args.output_file, args.num_workers)

    print('Done!')
