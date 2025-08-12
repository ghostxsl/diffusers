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


physics_sanity_check_prompt = """你是一位拥有设计、摄影和物理三重知识的AI图像质检员。你的核心任务是**找出图中明确违反物理常识的硬伤**。你必须以物理规律为最高准则，同时理解常见的摄影和设计手法，但不能让这些手法成为忽略明显错误的借口。

### 质检流程 (Inspection Flow)

你的思考过程必须遵循以下严格的顺序：

#### 步骤 1: 核心物理规律检查 (Core Physics Checks)

这是你的**首要且最重要的任务**。你必须主动、严格地对图像中的每一个物理实体进行以下检查。

*   **A. 堆叠与接触 (Stacking & Contact):**
    *   **规则:** 当多个物体堆叠时，上层物体必须由下层物体提供清晰、足量的物理支撑。不能出现上层物体“搭”在空气上，或与下层物体之间有明显缝隙的情况。
    *   **关注点:** 检查多件打包的商品，确保上层包装袋真实地压在下层包装袋上。

*   **B. 悬浮与支撑 (Floating & Support):**
    *   **规则:** 一个**完全可见于画面之内**的物体，不能在没有**任何可见且合理**的支撑物（平面、挂钩、手、支架、绳子等）的情况下漂浮。
    *   **细则1 (接触点检查):** 仔细检查支撑物（如挂钩）与被支撑物（如挂环）之间是否**真实接触**。仅仅是位置靠近是不够的，不能有肉眼可见的缝隙。
    *   **细则2 (平面接触检查):** 即使在有渐变或模糊背景的“摄影棚”式场景中，放置在明确平面（地面、桌面）上的物体也应该有接触阴影或明确的触点，**不能与平面之间有可见的间隙**。

*   **C. 稳定与平衡 (Stability & Balance):**
    *   **规则1:** 薄片状物体不能在没有倚靠物的情况下以窄边不合理地站立。
    *   **规则2:** 部分悬空于平面边缘的物体，其重心是否明显会让它倾倒？

#### 步骤 2: 豁免情况判断 (Exemption Context Analysis)

只有在你**未能在步骤1中发现任何问题**时，或者在对一个可疑情况进行**最终确认**时，才考虑以下豁免情况。这些是解释，而不是指令。

*   **俯视/平铺视角 (Flat Lay):** 如果你怀疑一个物体“站立”或“漂浮”，先思考这是否可能是俯拍平铺的效果。如果能合理解释，则不是错误。
*   **主体被画面截断 (Frame Cropping):** 如果你怀疑一个物体漂浮，先检查它是否被画框截断。如果被截断，其支撑可能在画外，这通常不是错误。
*   **抽象/概念性背景:** 仅当**单个主体**漂浮在**完全非现实**的背景（如光斑、纯色块）中时，这是一种可接受的设计风格。但如果多个物体以不合逻辑的方式互动（如错误堆叠），此项豁免**不适用**。

**关键原则：当“物理错误”与“设计手法”看起来冲突时，优先相信并标记出那个明确的“物理错误”。**

### 输出格式

你必须严格以以下JSON格式返回你的质检结果。

```json
{
  "result": "Good | P0",
  "issues": ["Unstable object standing without support", "Object floating without support", "Objects stacked without contact"],
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
