import argparse
from tqdm import tqdm
from typing import List
import base64
import json
import re
import concurrent
from concurrent.futures import ThreadPoolExecutor
from diffusers.data.utils import load_csv_or_xlsx_to_dict, json_save

from overpass_ad_creative_qwen_image_v1.clients.rpc.ad_creative_qwen_image_v1 import AdCreativeQwen_Image_V1Client


# 部署服务：https://ml.tiktok-row.net/deployment/serviceList/service/detail/ad.creative.reward_model_rl_eval_v2/deploy/2075984
client = AdCreativeQwen_Image_V1Client(psm='ad.creative.reward_model_rl_eval_v2', idc='maliva', cluster='default')


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--input_file", default=None, type=str)
    parser.add_argument(
        "--output_file", default=None, type=str)
    parser.add_argument(
        "--num_workers", default=10, type=int)

    args = parser.parse_args()
    return args


def get_ctr_comparison(image_paths: List[str], client: AdCreativeQwen_Image_V1Client):
    """
    对比多张图片的CTR表现并返回各自评分
    Args:
        image_paths (List[str]): 图片文件路径列表（支持URL或本地路径）
        prompt (str): 提示词
    Returns:
        Tuple[int, ...]: 每张图片的CTR评分 (1-100)
    """
    title = "Shop deals now."
    prompt = (
        "<image1><image2>你是TikTok的专业广告图片分析师。请对比评估提供的两张广告图片，预测它们的投放效果并提供详细原因。\n"
        f"图片标题: {title}\n"
        "分析要求包括但不限于：\n"
        "1. 视觉方面：图片清晰度、主体突出度、色彩吸引力（是否符合TikTok视觉风格）\n"
        "2. 标题关键词与图片内容的相关性\n"
        "3. 购买欲望：整体激发用户购买意愿的能力\n"
        "请为每张图片分别提供1到100的CTR表现评分（1表示很差，100表示优秀）。\n"
        "重要要求：两张图片的评分必须有明显区分度，不能相同。请先分析第一张图片，再分析第二张图片。\n"
        "思考过程必须包含在标签内。最终答案请分别用\\boxed{}包裹，按图片顺序排列。"
    )
    # 处理所有图片输入
    image_inputs = []
    for path in image_paths:
        if path.startswith(("http://", "https://")):
            image_inputs.append(path)
        else:
            image_base64 = encode_image_to_base64(path)
            image_inputs.append(f"data:image/jpeg;base64,{image_base64}")
    request_body = {
        "image": image_inputs,  # 修改为复数形式接收多张图片
        "prompt": prompt
    }
    return client.AiModel(request_body = json.dumps(request_body))


def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为base64格式"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        print(f"图片编码失败: {e}")
        return ""


def extract_multiple_scores(response: str, count: int) -> List[int]:
    """从模型响应中提取多张图片的CTR预测分数"""
    try:
        # 尝试提取boxed格式的分数
        boxed_patterns = [
            r'boxed\s*\{\{(.*?)\}\}',
            r'\\boxed\s*\{\{(.*?)\}\}',
            r'boxed\s*\{(.*?)\}',
            r'\\boxed\s*\{(.*?)\}'
        ]
        all_matches = []
        for pattern in boxed_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            all_matches.extend([m.strip() for m in matches])
        # 从匹配结果中提取有效分数
        scores = []
        for match in all_matches:
            try:
                num = int(match)
                if 1 <= num <= 100:
                    scores.append(num)
            except:
                continue
        # 如果boxed提取不足，从文本中提取数字补充
        if len(scores) < count:
            numbers = re.findall(r'\b\d{1,3}\b', response)
            for num_str in numbers:
                try:
                    num = int(num_str)
                    if 1 <= num <= 100 and num not in scores:
                        scores.append(num)
                    if len(scores) == count:
                        break
                except:
                    continue
        # 确保分数数量正确且不重复
        while len(scores) < count:
            scores.append(50)  # 补充默认值
        # 确保所有分数不同
        if len(set(scores)) < count:
            for i in range(1, count):
                while scores[i] == scores[i-1]:
                    scores[i] = max(1, min(100, scores[i] + 1))
        return scores[:count]
    except Exception as e:
        print(f"提取分数时出错: {e}")
        return [50] * count


def send_request(item):
    ori_url = json.loads(item["ori_urls"])[0]
    gen_url = json.loads(item["gen_urls"])[0]
    code, msg, result = get_ctr_comparison([ori_url, gen_url], client)
    respone_json = json.loads(result.result_body)
    model_response = respone_json["model_response"]
    item["src_gen_score"] = extract_multiple_scores(model_response, 2)
    item["msg"] = model_response

    return item


if __name__ == "__main__":
    args = parse_args()
    print(f"input_file: {args.input_file}, num_workers: {args.num_workers}")

    data = load_csv_or_xlsx_to_dict(args.input_file)
    url_data = []
    for line in data:
        gen_urls = json.loads(line["gen_urls"])
        if len(gen_urls) == 0:
            continue

        url_data.append(line)

    results = []
    error_results = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_url = {executor.submit(send_request, item): item for item in url_data}
        with tqdm(total=len(url_data)) as pbar:
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result_json = future.result()
                    pbar.update(1)  # Update progress bar

                    results.append(result_json)
                    if len(results) % 10 == 0:
                        json_save(results, args.output_file)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    error_results.append({"image_url": url, "error_reason": str(e)})

    json_save(results, args.output_file)
    print(error_results)
