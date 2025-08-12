# @author: wilson.xu.
import argparse
from tqdm import tqdm
import json
import concurrent
from concurrent.futures import ThreadPoolExecutor

from diffusers.data.byted.clients.azure_mllm import MLLMClient
from diffusers.data.utils import load_file, json_save


gpt_client = MLLMClient(model_name="gemini-2.5-flash", api_key="3H7OHTGn3JwHvlZHP9JUJzb850gp3TGR_GPT_AK")


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


search_text_check_prompt = """You are a professional poster selling point verification assistant. Your task is to check if the specified selling point has been correctly rendered in the AI-generated poster, by comparing it with the original poster.
1.  **Input**:
    -   First image: Original poster (no specified selling point).
    -   Second image: AI-generated poster (with the selling point added).
    -   Target selling point: {selling_point}
2.  **Verification Rules**:
    -   **Only focus on verifying the target selling point {selling_point}**. Ignore all other text, watermarks, and existing selling points in both the original and generated posters.
    -   Check if the target selling point {selling_point} **exists** in the generated poster.
    -   Check if the target selling point {selling_point} is **exactly the same** as the text in the generated poster (case-sensitive, spelling-sensitive, and punctuation-sensitive).
    -   If the selling point is missing, misspelled, or altered in any way, it is considered a rendering error.
3.  **Output**:
    -   Output **only a valid JSON object** with exactly two fields:
        -   "result": A boolean value (`true` if the selling point is correctly rendered, `false` if there is a rendering error).
        -   "reason": A string that clearly describes the verification result.
            -   If `true`: "The selling point '{selling_point}' is correctly rendered in the generated poster."
            -   If `false`: "The selling point '{selling_point}' was not found or was rendered incorrectly in the generated poster. Found: '[actual_text_found]'."
Focus on **strict textual fidelity**—no creative interpretation, only direct comparison of the visible text content. Do not include any explanations, markdown, or extra text outside the JSON object.
"""


ocr_check_prompt = """You are a professional poster text verification assistant. Your task is to compare the text content between the original e-commerce poster (first image) and the AI-generated poster (second image) to check for any discrepancies, omissions, or errors.
1.  **Extract all text** from the original poster, including product names, descriptions, features, volume, hashtags, and brand information. **Exclude any watermarks or watermark text from extraction.**
2.  **Extract all text** from the generated poster in the same way. **Exclude any watermarks or watermark text from extraction.**
3.  **Compare the two sets of text** line by line, and identify:
    -   Any text that is **missing** in the generated poster.
    -   Any text that is **incorrectly changed** in the generated poster.
    -   Any **additional text** that appears in the generated poster but not in the original.
4.  **Output only a valid JSON object** with exactly two fields:
    -   "result": A boolean value (`true` if all text in the generated poster exactly matches the original, `false` if any discrepancies are found).
    -   "reason": A string that clearly describes the specific differences found. If the result is `true`, this field should state "All text in the generated poster matches the original poster exactly." If the result is `false`, list all discrepancies, such as "The text 'Lem Khusus Sepatu' in the original was changed to 'Lem N50saus Sepatu' in the generated poster."
Focus on **strict textual fidelity**—no creative interpretation, only direct comparison of the visible text content. Do not include any explanations, markdown, or extra text outside the JSON object.
"""


def send_request(item):
    src_url = item[0]
    gen_url = item[-1][0]
    result_check_id_json = gpt_client.make_image_json_request(
        "", ocr_check_prompt, [], [src_url, gen_url], max_tokens=4000, timeout=60)
    item.append(result_check_id_json)

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
    print(len(error_results))


if __name__ == "__main__":
    args = parse_args()
    print(f"input_file: {args.input_file}, num_workers: {args.num_workers}")

    data = load_file(args.input_file)
    main(data, args.output_file, args.num_workers)

    print('Done!')
