'''
@author: alan02.xiao
@date: 4/9/2024
@descriptor: infering result using api
'''
# @author: wilson.xu.

import os
import requests
import argparse
import time
import json
import pickle
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="infer script.")
    parser.add_argument(
        "--vos_pkl",
        default=None,
        type=str,
        help="Path to image list on vos.")
    parser.add_argument(
        "--out_csv",
        default=None,
        type=str,
        help="File name to save.")
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int)

    args = parser.parse_args()
    assert args.vos_pkl is not None and os.path.exists(args.vos_pkl)
    assert args.out_csv is not None

    return args


def load_file(file_path):
    assert os.path.exists(file_path), f"File {file_path} does not exist."

    if os.path.splitext(file_path)[-1] == ".json":
        with open(file_path, 'r') as f:
            out = json.load(f)
        return out
    elif os.path.splitext(file_path)[-1] == ".csv":
        return pd.read_csv(file_path).values.tolist()
    else:
        with open(file_path, 'rb') as f:
            out = pickle.load(f)
        return out


def send_request(image_url, retry=3):
    url = "http://llava-vip-infer-serving.ai.vip.com/llava_vip"
    headers = {'Content-Type': 'application/json'}
    if prompt_dict is not None:
        req_dict = {'image_url': image_url, 'prompt_dict': prompt_dict}
    else:
        req_dict = {'image_url': image_url}

    for _ in range(retry):
        try:
            resp = requests.post(url, data=json.dumps(req_dict), headers=headers, timeout=100)
            if int(resp.status_code) == 200:
                return resp.text, resp.status_code
            elif int(resp.status_code) == 504:
                time.sleep(3)
            return resp.text, resp.status_code
        except Exception as e:
            #__import__('ipdb').set_trace()
            return e, resp.status_code


def write_to_csv(results, dst):
    df = pd.DataFrame(results)
    df.to_csv(dst, mode='a', index=False, header=not os.path.exists(dst))


def main(image_urls, dst, max_workers):
    results = []
    error_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(send_request, url): url for url in image_urls}
        with tqdm(total=len(image_urls)) as pbar:
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data, status = future.result()
                    pbar.update(1)  # Update progress bar
                    if int(status) == 200:
                        result = {'image_url': url, 'result': data}
                        results.append(result)
                    elif int(status) == 504:
                        result = {'image_url': url, 'error_reason': 'timeout'}
                        error_results.append(result)
                    else:
                        error_results.append({'image_url': url, 'error_reason': data})

                    if len(results) >= 100:
                        write_to_csv(results, dst)
                        results = []

                except Exception as e:
                    print(f"An error occurred for {url}: {e}")
                    error_results.append({'image_url': url, 'error_reason': str(e)})

        # Write remaining results
        if results:
            write_to_csv(results, dst)
        if error_results:
            write_to_csv(error_results, dst.replace('.csv', '_error.csv'))


if __name__ == "__main__":
    args = parse_args()
    # prompt_dict = {"整图描述": "请用一句话简要描述以下图片，用英文回答"}
    prompt_dict = {"整图描述": "Please briefly describe the following image in one sentence."}

    file_name = args.vos_pkl
    dst = args.out_csv
    data_pkl = load_file(file_name)

    image_urls = []
    if isinstance(data_pkl, list):
        if isinstance(data_pkl[0], str):
            image_urls = data_pkl
        else:
            image_urls = [a['image'] for a in data_pkl]
    elif isinstance(data_pkl, dict):
        for k, v in data_pkl.items():
            image_urls.extend([a['image'] for a in v])
    else:
        raise Exception(f"Error dataset_file: ({type(data_pkl)}){file_name}")

    print(file_name, len(image_urls))
    # 获得推理结果，成功推理结果保存至dst中，失败的推理结果路径保存在了dst.replace('.csv', '_error.csv')中
    main(image_urls, dst, args.num_workers)

    print('Done!')
