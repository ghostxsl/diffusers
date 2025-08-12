# Copyright (c) wilson.xu. All rights reserved.
import argparse
from tqdm import tqdm
from diffusers.data.utils import load_csv_or_xlsx_to_dict, csv_save, is_english

# https://aeolus-va.tiktok-row.net/pages/dataQuery?appId=555116&id=1941253447&sid=396381
# 1. 按照这个url拉取对应的数据csv, 注意只能是csv格式，其他格式会存在有些属性乱码
# https://aeolus-va.tiktok-row.net/pages/dataQuery?appId=555116&id=1941332838&sid=657149
# 2. 根据第一步中拉取到的Ad ID, 拉取对应的title


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input",
        default="0731_title", # title
        type=str,
        help="Path to image list on vos.")

    args = parser.parse_args()
    return args


def main(args):
    date_, tag = args.input.split('_')

    if tag == "info":
        data = load_csv_or_xlsx_to_dict(f"/Users/bytedance/Downloads/{date_}_info.csv")
        # url去重
        unique_out = {line['External Website URL']: line for line in data}
        unique_out.pop('(空字符串)')
        out = list(unique_out.values())

        print(len(out))
        csv_save(out, f"/Users/bytedance/Downloads/{date_}_unique.csv")
    elif tag == "title":
        data = load_csv_or_xlsx_to_dict(f"/Users/bytedance/Downloads/{date_}_unique.csv")
        title = load_csv_or_xlsx_to_dict(f"/Users/bytedance/Downloads/{date_}_title.csv")
        # 根据 ad_id 去重, 仅提取英文title
        title = {
            a['TTAM Ad ID (creative_id)']: a['Creative Title'] for a in title if is_english(a['Creative Title'])}

        # 合并ad title
        out = []
        for line in tqdm(data):
            if line['Ad ID'] in title:
                line['ad_title'] = title[line['Ad ID']]
                out.append(line)

        print(len(out))
        csv_save(out, f"/Users/bytedance/Downloads/{date_}_processed.csv")
    else:
        raise Exception(f"Error input: {args.input}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
