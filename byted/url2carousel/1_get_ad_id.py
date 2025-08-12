import os
from tqdm import tqdm
from diffusers.data.utils import load_csv_or_xlsx_to_dict, csv_save, xlsx_save


def is_english(input_text):
    """
    判断输入的文本是否为英文（忽略所有标点符号、数字及特定特殊字符）

    参数:
        input_text: 待检查的文本字符串

    返回:
        True: 如果文本是英文
        False: 如果文本包含非英文字符
    """
    # 定义需要忽略的特殊字符
    ignored_special_chars = {'°', '®', '™'}

    # 检查字符串中的每个字符
    for char in input_text:
        # 检查是否是英文字母（大小写）- 允许
        if 'A' <= char <= 'Z' or 'a' <= char <= 'z':
            continue
        # 检查是否是数字（0-9）- 忽略
        if '0' <= char <= '9':
            continue
        # 检查是否是需要忽略的特殊字符
        if char in ignored_special_chars:
            continue
        # 检查是否是°C组合中的C（已在字母检查中处理）
        # 所有其他字符都视为需要检查的字符
        # 如果字符不是英文字母、数字或忽略的特殊字符
        # 且不在基本拉丁符号范围内（即可能是非英文字符）
        if not (0x0020 <= ord(char) <= 0x007E):
            return False
    # 所有字符都检查通过
    return True


tag = "0901"
# 处理线上数据
data = load_csv_or_xlsx_to_dict(f"/Users/bytedance/Downloads/{tag}_info.csv")
# out = [str(a['Ad ID']) for a in data]
# csv_save(out, "/Users/bytedance/Downloads/0809_ad_id.csv")


temp_title = load_csv_or_xlsx_to_dict(f"/Users/bytedance/Downloads/{tag}_title.csv")
title = {}
for item in tqdm(temp_title):
    if '空字符串' in item['Creative Title']:
        continue
    if item['TTAM Ad ID (creative_id)'] in title:
        title[item['TTAM Ad ID (creative_id)']].append(item['Creative Title'])
    else:
        title[item['TTAM Ad ID (creative_id)']] = [item['Creative Title']]
title = {k: v[0] for k, v in title.items() if len(v) == 1 and is_english(v[0])}

out = []
for line in tqdm(data):
    if line['Ad ID'] in title:
        line['ad_title'] = title[line['Ad ID']]
        out.append(line)

unique_out = {line['Ad ID']: line for line in out}
out = list(unique_out.values())
csv_save(out, f"/Users/bytedance/Downloads/{tag}_processed.csv")

print('done')
