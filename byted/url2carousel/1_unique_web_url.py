# Copyright (c) wilson.xu. All rights reserved.
from tqdm import tqdm
from diffusers.data.utils import load_csv_or_xlsx_to_dict, csv_save, is_english

original_data_file = "url_data_pro.csv"
original_data = load_csv_or_xlsx_to_dict(f"/Users/bytedance/Downloads/{original_data_file}")
print(len(original_data))
unique_out = {line['External Website URL']: line for line in original_data}

input_data_file = "0731_processed.csv"
data = load_csv_or_xlsx_to_dict(f"/Users/bytedance/Downloads/{input_data_file}")

for line in tqdm(data):
    if line['External Website URL'] not in unique_out:
        original_data.append(line)

print(len(original_data))
csv_save(original_data, f"/Users/bytedance/Downloads/{original_data_file}")

print('done')
