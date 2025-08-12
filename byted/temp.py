# import os
# from os.path import join, splitext, basename
# from tqdm import tqdm
# import csv
# import time
# import json
# import numpy as np
# from PIL import Image
#
# from diffusers.data.utils import load_file, csv_save, load_csv_or_xlsx_to_dict, xlsx_save, json_save
# from diffusers.data.outer_vos_tools import encode_pil_bytes, decode_pil_bytes, load_or_download_image
# from diffusers.data.byted.tos import url_to_bytes, save_file_to_tos, save_tos, _gen_name



#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 设置中文字体（避免中文乱码，如不需要可注释）
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 英文无乱码，如需中文替换为 ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 1. 精准提取你提供的最新数据（数量+CTR）
# filter_conditions = [
#     'All Data',
#     'Click > 0',
#     'Show > 10',
#     'Show > 20',
#     'Show > 50',
#     'Show > 100',
#     'Show > 200',
#     'Show > 500'
# ]
# # 数量（从你的数据中提取的精准值）
# quantity = [427263, 117132, 97824, 77958, 46374, 26289, 13267, 4766]
# # 原始组CTR (%)（匹配你的表1）
# origin_ctr = [2.2877, 2.6049, 2.4684, 2.3169, 2.0422, 1.8136, 1.6003, 1.3478]
# # 生成组CTR (%)（匹配你的表1）
# generate_ctr = [2.3787, 2.3339, 2.2850, 2.2108, 2.0270, 1.8374, 1.6330, 1.3817]
# # 相对CTR变化 (%)（匹配你的表1）
# relative_ctr = [3.98, -10.40, -7.43, -4.58, -0.74, 1.31, 2.04, 2.52]
#
# # 2. 设置图表布局（上下子图，共享x轴）
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
#                                gridspec_kw={'height_ratios': [3, 1]})
# fig.suptitle('Overall Distribution - CTR, Relative Trend and Quantity Under Different Filter Conditions',
#              fontsize=14, fontweight='bold')
#
# # 3. 绘制上层图表（CTR柱状图 + 相对CTR折线图）
# x = np.arange(len(filter_conditions))
# width = 0.35  # 柱状图宽度
#
# # 绘制原始/生成CTR柱状图
# bars1 = ax1.bar(x - width/2, origin_ctr, width, label='origin CTR', color='#1f77b4')
# bars2 = ax1.bar(x + width/2, generate_ctr, width, label='generate CTR', color='#ff7f0e')
#
# # 绘制相对CTR折线图（双Y轴）
# ax1_twin = ax1.twinx()
# line = ax1_twin.plot(x, relative_ctr, color='#d62728', marker='o', linewidth=2,
#                      markersize=6, label='Relative CTR')
#
# # ========== 标注：保留两位小数，位置优化 ==========
# # 原始CTR标注
# for bar in bars1:
#     height = bar.get_height()
#     ax1.text(bar.get_x() + bar.get_width()/2., height + 0.03,
#              f'{height:.2f}%', ha='center', va='bottom', fontsize=8, fontweight='medium')
#
# # 生成CTR标注
# for bar in bars2:
#     height = bar.get_height()
#     ax1.text(bar.get_x() + bar.get_width()/2., height + 0.03,
#              f'{height:.2f}%', ha='center', va='bottom', fontsize=8, fontweight='medium')
#
# # 相对CTR标注（偏移优化避免重叠）
# offset = 0.4
# for i, val in enumerate(relative_ctr):
#     y_pos = val + offset if val >= 0 else val - offset
#     va_align = 'bottom' if val >= 0 else 'top'
#     ax1_twin.text(i, y_pos,
#                   f'{val:.2f}%', ha='center', va=va_align,
#                   color='#d62728', fontsize=8, fontweight='medium')
#
# # 上层图表样式
# ax1.set_ylabel('Click-Through Rate (%)', fontsize=12)
# ax1_twin.set_ylabel('Relative CTR (%)', fontsize=12, color='#d62728')
# ax1_twin.tick_params(axis='y', labelcolor='#d62728')
# ax1.set_ylim(0, 3)  # 适配CTR数值范围
# ax1_twin.set_ylim(-12, 6)  # 适配相对CTR范围
# ax1.grid(axis='y', alpha=0.3, linestyle='--')
#
# # 合并图例
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax1_twin.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)
#
# # 4. 绘制下层数量柱状图（精准匹配你的数量数据）
# bars3 = ax2.bar(x, quantity, color='#2ca02c', width=0.8)
# ax2.set_ylabel('Quantity', fontsize=12)
# ax2.set_xlabel('Filter Conditions', fontsize=12)
# ax2.set_ylim(0, max(quantity) * 1.1)  # 自适应数量范围
# ax2.grid(axis='y', alpha=0.3, linestyle='--')
#
# # 数量标注
# for bar in bars3:
#     height = bar.get_height()
#     ax2.text(bar.get_x() + bar.get_width()/2., height + max(quantity)*0.02,
#              f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='medium')
#
# # 5. 设置X轴（旋转避免文字拥挤）
# ax2.set_xticks(x)
# ax2.set_xticklabels(filter_conditions, rotation=15, ha='right', fontsize=9)
#
# # 6. 保存图片（无.show()，高清PNG）
# plt.tight_layout()
# plt.savefig('overall_distribution_ctr.png', dpi=300, bbox_inches='tight')
#
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 设置字体（与样例一致，英文无乱码）
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 1. 提取新数据（严格对应你的全量分布，字段与样例一一匹配）
# filter_conditions = [
#     'All Data',
#     'Click > 0',
#     'Show > 10',
#     'Show > 20',
#     'Show > 50',
#     'Show > 100',
#     'Show > 200',
#     'Show > 500'
# ]
# # 数量（各筛选条件下的第一条数值）
# quantity = [498087, 187703, 168518, 143650, 95856, 59887, 32529, 12259]
# # origin组CTR（各筛选条件下的第四个百分比数值）
# origin_ctr = [2.2812, 2.4713, 2.4119, 2.3299, 2.1508, 1.9731, 1.7779, 1.5247]
# # generate组CTR（各筛选条件下generate的第四个百分比数值）
# generate_ctr = [2.3655, 2.3425, 2.3209, 2.2795, 2.1559, 2.0035, 1.8172, 1.5580]
# # 相对CTR变化（各筛选条件下括号内的变化率数值）
# relative_ctr = [3.7, -5.21, -3.78, -2.16, 0.24, 1.54, 2.21, 2.19]
#
# # 2. 图表布局（与样例完全一致：上下子图、共享x轴、高度比3:1）
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
#                                gridspec_kw={'height_ratios': [3, 1]})
# fig.suptitle('Overall Distribution - CTR, Relative Trend and Quantity',
#              fontsize=14, fontweight='bold')
#
# # 3. 上层子图：CTR柱状图 + 相对CTR折线图（双Y轴，样式与样例完全一致）
# x = np.arange(len(filter_conditions))
# width = 0.35  # 柱状图宽度与样例一致
#
# # 绘制origin/generate CTR柱状图（配色与样例完全一致）
# bars1 = ax1.bar(x - width/2, origin_ctr, width, label='origin CTR', color='#1f77b4')
# bars2 = ax1.bar(x + width/2, generate_ctr, width, label='generate CTR', color='#ff7f0e')
#
# # 绘制相对CTR折线图（红色+圆点，样式与样例完全一致）
# ax1_twin = ax1.twinx()
# line = ax1_twin.plot(x, relative_ctr, color='#d62728', marker='o', linewidth=2,
#                      markersize=6, label='Relative CTR')
#
# # ========== 数值标注（与样例规则一致：保留两位小数、正负偏移避免重叠） ==========
# # origin CTR标注
# for bar in bars1:
#     height = bar.get_height()
#     ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
#              f'{height:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='medium')
#
# # generate CTR标注
# for bar in bars2:
#     height = bar.get_height()
#     ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
#              f'{height:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='medium')
#
# # 相对CTR折线标注（正负偏移优化，与样例逻辑一致）
# offset = 0.2  # 适配新数据范围的偏移量
# for i, val in enumerate(relative_ctr):
#     y_pos = val + offset if val >= 0 else val - offset
#     va_align = 'bottom' if val >= 0 else 'top'
#     ax1_twin.text(i, y_pos,
#                   f'{val:.2f}%', ha='center', va=va_align,
#                   color='#d62728', fontsize=9, fontweight='medium')
#
# # 上层样式设置（与样例一致，仅调整Y轴范围适配新数据）
# ax1.set_ylabel('Click-Through Rate (%)', fontsize=12)
# ax1_twin.set_ylabel('Relative CTR (%)', fontsize=12, color='#d62728')
# ax1_twin.tick_params(axis='y', labelcolor='#d62728')
# ax1.set_ylim(1.4, 2.6)  # 适配新数据的CTR范围
# ax1_twin.set_ylim(-6.0, 4.5)  # 适配新数据的相对变化率范围
# ax1.grid(axis='y', alpha=0.3, linestyle='--')
#
# # 合并图例（位置、样式与样例一致）
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax1_twin.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)
#
# # 4. 下层子图：数量柱状图（绿色，样式与样例完全一致）
# bars3 = ax2.bar(x, quantity, color='#2ca02c', width=0.8)
# ax2.set_ylabel('Quantity', fontsize=12)
# ax2.set_xlabel('Filter Conditions', fontsize=12)
# ax2.set_ylim(0, max(quantity) * 1.1)  # 自适应数量范围
# ax2.grid(axis='y', alpha=0.3, linestyle='--')
#
# # 数量标注（整数，位置与样例一致）
# for bar in bars3:
#     height = bar.get_height()
#     ax2.text(bar.get_x() + bar.get_width()/2., height + max(quantity)*0.02,
#              f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='medium')
#
# # 5. X轴设置（换行显示长条件，避免重叠）
# ax2.set_xticks(x)
# ax2.set_xticklabels(filter_conditions, rotation=0, ha='center', fontsize=10)
#
# # 6. 保存高清图片（与样例一致：300DPI、tight布局，无plt.show()）
# plt.tight_layout()
# plt.savefig('overall_distribution_ctr.png', dpi=300, bbox_inches='tight')


import matplotlib.pyplot as plt
import numpy as np

# 固定字体设置（避免乱码）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 新版曝光量分桶数据（精准提取自你的最新数据）
filter_conditions = [
    '0 ≤ show < 10',
    '10 ≤ show < 30',
    '30 ≤ show < 80',
    '80 ≤ show < 200',
    'show ≥ 200'
]
# 各分桶下的样本量（origin num）+ 占比
quantity = [209100, 118678, 86118, 48696, 36158]
quantity_ratio = ['41.92%', '23.80%', '17.27%', '9.76%', '7.25%']

# CTR 数据
origin_ctr = [3.4562, 3.8198, 3.6346, 3.1661, 1.7890]
generate_ctr = [4.3951, 4.1103, 3.7233, 3.2242, 1.8260]
relative_ctr = [27.17, 7.61, 2.44, 1.84, 2.07]

# CVR 数据
origin_cvr = [2.4821, 2.6200, 2.6353, 2.6598, 3.0105]
generate_cvr = [2.3607, 2.4327, 2.5348, 2.6740, 2.9506]
relative_cvr = [-4.89, -7.15, -3.81, 0.53, -1.99]

# CTCVR 数据
origin_ctcvr = [0.0858, 0.1001, 0.0958, 0.0842, 0.0539]
generate_ctcvr = [0.1038, 0.1000, 0.0944, 0.0862, 0.0539]
relative_ctcvr = [20.98, -0.10, -1.46, 2.38, 0.00]

# GMV 数据（单位：十亿，便于展示）
origin_gmv = [6.47, 17.46, 33.35, 36.36, 100.97]
generate_gmv = [9.89, 18.62, 31.18, 38.13, 98.32]
relative_gmv = [52.90, 6.64, -6.50, 4.88, -2.62]

# 2. 循环生成四个指标的图表
metrics = [
    ("CTR", origin_ctr, generate_ctr, relative_ctr, 1.6, 4.6, -2, 30),
    ("CVR", origin_cvr, generate_cvr, relative_cvr, 2.2, 3.2, -9, 2),
    ("CTCVR", origin_ctcvr, generate_ctcvr, relative_ctcvr, 0.05, 0.11, -3, 23),
    ("GMV (Billion)", origin_gmv, generate_gmv, relative_gmv, 4, 105, -8, 55)
]

for metric_name, origin_data, generate_data, relative_data, y_min, y_max, rel_min, rel_max in metrics:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'Overall Distribution - {metric_name}, Relative Trend and Quantity',
                 fontsize=14, fontweight='bold')

    x = np.arange(len(filter_conditions))
    width = 0.35

    # 上层子图：柱状图 + 折线图
    bars1 = ax1.bar(x - width/2, origin_data, width, label=f'origin {metric_name}', color='#1f77b4')
    bars2 = ax1.bar(x + width/2, generate_data, width, label=f'generate {metric_name}', color='#ff7f0e')

    ax1_twin = ax1.twinx()
    line = ax1_twin.plot(x, relative_data, color='#d62728', marker='o', linewidth=2,
                         markersize=6, label=f'Relative {metric_name}')

    # 数值标注
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (y_max - y_min)*0.01,
                 f'{height:.4f}' if metric_name == "CTCVR" else f'{height:.2f}',
                 ha='center', va='bottom', fontsize=9, fontweight='medium')
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (y_max - y_min)*0.01,
                 f'{height:.4f}' if metric_name == "CTCVR" else f'{height:.2f}',
                 ha='center', va='bottom', fontsize=9, fontweight='medium')

    for i, val in enumerate(relative_data):
        offset = 0.8 if val >= 0 else -0.8  # 适配大数值偏移
        ax1_twin.text(i, val + offset,
                      f'{val:.2f}%', ha='center', va='bottom' if val >= 0 else 'top',
                      color='#d62728', fontsize=9, fontweight='medium')

    # 上层样式设置
    ax1.set_ylabel(f'{metric_name} (%)' if metric_name != "GMV (Billion)" else f'{metric_name}', fontsize=12)
    ax1_twin.set_ylabel(f'Relative {metric_name} (%)', fontsize=12, color='#d62728')
    ax1_twin.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_ylim(y_min, y_max)
    ax1_twin.set_ylim(rel_min, rel_max)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)

    # 下层子图：数量柱状图（显示数量+占比）
    bars3 = ax2.bar(x, quantity, color='#2ca02c', width=0.8)
    ax2.set_ylabel('Quantity', fontsize=12)
    ax2.set_ylim(0, max(quantity) * 1.15)  # 调整Y轴范围，避免标注超出
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # 数量+占比标注
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        label_text = f'{int(height)} ({quantity_ratio[i]})'
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(quantity)*0.02,
                 label_text, ha='center', va='bottom', fontsize=9, fontweight='medium')

    # X轴设置
    ax2.set_xticks(x)
    ax2.set_xticklabels(filter_conditions, rotation=0, ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'exposure_binning_new_{metric_name.lower()}_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

# 额外生成新版分桶的数量占比饼图
fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(quantity, labels=filter_conditions, autopct='%1.2f%%',
                                  startangle=90, colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
ax.set_title('Exposure Bins (New) - Quantity Distribution', fontsize=14, fontweight='bold')

# 美化饼图标注
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

plt.tight_layout()
plt.savefig('exposure_binning_new_quantity_pie.png', dpi=300, bbox_inches='tight')
plt.close()



# # outpainting
# from diffusers.data.clients.toutiao_labcv_algo_vproxy import img2img_outpainting
# token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
# os.environ["SEC_TOKEN_STRING"] = token
# if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
#     token = os.popen("cat /tmp/identity.token").read()
#     os.environ["SEC_TOKEN_STRING"] = token
# os.environ["SEC_KV_AUTH"] = "1"
# os.environ["TCE_PSM"] = "ad.creative.image_core_solution"
#
#
# def pad_image(image, padding, fill_color=(0, 0, 0)):
#     """
#     为图像添加填充
#
#     参数:
#         image: PIL Image对象
#         padding: 填充大小，可以是一个整数(四周填充相同大小)或元组(左,右,上,下)
#         fill_color: 填充区域的颜色，默认为白色(255,255,255)
#
#     返回:
#         填充后的PIL Image对象
#     """
#     # 处理填充参数
#     if isinstance(padding, int):
#         left = right = top = bottom = padding
#     else:
#         left, right, top, bottom = padding
#
#     # 获取原图尺寸
#     width, height = image.size
#
#     # 计算新图像尺寸
#     new_width = width + left + right
#     new_height = height + top + bottom
#
#     # 创建新图像，使用指定颜色填充
#     new_image = Image.new(image.mode, (new_width, new_height), fill_color)
#
#     # 将原图粘贴到新图像的指定位置
#     new_image.paste(image, (left, top))
#
#     return new_image
#
#
# row_dict = load_csv_or_xlsx_to_dict("/mnt/bn/ttcc-algo-bytenas/zjn/data/poster_train/results_all_done_4567.csv")
#
# width, height = 720, 1280
# for line in tqdm(row_dict):
#     img = load_or_download_image(line['gen_url'])
#     w, h = img.size
#     oh = int(h * width / w)
#     img = img.resize((width, oh), 1)
#     pad_ = (height - oh) // 2
#     img = pad_image(img, (0, 0, pad_, pad_))
#
#     mask = Image.new("L", (width, oh), 0)
#     mask = pad_image(mask, (0, 0, pad_, pad_), 255)
#
#     result = img2img_outpainting(
#         image=encode_pil_bytes(img, False),
#         mask=encode_pil_bytes(mask, False),
#         max_height=height,
#         max_width=width,
#         custom_prompt=line['gen_bg_prompt']
#     )
#     out_img = decode_pil_bytes(result, False)
#
# # xlsx_save(row_dict, "results_all_done_4567_v1.xlsx")
# print('done')


#
# # 主体分割
# from diffusers.data.byted.clients.creative_ai_capability import image_subject_seg
# token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
# os.environ["SEC_TOKEN_STRING"] = token  # os.environ['SEC_TOKEN_STRING'] = "toutiao.growth.xenon"
# if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
#     token = os.popen("cat /tmp/identity.token").read()
#     os.environ["SEC_TOKEN_STRING"] = token
# os.environ["SEC_KV_AUTH"] = "1"
#
# row_dict = load_csv_or_xlsx_to_dict("/mnt/bn/ttcc-algo-bytenas/zjn/data/poster_train/results_all_done_4567.csv")
#
# for line in tqdm(row_dict):
#     url = line['gen_url']
#     result = image_subject_seg(image_urls=[url], only_mask=1, refine_mask=0)
#     line['gen_mask_url'] = result.success_image_urls[0]
#
# xlsx_save(row_dict, "results_all_done_4567_v1.xlsx")
# print('done')




#
# # 上传图像到tos
# row_dict = load_csv_or_xlsx_to_dict("/mnt/bn/creative-algo/xsl/data/gpt_dataset/test_src.xlsx")
#
# test_dir = "qwen_lora_test"
# for line in tqdm(row_dict):
#     name = basename(line['gen_url'])
#     img_url = save_file_to_tos(join(test_dir, name + "_lora.jpg"), _gen_name('xsl'))
#     line['infer_url'] = img_url
#
# xlsx_save(row_dict, "test_src_v1.xlsx")
# print('done')




# # html渲染
# from diffusers.data.render_html.render_html import renderUrl2CarouselSingleHtml
# row_dict = load_csv_or_xlsx_to_dict("/mnt/bn/creative-algo/xsl/data/gpt_dataset/test_src.xlsx")
# test_dir = "/mlx_devbox/users/xushangliang/playground/gpt_lora_test"
#
# ind = 0
# html_result = []
# for item in tqdm(row_dict):
#     html_item = {
#         "index": ind,
#         "product_name": item["text_product_names"],
#         "primary_selling_points": item["text_primary_selling_points"],
#         "secondary_selling_points": item["text_secondary_selling_points"],
#         "prompt": item["gpt_correction_prompt"],
#     }
#
#     upload_name = _gen_name("xsl")
#     for i in range(5):
#         img_url = save_file_to_tos(join(test_dir, basename(item['gen_url']) + f'_lora.jpg'), upload_name)
#         if img_url is not None:
#             break
#         else:
#             time.sleep(1)
#
#     html_item["ImageDiff"] = [
#         {
#             "title": "product",
#             "imageUrl": item['product_url'],
#         },
#         {
#             "title": "GT",
#             "imageUrl": item['gen_url'],
#         },
#         {
#             "title": "qwen_lora",
#             "imageUrl": img_url,
#         },
#     ]
#     html_result.append(html_item)
#     ind += 1
# for i in range(3):
#     code, msg, url = renderUrl2CarouselSingleHtml("/t2d_demo", f"xsl_exp_{time.time()}.html", json.dumps(html_result))
#     if url is not None:
#         break
#     else:
#         time.sleep(1)
# print(url)
# print('done')





# # OCR
# from diffusers.data.clients.creative_ai_capability import image_subject_seg
# from diffusers.data.clients.creative_ai_capability import image_ocr_to_bbox
# from overpass_ad_creative_ai_capabilities.euler_gen.idl.i18n_ad.creative.creative_factory.strategy.capability_thrift import ImageInfo
# from diffusers.metrics.ocr_word_metrics import calculate_word_accuracy, calculate_wer
#
# token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
# os.environ["SEC_TOKEN_STRING"] = token  # os.environ['SEC_TOKEN_STRING'] = "toutiao.growth.xenon"
# if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
#     token = os.popen("cat /tmp/identity.token").read()
#     os.environ["SEC_TOKEN_STRING"] = token
# os.environ["SEC_KV_AUTH"] = "1"
#
#
# def get_image_text(url):
#     result = image_subject_seg(image_urls=[url], only_mask=1, refine_mask=0)
#     mask = load_or_download_image(result.success_image_urls[0])
#     img = load_or_download_image(url)
#     mask = np.array(mask.convert('L'))
#
#     mask = np.float32(mask[..., None]) / 255
#     bg_image = (1 - mask) * np.array(img, dtype=np.float32) + mask * 255
#     bg_image = Image.fromarray(np.uint8(np.clip(bg_image, 0, 255)))
#     image_info = ImageInfo(Binary=encode_pil_bytes(bg_image, False))
#
#     temp = []
#     for i in range(5):
#         if len(temp) > 0:
#             break
#         temp = image_ocr_to_bbox(image_infos=[image_info])[0]
#
#     pred_text = []
#     for item in temp:
#         pred_text.append(item['text'])
#     pred_text = " ".join(pred_text)
#     pred_text = pred_text.lower().replace(',', '').replace('，', '').replace('.', '')
#     return pred_text
#
# # 解析html中的data
# with open("xsl_exp_1755657489.3030906.html", "r", encoding="utf-8") as file:
#     html_content = file.read()
# html_content = html_content.split('var data = ')[1]
# html_content = html_content.split(';\n\n    /* ---------------- 兼容')[0]
# data = json.loads(html_content)
#
# avg_acc1 = []
# avg_wer1 = []
# avg_acc2 = []
# avg_wer2 = []
# for line in tqdm(data):
#     gt_text = get_image_text(line['ImageDiff'][1]['imageUrl'])
#     line['gt_text'] = gt_text
#     pred_text = get_image_text(line['ImageDiff'][2]['imageUrl'])
#     line['pred_text'] = pred_text
#     open_text = get_image_text(line['ImageDiff'][3]['imageUrl'])
#     line['open_text'] = open_text
#
#     if len(gt_text) > 0 and len(pred_text) > 0 and len(open_text) > 0:
#         avg_acc1.append(calculate_word_accuracy(pred_text, gt_text))
#         avg_wer1.append(calculate_wer(pred_text, gt_text))
#
#         avg_acc2.append(calculate_word_accuracy(open_text, gt_text))
#         avg_wer2.append(calculate_wer(open_text, gt_text))
#     else:
#         print(gt_text, pred_text, open_text)
#
# print(np.mean(avg_acc1), np.mean(avg_wer1))
# print(np.mean(avg_acc2), np.mean(avg_wer2))
# json_save(data, "ocr_results.json")
#
# print('done')
