# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import exists, join
import argparse
import math
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from diffusers.data.utils import load_file, csv_save
from diffusers.data.outer_vos_tools import load_or_download_image


def parse_args():
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument(
        "--input_file", default=None, type=str)
    parser.add_argument(
        "--output_file", default="output.csv", type=str)
    parser.add_argument(
        "--save_dir", default="output", type=str)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    return args


def plot_click_rate_bar(x_labels, y_rates, save_path):
    """
    绘制点击率条形图

    参数:
    x_labels: x轴标签列表（如['a', 'b', 'c', ...]）
    y_rates: 对应的点击率列表（如[0.1, 0.1, 0.2, 0.3, ...]）
    """
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    if len(x_labels) != len(y_rates):
        raise ValueError("x轴标签与y值数量必须一致")

    # 紧凑画布比例
    plt.figure(figsize=(8, 6))

    # 绘制深蓝色条形
    bars = plt.bar(x_labels, y_rates, color="steelblue", edgecolor="none")

    # 添加标签和标题
    plt.ylabel("CTR", fontsize=12)
    # 调整x轴标签角度（避免文字过长重叠）
    plt.xticks(rotation=45, ha="right", fontsize=10)  # rotation控制旋转角度，ha='right'右对齐

    # 添加y轴网格线，便于读数
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # 调整布局，避免标签被截断
    plt.tight_layout()
    # 显示图形
    plt.savefig(save_path)
    print(f"save figure to {save_path}")


def plot_click_rate_comparison(
        categories, original_rates, generated_rates, save_path,
        ori_key="origin", tar_key="generate",
        ori_color="#1f77b4", tar_color="#ff7f0e",
):
    """
    绘制原图和生成图点击率对比的条形图

    参数:
    categories: 商品类目列表
    original_rates: 原图点击率列表
    generated_rates: 生成图点击率列表
    """
    sorted_data = sorted(
        zip(categories, original_rates, generated_rates), key=lambda item: item[0]
    )  # 按第一个元素（类别）的首字母排序
    categories, original_rates, generated_rates = zip(*sorted_data)

    # 设置画布大小
    plt.figure(figsize=(16, 8))
    # 解决负号显示问题
    plt.rcParams["axes.unicode_minus"] = False

    # 设置条形图位置
    x = np.arange(len(categories))  # 类目数量
    width = 0.35  # 条形宽度

    # 绘制条形图
    bars1 = plt.bar(x - width / 2, original_rates, width, label=ori_key, color=ori_color, edgecolor="black")
    bars2 = plt.bar(x + width / 2, generated_rates, width, label=tar_key, color=tar_color, edgecolor="black")

    # 添加标签、标题和图例
    plt.ylabel("CTR", fontsize=12)
    plt.xticks(x, categories, rotation=45, ha="right", fontsize=10)  # 旋转类目名称，避免重叠
    plt.legend(fontsize=12)

    # 添加网格线，使数据更易读
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # 调整布局
    plt.tight_layout()
    # 显示图形
    plt.savefig(save_path)
    print(f"save figure to {save_path}")


def get_subject_ratios(data, show_key, click_key, ratio_key):
    stat = [{"show": 0, "click": 0} for _ in range(10)]
    ratios = []
    for line in data:
        subject_ratio = line[ratio_key]
        ratios.append(subject_ratio)
        if subject_ratio == 1.0:
            stat[-1]["show"] += line[show_key]
            stat[-1]["click"] += line[click_key]
        else:
            ind = int(subject_ratio * 10)
            stat[ind]["show"] += line[show_key]
            stat[ind]["click"] += line[click_key]

    ctrs = [a["click"] / (a["show"] + 1e-8) for a in stat]
    print(f"{ratio_key}: mean({np.mean(ratios)}), min({np.min(ratios)}), max({np.max(ratios)})")
    return ctrs


def draw_pos_and_neg_subject_ratio_hist(data, src_ctr_key, gen_ctr_key):
    pos_ctr_ratios = []
    neg_ctr_ratios = []
    for line in tqdm(data):
        src_ctr = line[src_ctr_key]
        gen_ctr = line[gen_ctr_key]
        diff = gen_ctr - src_ctr
        if not math.isnan(diff):
            if diff > 0:
                pos_ctr_ratios.append(line['gen_subject_ratio'])
            elif diff < 0:
                neg_ctr_ratios.append(line["gen_subject_ratio"])
    # 1. 正向ctr直方图
    plt.figure()
    weights = [1/len(pos_ctr_ratios)] * len(pos_ctr_ratios)
    plt.hist(pos_ctr_ratios, bins=np.linspace(0, 1, 11).tolist(), edgecolor='black', weights=weights)
    plt.title('positive')
    plt.xlabel('subject_ratio')
    plt.ylabel('count')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.savefig(join(args.save_dir, '1_hist_pos_subject_ratio.png'), dpi=150, bbox_inches='tight')

    # 2. 负向ctr直方图
    plt.figure()
    weights = [1/len(neg_ctr_ratios)] * len(neg_ctr_ratios)
    plt.hist(neg_ctr_ratios, bins=np.linspace(0, 1, 11).tolist(), edgecolor='black', weights=weights)
    plt.title('negative')
    plt.xlabel('subject_ratio')
    plt.ylabel('count')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.savefig(join(args.save_dir, '1_hist_neg_subject_ratio.png'), dpi=150, bbox_inches='tight')


def get_pos_and_neg_subject_ratio_ctrs(data, src_ctr_key, gen_ctr_key, gen_show_key, gen_click_key):
    pos_stat = [{"show": 0, "click": 0} for _ in range(10)]
    pos_ratios = []
    neg_stat = [{"show": 0, "click": 0} for _ in range(10)]
    neg_ratios = []
    for line in data:
        diff = line[gen_ctr_key] - line[src_ctr_key]
        subject_ratio = line["gen_subject_ratio"]
        if diff > 0:
            pos_ratios.append(subject_ratio)
            if subject_ratio == 1.0:
                pos_stat[-1]["show"] += line[gen_show_key]
                pos_stat[-1]["click"] += line[gen_click_key]
            else:
                ind = int(subject_ratio * 10)
                pos_stat[ind]["show"] += line[gen_show_key]
                pos_stat[ind]["click"] += line[gen_click_key]
        elif diff < 0:
            neg_ratios.append(subject_ratio)
            if subject_ratio == 1.0:
                neg_stat[-1]["show"] += line[gen_show_key]
                neg_stat[-1]["click"] += line[gen_click_key]
            else:
                ind = int(subject_ratio * 10)
                neg_stat[ind]["show"] += line[gen_show_key]
                neg_stat[ind]["click"] += line[gen_click_key]

    pos_ctrs = [a["click"] / (a["show"] + 1e-8) for a in pos_stat]
    print(f"正向CTR-主体占比: mean({np.mean(pos_ratios)}), min({np.min(pos_ratios)}), max({np.max(pos_ratios)})")
    neg_ctrs = [a["click"] / (a["show"] + 1e-8) for a in neg_stat]
    print(f"负向CTR-主体占比: mean({np.mean(neg_ratios)}), min({np.min(neg_ratios)}), max({np.max(neg_ratios)})")
    return pos_ctrs, neg_ctrs


def stat_subject_ratio(
        data,
        src_ctr="click_rate_74803023",
        gen_ctr="click_rate_74803024",
        src_show_key="show_74803023",
        src_click_key="click_74803023",
        gen_show_key="show_74803024",
        gen_click_key="click_74803024",
):
    x_labels = [
        "0.0-0.1",
        "0.1-0.2",
        "0.2-0.3",
        "0.3-0.4",
        "0.4-0.5",
        "0.5-0.6",
        "0.6-0.7",
        "0.7-0.8",
        "0.8-0.9",
        "0.9-1.0",
    ]
    # 1. 绘制原图vs生成图主体ratio-ctr条形图
    ori_ctrs = get_subject_ratios(data, src_show_key, src_click_key, "src_subject_ratio")
    gen_ctrs = get_subject_ratios(data, gen_show_key, gen_click_key, "gen_subject_ratio")
    plot_click_rate_comparison(x_labels, ori_ctrs, gen_ctrs, join(args.save_dir, "1_subject_ratio_ctrs.png"))

    # 2. 正向/负向点击率统计
    pos_ctrs, neg_ctrs = get_pos_and_neg_subject_ratio_ctrs(data, src_ctr, gen_ctr, gen_show_key, gen_click_key)
    plot_click_rate_bar(x_labels, pos_ctrs, join(args.save_dir, "1_pos_subject_ratio_ctrs.png"))
    plot_click_rate_bar(x_labels, neg_ctrs, join(args.save_dir, "1_neg_subject_ratio_ctrs.png"))

    # 3. 生成图:原图主体占比-ctr正向/负向直方图
    draw_pos_and_neg_subject_ratio_hist(data, src_ctr, gen_ctr)


def get_color_delta_e(data, show_key, click_key, color_key):
    stat = [{"show": 0, "click": 0} for _ in range(3)]
    delta_es = []
    for line in data:
        delta_e = line[color_key]
        delta_es.append(delta_e)
        if delta_e <= 2.0:
            ind = 0
        elif delta_e > 10.0:
            ind = 2
        else:
            ind = 1
        stat[ind]["show"] += line[show_key]
        stat[ind]["click"] += line[click_key]

    ctrs = [a["click"] / (a["show"] + 1e-8) for a in stat]
    print(f"{color_key}: mean({np.mean(delta_es)}), min({np.min(delta_es)}), max({np.max(delta_es)})")
    return ctrs


def draw_pos_and_neg_color(data, src_ctr_key, gen_ctr_key):
    pos_ctr = [[] for _ in range(3)]
    neg_ctr = [[] for _ in range(3)]
    for line in tqdm(data):
        src_ctr = line[src_ctr_key]
        gen_ctr = line[gen_ctr_key]
        diff = gen_ctr - src_ctr
        if not math.isnan(diff):
            if diff > 0:
                if line["gen_delta_e"] <= 2.0:
                    ind = 0
                elif line["gen_delta_e"] > 10.0:
                    ind = 2
                else:
                    ind = 1
                pos_ctr[ind].append(diff)
            elif diff < 0:
                if line["gen_delta_e"] <= 2.0:
                    ind = 0
                elif line["gen_delta_e"] > 10.0:
                    ind = 2
                else:
                    ind = 1
                neg_ctr[ind].append(diff)
    # 1. 正向ctr
    x_labels = ["Low", "Medium", "High"]
    ctrs = [np.mean(a) if len(a) > 0 else 0 for a in pos_ctr]

    plt.figure()
    plt.plot(
        range(len(x_labels)),
        ctrs,  # x和y数据
        marker="o",  # 数据点标记（可选，如圆圈、方块等）
        color="b",
        linestyle="-",
    )
    plt.xticks(
        range(len(x_labels)),  # 刻度位置（0-9）
        labels=x_labels,  # 替换为对应的字符串标签
        rotation=45,
    )
    plt.title("positive")
    plt.ylabel("CTR")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(join(args.save_dir, "2_pos_color_difference.png"))

    # 2. 负向ctr
    ctrs = [np.mean(a) if len(a) > 0 else 0 for a in neg_ctr]

    plt.figure()
    plt.plot(
        range(len(x_labels)),
        ctrs,  # x和y数据
        marker="o",  # 数据点标记（可选，如圆圈、方块等）
        color="b",
        linestyle="-",
    )
    plt.xticks(
        range(len(x_labels)),  # 刻度位置（0-9）
        labels=x_labels,  # 替换为对应的字符串标签
        rotation=45,
    )
    plt.title("negative")
    plt.ylabel("CTR")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(join(args.save_dir, "2_neg_color_difference.png"))


def stat_color_difference(
        data,
        src_ctr="click_rate_74803023",
        gen_ctr="click_rate_74803024",
        src_show_key="show_74803023",
        src_click_key="click_74803023",
        gen_show_key="show_74803024",
        gen_click_key="click_74803024",
):
    x_labels = ["Low", "Medium", "High"]
    # 1. 绘制原图vs生成图前背景对比度-ctr条形图
    ori_ctrs = get_color_delta_e(data, src_show_key, src_click_key, "src_delta_e")
    gen_ctrs = get_color_delta_e(data, gen_show_key, gen_click_key, "gen_delta_e")
    plot_click_rate_comparison(x_labels, ori_ctrs, gen_ctrs, join(args.save_dir, "2_color_difference_ctrs.png"))

    # 3. 生成图:原图对比度-ctr正向/负向折线图
    draw_pos_and_neg_color(data, src_ctr, gen_ctr)


def draw_word_count_to_ctr(data, ctr_key, word_key):
    sorted_data = sorted([a for a in data if not math.isnan(a[ctr_key])], key=lambda x: x[ctr_key])
    word_count = [len(a[word_key]) for a in sorted_data]
    ctrs = [a[ctr_key] for a in sorted_data]

    plt.figure()
    plt.scatter(
        word_count, ctrs,  # x和y数据
        marker="o",  # 数据点标记（可选，如圆圈、方块等）
        color="b",
        s=50,  # 点的大小（面积）
        alpha=0.7,
    )
    plt.title(f"{word_key}-{ctr_key}")
    plt.xlabel(f"{word_key}")
    plt.ylabel(f"{ctr_key}")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(join(args.save_dir, f"3_{word_key}-{ctr_key}.png"))


def draw_pos_and_neg_text_height_ratio_hist(data, src_ctr_key, gen_ctr_key):
    pos_ctr_ratios = []
    neg_ctr_ratios = []
    for line in tqdm(data):
        if line['gen_text_height_ratio'] == 0:
            continue
        src_ctr = line[src_ctr_key]
        gen_ctr = line[gen_ctr_key]
        diff = gen_ctr - src_ctr
        if not math.isnan(diff):
            if diff > 0:
                pos_ctr_ratios.append(line['gen_text_height_ratio'])
            elif diff < 0:
                neg_ctr_ratios.append(line["gen_text_height_ratio"])

    # 1. 正向ctr直方图
    plt.figure()
    weights = [1/len(pos_ctr_ratios)] * len(pos_ctr_ratios)
    plt.hist(pos_ctr_ratios, bins=np.linspace(0, 1, 11).tolist(), edgecolor='black', weights=weights)
    plt.title('positive')
    plt.xlabel('text_ratio')
    plt.ylabel('count')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.savefig(join(args.save_dir, '3_hist_pos_text_height_ratio.png'), dpi=150, bbox_inches='tight')

    # 2. 负向ctr直方图
    plt.figure()
    weights = [1/len(neg_ctr_ratios)] * len(neg_ctr_ratios)
    plt.hist(neg_ctr_ratios, bins=np.linspace(0, 1, 11).tolist(), edgecolor='black', weights=weights)
    plt.title('negative')
    plt.xlabel('text_ratio')
    plt.ylabel('count')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.savefig(join(args.save_dir, '3_hist_neg_text_height_ratio.png'), dpi=150, bbox_inches='tight')


def get_pos_and_neg_text_height_ratio_ctrs(data, src_ctr_key, gen_ctr_key, gen_show_key, gen_click_key):
    pos_stat = [{"show": 0, "click": 0} for _ in range(10)]
    pos_ratios = []
    neg_stat = [{"show": 0, "click": 0} for _ in range(10)]
    neg_ratios = []
    for line in data:
        if line['gen_text_height_ratio'] == 0:
            continue
        diff = line[gen_ctr_key] - line[src_ctr_key]
        text_ratio = line["gen_text_height_ratio"]
        if diff > 0:
            pos_ratios.append(text_ratio)
            if text_ratio == 1.0:
                pos_stat[-1]["show"] += line[gen_show_key]
                pos_stat[-1]["click"] += line[gen_click_key]
            else:
                ind = int(text_ratio * 10)
                pos_stat[ind]["show"] += line[gen_show_key]
                pos_stat[ind]["click"] += line[gen_click_key]
        elif diff < 0:
            neg_ratios.append(text_ratio)
            if text_ratio == 1.0:
                neg_stat[-1]["show"] += line[gen_show_key]
                neg_stat[-1]["click"] += line[gen_click_key]
            else:
                ind = int(text_ratio * 10)
                neg_stat[ind]["show"] += line[gen_show_key]
                neg_stat[ind]["click"] += line[gen_click_key]

    pos_ctrs = [a["click"] / (a["show"] + 1e-8) for a in pos_stat]
    print(f"正向CTR-文本行高占比: mean({np.mean(pos_ratios)}), min({np.min(pos_ratios)}), max({np.max(pos_ratios)})")
    neg_ctrs = [a["click"] / (a["show"] + 1e-8) for a in neg_stat]
    print(f"负向CTR-文本行高占比: mean({np.mean(neg_ratios)}), min({np.min(neg_ratios)}), max({np.max(neg_ratios)})")
    return pos_ctrs, neg_ctrs


def stat_text(
    data,
    src_ctr="click_rate_74803023",
    gen_ctr="click_rate_74803024",
    gen_show_key="show_74803024",
    gen_click_key="click_74803024",
):
    # 1. 原图单词个数-ctr散点图
    draw_word_count_to_ctr(data, src_ctr, "src_ocr_text")
    # 2. 生成图单词个数-ctr散点图
    draw_word_count_to_ctr(data, gen_ctr, "gen_ocr_text")

    x_labels = [
        "0.0-0.1",
        "0.1-0.2",
        "0.2-0.3",
        "0.3-0.4",
        "0.4-0.5",
        "0.5-0.6",
        "0.6-0.7",
        "0.7-0.8",
        "0.8-0.9",
        "0.9-1.0",
    ]
    # 3. 正向/负向点击率统计
    pos_ctrs, neg_ctrs = get_pos_and_neg_text_height_ratio_ctrs(
        data, src_ctr, gen_ctr, gen_show_key, gen_click_key)
    plot_click_rate_bar(x_labels, pos_ctrs, join(args.save_dir, "3_pos_text_height_ratio_ctrs.png"))
    plot_click_rate_bar(x_labels, neg_ctrs, join(args.save_dir, "3_neg_text_height_ratio_ctrs.png"))

    # 4. 生成图vs原图文案行高占比-ctr正向/负向直方图
    draw_pos_and_neg_text_height_ratio_hist(data, src_ctr, gen_ctr)


def stat_background_type(data_list, src_ctr_key="click_rate_74803023", gen_ctr_key="click_rate_74803024"):
    # 1. 统计原图中背景类型占比
    out = {}
    for line in data_list:
        if line['src_label']['background_classification'] not in out:
            out[line['src_label']['background_classification']] = 1
        else:
            out[line["src_label"]["background_classification"]] += 1
    num_tot = sum(list(out.values()))
    print("原图背景类型占比：")
    print([f"{k}: {v / num_tot * 100}%" for k, v in out.items()])

    # 2. 统计生成图中背景类型占比
    out = {}
    for line in data_list:
        if line["gen_label"]["background_classification"] not in out:
            out[line["gen_label"]["background_classification"]] = 1
        else:
            out[line["gen_label"]["background_classification"]] += 1
    num_tot = sum(list(out.values()))
    print("生成图背景类型占比：")
    print([f"{k}: {v / num_tot * 100}%" for k, v in out.items()])

    # 3. 生成图: 正向/负向的背景类型占比
    pos_bg = {}
    neg_bg = {}
    for line in tqdm(data_list):
        src_ctr = line[src_ctr_key]
        gen_ctr = line[gen_ctr_key]
        diff = gen_ctr - src_ctr
        if not math.isnan(diff):
            if diff > 0:
                if line["gen_label"]["background_classification"] not in pos_bg:
                    pos_bg[line["gen_label"]["background_classification"]] = 1
                else:
                    pos_bg[line["gen_label"]["background_classification"]] += 1
            elif diff < 0:
                if line["gen_label"]["background_classification"] not in neg_bg:
                    neg_bg[line["gen_label"]["background_classification"]] = 1
                else:
                    neg_bg[line["gen_label"]["background_classification"]] += 1
    num_tot = sum(list(pos_bg.values()))
    print("正向生成图背景类型占比：")
    print([f"{k}: {v / num_tot * 100}%" for k, v in pos_bg.items()])

    num_tot = sum(list(neg_bg.values()))
    print("负向生成图背景类型占比：")
    print([f"{k}: {v / num_tot * 100}%" for k, v in neg_bg.items()])


def plot_category_pie(categories_dict, top_n=12):
    """
    绘制商品类别占比饼图（支持字典输入，标签水平显示）
    :param categories_dict: 字典，格式为 {类别名称: 数量, ...}
    :param top_n: 重点显示前N个类别，其余合并为"其他"
    """
    # 从字典中提取类别和数量
    categories = list(categories_dict.keys())
    counts = list(categories_dict.values())

    # 计算占比并处理多类别
    total = sum(counts)
    percentages = [count/total*100 for count in counts]

    if len(categories) > top_n:
        # 按数量排序，取前top_n个
        sorted_data = sorted(zip(categories, counts, percentages), key=lambda x: x[1], reverse=True)
        top_categories = [x[0] for x in sorted_data[:top_n]]
        top_counts = [x[1] for x in sorted_data[:top_n]]
        top_percentages = [x[2] for x in sorted_data[:top_n]]

        # 合并剩余类别为"其他"
        other_count = sum([x[1] for x in sorted_data[top_n:]])
        other_percent = sum([x[2] for x in sorted_data[top_n:]])

        categories = top_categories + ["Other"]
        counts = top_counts + [other_count]
        percentages = top_percentages + [other_percent]

    # 设置画布大小
    plt.figure(figsize=(12, 10))

    # 绘制饼图（标签水平显示）
    wedges, texts, autotexts = plt.pie(
        counts,
        labels=categories,
        autopct='%1.1f%%',
        startangle=90,
        shadow=False,
        explode=[0.05 if i == counts.index(max(counts)) else 0 for i in range(len(counts))],
        textprops={'fontsize': 12, 'rotation': 0},  # 标签水平显示
        pctdistance=0.75
    )

    # 优化百分比文字
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    # 调整类目标签位置，避免重叠
    for text in texts:
        text.set_ha('center')
        text.set_va('center')
        x, y = text.get_position()
        text.set_position((x * 1.06, y * 1.06))  # 向外移动标签

    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(join(args.save_dir, 'categories_proportion.png'))


def stat_product_category(data):
    # 1. 原图
    product_category_src = {}
    count_category = {}
    for line in tqdm(data):
        if line['src_label']['subject_classification'] not in product_category_src:
            product_category_src[line['src_label']['subject_classification']] = {
                "show": line['show_74803023'],
                "click": line['click_74803023'],
            }
            count_category[line['src_label']['subject_classification']] = 1
        else:
            product_category_src[line["src_label"]["subject_classification"]]["show"] += line['show_74803023']
            product_category_src[line["src_label"]["subject_classification"]]["click"] += line["click_74803023"]
            count_category[line["src_label"]["subject_classification"]] += 1
    print("原图商品垂类CTR:")
    for k, v in product_category_src.items():
        print(f"{k}: {v['click'] / v['show']}")

    # 绘制饼图
    plot_category_pie(count_category)

    # 2. 生成图
    product_category = {}
    for line in tqdm(data):
        if line["gen_label"]["subject_classification"] not in product_category:
            product_category[line["gen_label"]["subject_classification"]] = {
                "show": line["show_74803024"],
                "click": line["click_74803024"],
            }
        else:
            product_category[line["gen_label"]["subject_classification"]]["show"] += line["show_74803024"]
            product_category[line["gen_label"]["subject_classification"]]["click"] += line["click_74803024"]
    print("生成图商品垂类CTR:")
    for k, v in product_category.items():
        print(f"{k}: {v['click'] / v['show']}")

    # 绘制ctr对比图
    categories, original_rates, generated_rates = [], [], []
    for k in product_category_src.keys():
        categories.append(k)
        original_rates.append(product_category_src[k]['click'] / product_category_src[k]['show'])
        generated_rates.append(product_category[k]['click'] / product_category[k]['show'])
    plot_click_rate_comparison(
        categories,
        original_rates,
        generated_rates,
        join(args.save_dir, "categories_ctr.png"),
        ori_color="#9467bd",
        tar_color="#ffbb78",
    )


def preprocess_data(data):
    out = []
    for line in tqdm(data):
        if math.isnan(line["show_74803023"]) or math.isnan(line["show_74803024"]):
            continue
        if math.isnan(line["click_74803023"]):
            line["click_74803023"] = 0
            line["click_rate_74803023"] = 0
        if math.isnan(line["click_74803024"]):
            line["click_74803024"] = 0
            line["click_rate_74803024"] = 0

        out.append(line)
    print(f"input {len(data)}, output {len(out)}.")
    return out


def main(args):
    # load dataset
    data_list = load_file(args.input_file)
    data_list = preprocess_data(data_list)

    # 1. 统计主体占比
    stat_subject_ratio(data_list, src_ctr="click_rate_74803023", gen_ctr="click_rate_74803024")

    # 2. 对比度统计分析
    stat_color_difference(data_list, src_ctr="click_rate_74803023", gen_ctr="click_rate_74803024")

    # 3. 文案统计
    stat_text(data_list, src_ctr="click_rate_74803023", gen_ctr="click_rate_74803024")

    # 4. 背景类型统计
    stat_background_type(data_list, src_ctr_key="click_rate_74803023", gen_ctr_key="click_rate_74803024")

    # 5. 商品类目统计
    stat_product_category(data_list)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
