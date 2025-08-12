import numpy as np
from typing import List

def calculate_word_accuracy(predicted: str, ground_truth: str) -> float:
    """
    计算单词准确率(Word Accuracy)
    
    单词准确率是完全正确识别的单词数量占总单词数量的比例
    
    参数:
        predicted: 模型预测的字符串
        ground_truth: 真实标签字符串
        
    返回:
        单词准确率，范围在0到1之间
    """
    # 将字符串按空格分割为单词列表
    pred_words = predicted.split()
    gt_words = ground_truth.split()
    
    # 如果没有单词，视为完全匹配
    if len(gt_words) == 0:
        return 1.0 if len(pred_words) == 0 else 0.0
    
    # 计算完全匹配的单词数量
    correct_count = 0
    min_length = min(len(pred_words), len(gt_words))
    
    # 比较相同位置的单词
    for i in range(min_length):
        if pred_words[i] == gt_words[i]:
            correct_count += 1
    
    # 剩余的单词视为不匹配
    return correct_count / len(gt_words)

def levenshtein_distance(s1: List[str], s2: List[str]) -> int:
    """
    计算两个单词列表之间的编辑距离(莱文斯坦距离)
    用于计算WER
    
    参数:
        s1: 第一个单词列表(预测结果)
        s2: 第二个单词列表(真实标签)
        
    返回:
        两个列表之间的编辑距离
    """
    # 创建距离矩阵
    matrix = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)
    
    # 初始化矩阵
    for i in range(len(s1) + 1):
        matrix[i][0] = i
    for j in range(len(s2) + 1):
        matrix[0][j] = j
    
    # 填充矩阵
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i-1] == s2[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                # 插入、删除、替换操作的最小代价
                insert = matrix[i][j-1] + 1
                delete = matrix[i-1][j] + 1
                replace = matrix[i-1][j-1] + 1
                matrix[i][j] = min(insert, delete, replace)
    
    return matrix[len(s1)][len(s2)]

def calculate_wer(predicted: str, ground_truth: str) -> float:
    """
    计算单词错误率(Word Error Rate, WER)
    
    WER是基于编辑距离(插入、删除、替换)计算的错误率
    
    参数:
        predicted: 模型预测的字符串
        ground_truth: 真实标签字符串
        
    返回:
        单词错误率，范围在0到正无穷大之间
    """
    # 将字符串按空格分割为单词列表
    pred_words = predicted.split()
    gt_words = ground_truth.split()
    
    # 处理空字符串情况
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else float('inf')
    
    # 计算编辑距离
    distance = levenshtein_distance(pred_words, gt_words)
    
    # 计算WER
    return distance / len(gt_words)

# 示例用法
if __name__ == "__main__":
    # 测试案例1
    predicted1 = "hello world"
    ground_truth1 = "hello world"
    print(f"测试案例1:")
    print(f"预测: {predicted1}")
    print(f"真实: {ground_truth1}")
    print(f"单词准确率: {calculate_word_accuracy(predicted1, ground_truth1):.4f}")
    print(f"WER: {calculate_wer(predicted1, ground_truth1):.4f}\n")
    
    # 测试案例2
    predicted2 = "hello there"
    ground_truth2 = "hello world"
    print(f"测试案例2:")
    print(f"预测: {predicted2}")
    print(f"真实: {ground_truth2}")
    print(f"单词准确率: {calculate_word_accuracy(predicted2, ground_truth2):.4f}")
    print(f"WER: {calculate_wer(predicted2, ground_truth2):.4f}\n")
    
    # 测试案例3
    predicted3 = "hello beautiful world"
    ground_truth3 = "hello world"
    print(f"测试案例3:")
    print(f"预测: {predicted3}")
    print(f"真实: {ground_truth3}")
    print(f"单词准确率: {calculate_word_accuracy(predicted3, ground_truth3):.4f}")
    print(f"WER: {calculate_wer(predicted3, ground_truth3):.4f}\n")
    
    # 测试案例4
    predicted4 = "hi world"
    ground_truth4 = "hello beautiful world"
    print(f"测试案例4:")
    print(f"预测: {predicted4}")
    print(f"真实: {ground_truth4}")
    print(f"单词准确率: {calculate_word_accuracy(predicted4, ground_truth4):.4f}")
    print(f"WER: {calculate_wer(predicted4, ground_truth4):.4f}")
