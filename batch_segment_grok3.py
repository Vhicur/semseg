import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image
import sys

sys.path.append('.')  # 添加当前目录到 PYTHONPATH

from tool.demo import build_model, inference  # 假设 demo.py 提供这些函数

# 设置图片文件夹路径
image_folder = r'D:\NanJing\jiejing_point\nanjing_point'
config_file = 'config/ade20k/ade20k_pspnet50.yaml'

# 你需要的类别 ID
desired_classes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 26, 43, 44, 49, 53, 54, 55,
                   56, 61, 62, 81, 83, 84, 85, 92, 93, 94, 95, 127, 128, 129, 137}

# 获取所有图片文件
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

# 加载模型（只加载一次）
model = build_model(config_file)  # 假设 demo.py 提供 build_model 函数
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

# 存储类别占比的列表
stats = []

# 遍历每张图片
with torch.no_grad():  # 禁用梯度计算以节省内存
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, image_file)

        # 加载并预处理图片（参考 demo.py 的逻辑）
        img = Image.open(image_path).convert('RGB')
        seg_map = inference(model, img, config_file)  # 假设 inference 返回分割结果

        # 转换为 numpy 数组
        seg_img = np.array(seg_map)
        unique, counts = np.unique(seg_img, return_counts=True)
        total_pixels = seg_img.size

        # 只统计需要的类别
        class_ratios = {int(k): v / total_pixels for k, v in zip(unique, counts) if int(k) in desired_classes}

        # 添加到统计列表
        stats.append({'image': image_file, **class_ratios})

# 将结果保存到 CSV
df = pd.DataFrame(stats)
class_names = {
    1: 'wall', 2: 'building', 3: 'sky', 4: 'floor', 5: 'tree', 6: 'ceiling', 7: 'road', 8: 'bed',
    9: 'windowpane', 10: 'grass', 11: 'cabinet', 12: 'sidewalk', 13: 'person', 14: 'earth', 15: 'door',
    16: 'table', 17: 'mountain', 18: 'plant', 21: 'car', 22: 'water', 26: 'house', 43: 'column',
    44: 'signboard', 49: 'skyscraper', 53: 'path', 54: 'stairs', 55: 'runway', 56: 'case', 61: 'river',
    62: 'bridge', 81: 'bus', 83: 'light', 84: 'truck', 85: 'tower', 92: 'dirt track', 93: 'apparel',
    94: 'pole', 95: 'land', 127: 'animal', 128: 'bicycle', 129: 'lake', 137: 'traffic light'
}
df.rename(columns=class_names, inplace=True)
df.to_csv('category_ratios.csv', index=False)
print("Statistics completed. Results saved to 'category_ratios.csv'.")