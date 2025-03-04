import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.pspnet import PSPNet
import yaml  # 用于加载配置文件


# 配置参数
class Args:
    def __init__(self):
        self.arch = 'pspnet50_ade'
        self.cfg = 'config/ade20k/ade20k_pspnet101.yaml'  # 配置文件路径
        self.pth = 'initmodel/ade20k/ade_pspnet101.pth'  # 预训练权重路径
        self.input_dir = r'D:\NanJing\test\input'  # 输入图片目录
        self.output_dir = r'D:\NanJing\test\output'  # 输出掩码目录
        self.save_stats = r'D:\NanJing\test\output\class_stats.csv'  # 统计结果保存路径


# 加载配置文件
def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


# 初始化模型
def init_model(args):
    cfg = load_config(args.cfg)
    model = PSPNet(layers=cfg['layers'], classes=cfg['classes'], zoom_factor=cfg['zoom_factor'])
    checkpoint = torch.load(args.pth, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval().cuda()


# 处理单张图像
def process_image(model, img_path, output_dir):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # 预处理
    image = cv2.resize(image, (473, 473))
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().cuda()

    # 推理
    with torch.no_grad():
        output = model(image)
    pred = output.argmax(1).squeeze().cpu().numpy()
    pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # 保存掩码
    mask_path = os.path.join(output_dir, os.path.basename(img_path).replace('.jpg', '.png'))
    cv2.imwrite(mask_path, pred)
    return pred


# 统计类别占比
def calculate_class_ratio(pred_mask, num_classes=150):
    counts = np.bincount(pred_mask.flatten(), minlength=num_classes)
    total = pred_mask.size
    return counts / total


# 主函数
def main():
    args = Args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = init_model(args)
    image_files = [f for f in os.listdir(args.input_dir) if f.endswith(('jpg', 'png'))]

    stats = []
    for img_file in tqdm(image_files):
        img_path = os.path.join(args.input_dir, img_file)
        try:
            pred_mask = process_image(model, img_path, args.output_dir)
            ratios = calculate_class_ratio(pred_mask)
            stats.append({
                'image': img_file,
                **{f'class_{i}': ratios[i] for i in range(len(ratios))}
            })
        except Exception as e:
            print(f"处理 {img_file} 失败: {str(e)}")

    # 保存统计结果
    df = pd.DataFrame(stats)
    df.to_csv(args.save_stats, index=False)
    print(f"处理完成！统计结果已保存至 {args.save_stats}")


if __name__ == '__main__':
    main()