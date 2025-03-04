import os
import logging
import argparse
import glob  # 导入 glob 模块，用于查找文件
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import pandas as pd  # 导入 pandas 模块

from util import config
from util.util import colorize

cv2.ocl.setUseOpenCL(False)

#运行命令 python gemini_demo.py --config config/ade20k/ade20k_pspnet101.yaml --image_dir test/input
#
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet101.yaml', help='config file')
    parser.add_argument('--image_dir', type=str, default=None, help='input image directory')  # 修改为目录
    parser.add_argument('--image', type=str, default='figure/demo/ADE_val_00001515.jpg', help='input image')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    if args.image_dir is None:
        cfg = config.load_cfg_from_cfg_file(args.config)
        cfg.image = args.image
    else:
        cfg = config.load_cfg_from_cfg_file(args.config)
        cfg.image_dir = args.image_dir
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert args.split in ['train', 'val', 'test']
    if args.arch == 'psp':
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    elif args.arch == 'psa':
        if args.compact:
            args.mask_h = (args.train_h - 1) // (8 * args.shrink_factor) + 1
            args.mask_w = (args.train_w - 1) // (8 * args.shrink_factor) + 1
        else:
            assert (args.mask_h is None and args.mask_w is None) or (args.mask_h is not None and args.mask_w is not None)
            if args.mask_h is None and args.mask_w is None:
                args.mask_h = 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1
                args.mask_w = 2 * ((args.train_w - 1) // (8 * args.shrink_factor) + 1) - 1
            else:
                assert (args.mask_h % 2 == 1) and (args.mask_h >= 3) and (
                        args.mask_h <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
                assert (args.mask_w % 2 == 1) and (args.mask_w >= 3) and (
                        args.mask_w <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
    else:
        raise Exception('architecture not supported yet'.format(args.arch))

# 添加一个函数，计算分割结果的类别比例
def calculate_class_proportions(prediction, num_classes):
    """计算分割结果中每个类别的像素比例。"""
    h, w = prediction.shape
    total_pixels = h * w
    class_counts = np.zeros(num_classes)
    for i in range(num_classes):
        class_counts[i] = np.sum(prediction == i)
    class_proportions = class_counts / total_pixels
    return class_proportions


def main():
    global args, logger
    args = get_parser()
    check(args)
    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    colors = np.loadtxt(args.colors_path).astype('uint8')

    if args.arch == 'psp':
        from model.pspnet import PSPNet
        model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False)
    elif args.arch == 'psa':
        from model.psanet import PSANet
        model = PSANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, compact=args.compact,
                       shrink_factor=args.shrink_factor, mask_h=args.mask_h, mask_w=args.mask_w,
                       normalization_factor=args.normalization_factor, psa_softmax=args.psa_softmax, pretrained=False)
    logger.info(model)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    # 创建一个 DataFrame 来存储类别比例数据
    class_proportions_data = []

    # 批量处理图片
    if args.image_dir:
        image_files = glob.glob(os.path.join(args.image_dir, '*.*'))  # 获取目录下所有图片文件
        image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp']] # filter files which are images
        if not image_files:
            logger.error(f"No images found in directory: {args.image_dir}")
            return

        for image_path in image_files:
            logger.info(f"Processing image: {image_path}")
            start_time = time.time()
            # 调用 test 函数处理每一张图片，并获取类别比例
            class_proportions = test(model.eval(), image_path, args.classes, mean, std, args.base_size, args.test_h, args.test_w, args.scales, colors)
            end_time = time.time()
            logger.info(f"Processed in {end_time - start_time:.2f} seconds")

            # 将类别比例数据添加到 DataFrame
            image_name = os.path.basename(image_path)
            class_proportions_data.append([image_name] + class_proportions.tolist())

        # 创建 DataFrame
        columns = ['image_name'] + [f'class_{i}' for i in range(args.classes)]
        df = pd.DataFrame(class_proportions_data, columns=columns)

        # 保存 DataFrame 到 CSV 文件
        csv_path = os.path.join('./test/output/', 'class_proportions.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Class proportions data saved to: {csv_path}")

    else:
        # 如果没有指定目录，则按原方式处理单张图片
        test(model.eval(), args.image, args.classes, mean, std, args.base_size, args.test_h, args.test_w, args.scales, colors)

def net_process(model, image, mean, std=None, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def test(model, image_path, classes, mean, std, base_size, crop_h, crop_w, scales, colors):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
    h, w, _ = image.shape
    prediction = np.zeros((h, w, classes), dtype=float)
    for scale in scales:
        long_size = round(scale * base_size)
        new_h = long_size
        new_w = long_size
        if h > w:
            new_w = round(long_size/float(h)*w)
        else:
            new_h = round(long_size/float(w)*h)
        image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)

    prediction = scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std) #Fix bug by removing duplicate
    prediction = np.argmax(prediction, axis=2)
    gray = np.uint8(prediction)
    color = colorize(gray, colors)
    image_name = os.path.basename(image_path).split('.')[0] # use basename
    gray_path = os.path.join('./test/output/', image_name + '_gray.png')
    color_path = os.path.join('./test/output/', image_name + '_color.png')
    cv2.imwrite(gray_path, gray)
    color.save(color_path)
    logger.info("=> Prediction saved in {}".format(color_path))

    # 计算并返回类别比例
    class_proportions = calculate_class_proportions(prediction, classes)
    return class_proportions


if __name__ == '__main__':
    main()