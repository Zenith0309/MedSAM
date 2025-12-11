# -*- coding: utf-8 -*-
"""
BDD100K 数据集预处理脚本
将 BDD100K 语义分割数据集转换为 SAM 训练所需的 npy 格式
"""

import numpy as np
import os
from PIL import Image
import cv2
from tqdm import tqdm
import argparse

join = os.path.join

# BDD100K 19个语义分割类别
# 类别ID: 0=road, 1=sidewalk, 2=building, 3=wall, 4=fence, 5=pole, 
# 6=traffic light, 7=traffic sign, 8=vegetation, 9=terrain, 10=sky,
# 11=person, 12=rider, 13=car, 14=truck, 15=bus, 16=train, 17=motorcycle, 18=bicycle
BDD100K_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

# 定义类别映射 - 如果您的标注使用不同的ID，请修改此映射
# BDD100K 原始标注使用的类别ID可能需要根据实际数据集调整
CLASS_MAPPING = {
    0: 0,   # road
    1: 1,   # sidewalk
    2: 2,   # building
    3: 3,   # wall
    4: 4,   # fence
    5: 5,   # pole
    6: 6,   # traffic light
    7: 7,   # traffic sign
    8: 8,   # vegetation
    9: 9,   # terrain
    10: 10, # sky
    11: 11, # person
    12: 12, # rider
    13: 13, # car
    14: 14, # truck
    15: 15, # bus
    16: 16, # train
    17: 17, # motorcycle
    18: 18, # bicycle
}


def preprocess_bdd100k(
    img_path,
    gt_path,
    output_path,
    image_size=1024,
    split='train'
):
    """
    预处理 BDD100K 数据集
    
    参数:
        img_path: 图像文件夹路径
        gt_path: 标注文件夹路径
        output_path: 输出 npy 文件夹路径
        image_size: 目标图像尺寸（默认1024x1024）
        split: 数据集划分（train/val/test）
    """
    # 创建输出目录
    img_output_dir = join(output_path, split, "imgs")
    gt_output_dir = join(output_path, split, "gts")
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(gt_output_dir, exist_ok=True)
    
    # 获取所有图像文件
    img_names = sorted([f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png'))])
    print(f"找到 {len(img_names)} 个图像文件在 {split} 集")
    
    processed_count = 0
    skipped_count = 0
    
    for img_name in tqdm(img_names, desc=f"处理 {split} 集"):
        # 构建对应的标注文件名
        # BDD100K 标注文件通常和图像文件同名，但扩展名可能为 .png
        base_name = os.path.splitext(img_name)[0]
        gt_name = base_name + '.png'  # 根据实际情况调整
        
        img_file = join(img_path, img_name)
        gt_file = join(gt_path, gt_name)
        
        # 检查标注文件是否存在
        if not os.path.exists(gt_file):
            print(f"警告: 标注文件不存在: {gt_file}")
            skipped_count += 1
            continue
        
        try:
            # 读取图像 (使用 OpenCV 以保持与原始代码一致)
            img = cv2.imread(img_file)
            if img is None:
                print(f"警告: 无法读取图像: {img_file}")
                skipped_count += 1
                continue
            
            # 转换 BGR 到 RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize 到目标尺寸 (1024x1024)
            img_resized = cv2.resize(
                img, 
                (image_size, image_size), 
                interpolation=cv2.INTER_LINEAR
            )
            
            # 归一化到 [0, 1]
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # 确保范围在 [0, 1]
            assert np.max(img_normalized) <= 1.0 and np.min(img_normalized) >= 0.0, \
                f"图像归一化失败: max={np.max(img_normalized)}, min={np.min(img_normalized)}"
            
            # 读取标注 (语义分割标签)
            # 假设标注是单通道图像，每个像素值代表类别ID
            gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
            if gt is None:
                # 尝试使用 PIL 读取
                gt = np.array(Image.open(gt_file))
                if len(gt.shape) == 3:
                    gt = gt[:, :, 0]  # 取第一个通道
            
            # Resize 标注（使用最近邻插值以保持标签完整性）
            gt_resized = cv2.resize(
                gt,
                (image_size, image_size),
                interpolation=cv2.INTER_NEAREST
            )
            
            # 应用类别映射（如果需要）
            gt_mapped = np.zeros_like(gt_resized, dtype=np.uint8)
            for original_id, mapped_id in CLASS_MAPPING.items():
                gt_mapped[gt_resized == original_id] = mapped_id
            
            # 检查是否有有效的标注
            unique_labels = np.unique(gt_mapped)
            if len(unique_labels) <= 1:  # 只有背景
                print(f"警告: 标注中没有前景对象: {gt_file}")
                skipped_count += 1
                continue
            
            # 保存为 npy 格式
            # 图像保存为 (H, W, 3) 格式，范围 [0, 1]
            output_img_file = join(img_output_dir, base_name + '.npy')
            np.save(output_img_file, img_normalized)
            
            # 标注保存为 (H, W) 格式，类别ID
            output_gt_file = join(gt_output_dir, base_name + '.npy')
            np.save(output_gt_file, gt_mapped)
            
            processed_count += 1
            
        except Exception as e:
            print(f"错误: 处理 {img_name} 时出错: {str(e)}")
            skipped_count += 1
            continue
    
    print(f"\n{split} 集处理完成:")
    print(f"  成功处理: {processed_count} 个样本")
    print(f"  跳过: {skipped_count} 个样本")
    print(f"  输出目录: {output_path}/{split}/")


def main():
    parser = argparse.ArgumentParser(description='BDD100K 数据集预处理')
    parser.add_argument(
        '--bdd100k_root',
        type=str,
        required=True,
        help='BDD100K 数据集根目录路径'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='data/bdd100k_npy',
        help='输出 npy 文件路径 (默认: data/bdd100k_npy)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=1024,
        help='目标图像尺寸 (默认: 1024)'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val'],
        help='要处理的数据集划分 (默认: train val)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BDD100K 数据集预处理")
    print("=" * 60)
    print(f"数据集根目录: {args.bdd100k_root}")
    print(f"输出路径: {args.output_path}")
    print(f"图像尺寸: {args.image_size}x{args.image_size}")
    print(f"处理的数据集划分: {args.splits}")
    print("=" * 60)
    
    # BDD100K 标准目录结构:
    # bdd100k_root/
    #   ├── images/
    #   │   └── 10k/
    #   │       ├── train/
    #   │       ├── val/
    #   │       └── test/
    #   └── labels/
    #       └── sem_seg/
    #           ├── masks/
    #           │   ├── train/
    #           │   ├── val/
    #           │   └── test/
    
    for split in args.splits:
        img_path = join(args.bdd100k_root, 'images', '10k', split)
        gt_path = join(args.bdd100k_root, 'labels', 'sem_seg', 'masks', split)
        
        # 检查路径是否存在
        if not os.path.exists(img_path):
            print(f"错误: 图像路径不存在: {img_path}")
            print("请检查 BDD100K 数据集路径是否正确")
            continue
        
        if not os.path.exists(gt_path):
            print(f"错误: 标注路径不存在: {gt_path}")
            print("请检查 BDD100K 数据集路径是否正确")
            continue
        
        preprocess_bdd100k(
            img_path=img_path,
            gt_path=gt_path,
            output_path=args.output_path,
            image_size=args.image_size,
            split=split
        )
        print()
    
    print("=" * 60)
    print("所有数据集预处理完成！")
    print("=" * 60)
    print(f"\n预处理后的数据存储在: {args.output_path}/")
    print("\n数据格式:")
    print("  - 图像: (1024, 1024, 3), float32, 范围 [0, 1]")
    print("  - 标注: (1024, 1024), uint8, 类别ID [0-18]")
    print("\n可以使用以下命令开始训练:")
    print(f"  python train_bdd100k.py -i {args.output_path}/train")


if __name__ == "__main__":
    main()
