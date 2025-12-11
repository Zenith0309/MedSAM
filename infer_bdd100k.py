# -*- coding: utf-8 -*-
"""
BDD100K 推理和评估脚本
使用微调后的 SAM 模型在 BDD100K 验证集上进行推理和评估
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image

join = os.path.join
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
from datetime import datetime
import glob

# BDD100K 19个类别
BDD100K_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

# 为可视化定义颜色映射（19种颜色）
COLORS = [
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32],    # bicycle
]


class BDD100KInferDataset(Dataset):
    """BDD100K 推理数据集加载器"""
    def __init__(self, data_root, num_classes=19):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.num_classes = num_classes
        print(f"推理数据集图像数量: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # 加载图像
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(join(self.img_path, img_name), allow_pickle=True)
        img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, 1024, 1024)
        
        # 加载真值标注
        gt = np.load(self.gt_path_files[index], allow_pickle=True)
        
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt).long(),
            img_name,
        )


class MedSAM(nn.Module):
    """SAM 模型包装类"""
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, box):
        with torch.no_grad():
            image_embedding = self.image_encoder(image)
            
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            
            low_res_masks, _ = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            ori_res_masks = F.interpolate(
                low_res_masks,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
        return ori_res_masks


def compute_iou(pred, target, num_classes):
    """
    计算 IoU（Intersection over Union）
    
    参数:
        pred: 预测结果 (H, W)
        target: 真值标注 (H, W)
        num_classes: 类别数量
    
    返回:
        iou_per_class: 每个类别的 IoU
        mean_iou: 平均 IoU
    """
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union == 0:
            # 如果该类别不存在，跳过
            iou = float('nan')
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    # 计算 mIoU（排除 NaN）
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if len(valid_ious) > 0 else 0.0
    
    return ious, mean_iou


def visualize_prediction(image, gt, pred, save_path, img_name):
    """
    可视化预测结果
    
    参数:
        image: 原始图像 (3, H, W), [0, 1]
        gt: 真值标注 (H, W)
        pred: 预测结果 (H, W)
        save_path: 保存路径
        img_name: 图像名称
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 转换图像格式
    image_np = image.permute(1, 2, 0).cpu().numpy()
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    
    # 原始图像
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 真值
    gt_colored = np.zeros((gt_np.shape[0], gt_np.shape[1], 3), dtype=np.uint8)
    for cls_id in range(len(COLORS)):
        mask = gt_np == cls_id
        gt_colored[mask] = COLORS[cls_id]
    axes[1].imshow(gt_colored)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # 预测
    pred_colored = np.zeros((pred_np.shape[0], pred_np.shape[1], 3), dtype=np.uint8)
    for cls_id in range(len(COLORS)):
        mask = pred_np == cls_id
        pred_colored[mask] = COLORS[cls_id]
    axes[2].imshow(pred_colored)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(join(save_path, img_name.replace('.npy', '_vis.png')), dpi=150, bbox_inches='tight')
    plt.close()


def get_bbox_from_mask(mask):
    """从掩码中提取边界框"""
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return np.array([x_min, y_min, x_max, y_max])


def infer_bdd100k(model, dataloader, device, output_dir, num_classes=19, visualize=True):
    """
    在 BDD100K 数据集上进行推理和评估
    
    参数:
        model: 微调后的模型
        dataloader: 数据加载器
        device: 设备
        output_dir: 输出目录
        num_classes: 类别数量
        visualize: 是否可视化结果
    """
    model.eval()
    
    # 创建输出目录
    vis_dir = join(output_dir, 'visualizations')
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
    
    all_ious = []
    class_ious_sum = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    
    print("开始推理...")
    
    with torch.no_grad():
        for idx, (image, gt, img_name_tuple) in enumerate(tqdm(dataloader)):
            img_name = img_name_tuple[0]
            image = image.to(device)
            gt = gt[0].cpu().numpy()  # (1024, 1024)
            
            # 为每个类别分别推理
            pred_full = np.zeros_like(gt, dtype=np.int64)
            
            for cls_id in range(num_classes):
                # 获取当前类别的真值掩码
                cls_mask = (gt == cls_id)
                
                if cls_mask.sum() == 0:
                    continue
                
                # 从真值中获取边界框（推理时使用真值 bbox）
                bbox = get_bbox_from_mask(cls_mask)
                
                if bbox is None:
                    continue
                
                # 进行推理
                bbox_input = bbox[None, :].astype(np.float32)
                pred_mask = model(image, bbox_input)  # (1, 1, 1024, 1024)
                
                # 应用 sigmoid 并二值化
                pred_mask = torch.sigmoid(pred_mask)
                pred_mask_binary = (pred_mask > 0.5).cpu().numpy()[0, 0]  # (1024, 1024)
                
                # 将预测结果添加到完整预测中
                pred_full[pred_mask_binary] = cls_id
            
            # 计算 IoU
            ious_per_class, mean_iou = compute_iou(pred_full, gt, num_classes)
            all_ious.append(mean_iou)
            
            # 累积每个类别的 IoU
            for cls_id, iou in enumerate(ious_per_class):
                if not np.isnan(iou):
                    class_ious_sum[cls_id] += iou
                    class_counts[cls_id] += 1
            
            # 可视化（只保存前20张）
            if visualize and idx < 20:
                visualize_prediction(
                    image[0],
                    torch.tensor(gt),
                    torch.tensor(pred_full),
                    vis_dir,
                    img_name
                )
    
    # 计算整体指标
    overall_miou = np.mean(all_ious)
    
    # 计算每个类别的平均 IoU
    class_mean_ious = []
    for cls_id in range(num_classes):
        if class_counts[cls_id] > 0:
            class_mean_iou = class_ious_sum[cls_id] / class_counts[cls_id]
        else:
            class_mean_iou = 0.0
        class_mean_ious.append(class_mean_iou)
    
    return overall_miou, class_mean_ious


def main():
    parser = argparse.ArgumentParser(description='BDD100K 推理和评估')
    parser.add_argument(
        '--val_npy_path',
        type=str,
        default='data/bdd100k_npy/val',
        help='验证集 npy 文件路径'
    )
    parser.add_argument(
        '--model_checkpoint',
        type=str,
        required=True,
        help='微调后的模型检查点路径'
    )
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default='work_dir/SAM/sam_vit_b_01ec64.pth',
        help='SAM 预训练权重路径（用于加载 Image Encoder 和 Prompt Encoder）'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='vit_b',
        help='模型类型'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./inference_results',
        help='推理结果输出目录'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='批次大小（推理时建议为1）'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='数据加载线程数'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='推理设备'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='是否可视化结果'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=19,
        help='类别数量'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("加载数据集...")
    val_dataset = BDD100KInferDataset(args.val_npy_path, num_classes=args.num_classes)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # 加载模型
    print("加载模型...")
    sam_model = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    
    # 加载微调的权重
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    
    # 处理 DDP 保存的模型
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 移除 'module.' 前缀
        else:
            new_state_dict[k] = v
    
    medsam_model.load_state_dict(new_state_dict)
    print(f"成功加载检查点: {args.model_checkpoint}")
    print(f"训练轮次: {checkpoint['epoch']}")
    
    # 进行推理和评估
    print("\n" + "=" * 60)
    print("开始推理和评估")
    print("=" * 60)
    
    overall_miou, class_mean_ious = infer_bdd100k(
        model=medsam_model,
        dataloader=val_dataloader,
        device=device,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        visualize=args.visualize
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"整体 mIoU: {overall_miou:.4f}")
    print("\n每个类别的 IoU:")
    print("-" * 60)
    
    for cls_id, (cls_name, cls_iou) in enumerate(zip(BDD100K_CLASSES, class_mean_ious)):
        print(f"{cls_id:2d}. {cls_name:20s}: {cls_iou:.4f}")
    
    print("=" * 60)
    
    # 保存结果到文件
    results_file = join(args.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("BDD100K 评估结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"模型检查点: {args.model_checkpoint}\n")
        f.write(f"训练轮次: {checkpoint['epoch']}\n")
        f.write(f"整体 mIoU: {overall_miou:.4f}\n")
        f.write("\n每个类别的 IoU:\n")
        f.write("-" * 60 + "\n")
        for cls_id, (cls_name, cls_iou) in enumerate(zip(BDD100K_CLASSES, class_mean_ious)):
            f.write(f"{cls_id:2d}. {cls_name:20s}: {cls_iou:.4f}\n")
        f.write("=" * 60 + "\n")
    
    print(f"\n结果已保存到: {results_file}")
    
    if args.visualize:
        print(f"可视化结果已保存到: {join(args.output_dir, 'visualizations')}")


if __name__ == "__main__":
    main()
