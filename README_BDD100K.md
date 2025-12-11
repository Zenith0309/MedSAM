# BDD100K 自动驾驶数据集上的 SAM Mask Decoder 微调

## 项目目标

本项目旨在将 Segment Anything Model (SAM) 应用于自动驾驶场景，通过在 BDD100K 数据集上**仅微调 Mask Decoder 部分**来实现高效的语义分割。为了节省显存并提高训练效率，我们**冻结 Image Encoder 和 Prompt Encoder**，仅训练 Mask Decoder。

### 主要特点

- ✅ **仅微调 Mask Decoder**：冻结 Image Encoder 和 Prompt Encoder，大幅减少显存占用
- ✅ **多 GPU 分布式训练**：支持 4x NVIDIA 2080Ti（11GB 显存）分布式训练
- ✅ **混合精度训练（AMP）**：使用 PyTorch AMP 进一步节省显存
- ✅ **梯度累积**：支持梯度累积以模拟更大的 batch size
- ✅ **BDD100K 适配**：完整支持 BDD100K 数据集的 19 个语义类别
- ✅ **完整的训练和推理流程**：包含数据预处理、训练、推理和评估

---

## 环境配置

### 系统要求

- **操作系统**: Linux (推荐 Ubuntu 18.04+)
- **GPU**: 4x NVIDIA 2080Ti (11GB 显存) 或同等配置
- **CUDA**: 11.1+
- **Python**: 3.8+

### 安装步骤

1. **创建虚拟环境**
```bash
conda create -n medsam python=3.10 -y
conda activate medsam
```

2. **安装 PyTorch**
```bash
# CUDA 11.8 版本
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 11.7 版本
# pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu117
```

3. **克隆仓库并安装依赖**
```bash
git clone https://github.com/Zenith0309/MedSAM.git
cd MedSAM
pip install -e .
```

4. **安装额外依赖**
```bash
pip install opencv-python matplotlib tqdm monai
```

5. **下载 SAM 预训练权重**
```bash
mkdir -p work_dir/SAM
cd work_dir/SAM
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
cd ../..
```

---

## BDD100K 数据集准备

### 数据集简介

**BDD100K** 是一个大规模自动驾驶数据集，包含：
- **10,000 张**语义分割标注图像
  - 7,000 张训练集
  - 1,000 张验证集
  - 2,000 张测试集
- **19 个语义类别**：
  - 0: road（道路）
  - 1: sidewalk（人行道）
  - 2: building（建筑物）
  - 3: wall（墙）
  - 4: fence（围栏）
  - 5: pole（杆）
  - 6: traffic light（交通灯）
  - 7: traffic sign（交通标志）
  - 8: vegetation（植被）
  - 9: terrain（地形）
  - 10: sky（天空）
  - 11: person（行人）
  - 12: rider（骑行者）
  - 13: car（汽车）
  - 14: truck（卡车）
  - 15: bus（公交车）
  - 16: train（火车）
  - 17: motorcycle（摩托车）
  - 18: bicycle（自行车）

### 下载数据集

1. 访问 [BDD100K 官网](https://bdd-data.berkeley.edu/)
2. 注册账号并下载以下数据：
   - `bdd100k_images_10k.zip` - 图像数据
   - `bdd100k_sem_seg_labels_trainval.zip` - 语义分割标注

3. 解压数据集到指定目录：
```bash
mkdir -p data/bdd100k
cd data/bdd100k
unzip bdd100k_images_10k.zip
unzip bdd100k_sem_seg_labels_trainval.zip
cd ../..
```

### 数据集目录结构

解压后的目录结构应如下：
```
data/bdd100k/
├── images/
│   └── 10k/
│       ├── train/       # 7,000 张训练图像
│       ├── val/         # 1,000 张验证图像
│       └── test/        # 2,000 张测试图像
└── labels/
    └── sem_seg/
        └── masks/
            ├── train/   # 训练集标注
            ├── val/     # 验证集标注
            └── test/    # 测试集标注
```

---

## 数据预处理

### 预处理步骤

运行预处理脚本将 BDD100K 数据转换为 SAM 训练所需的格式：

```bash
python pre_bdd100k.py \
    --bdd100k_root data/bdd100k \
    --output_path data/bdd100k_npy \
    --image_size 1024 \
    --splits train val
```

### 参数说明

- `--bdd100k_root`: BDD100K 数据集根目录
- `--output_path`: 输出 npy 文件路径（默认: `data/bdd100k_npy`）
- `--image_size`: 目标图像尺寸（默认: 1024）
- `--splits`: 要处理的数据集划分（默认: `train val`）

### 预处理后的数据格式

- **图像**: `(1024, 1024, 3)`, `float32`, 范围 `[0, 1]`
- **标注**: `(1024, 1024)`, `uint8`, 类别 ID `[0-18]`

预处理后的目录结构：
```
data/bdd100k_npy/
├── train/
│   ├── imgs/        # 训练图像 (.npy)
│   └── gts/         # 训练标注 (.npy)
└── val/
    ├── imgs/        # 验证图像 (.npy)
    └── gts/         # 验证标注 (.npy)
```

---

## 模型训练

### 单卡训练

适用于测试或小规模实验：

```bash
python train_bdd100k.py \
    -i data/bdd100k_npy/train \
    -task_name BDD100K-SAM-MaskDecoder \
    -model_type vit_b \
    -checkpoint work_dir/SAM/sam_vit_b_01ec64.pth \
    -work_dir ./work_dir \
    -batch_size 4 \
    -num_epochs 100 \
    -lr 5e-5 \
    -use_amp \
    --grad_acc_steps 4 \
    --device cuda:0
```

### 多卡训练（推荐）

针对 4x 2080Ti 优化的分布式训练命令：

```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 启动分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_bdd100k.py \
    -i data/bdd100k_npy/train \
    -task_name BDD100K-SAM-MaskDecoder \
    -model_type vit_b \
    -checkpoint work_dir/SAM/sam_vit_b_01ec64.pth \
    -work_dir ./work_dir \
    -batch_size 4 \
    -num_epochs 100 \
    -lr 5e-5 \
    -use_amp \
    --grad_acc_steps 4 \
    --world_size 4 \
    --distributed
```

### 训练参数说明

#### 基础参数
- `-i, --tr_npy_path`: 训练数据路径
- `-task_name`: 任务名称（用于保存目录）
- `-model_type`: 模型类型（`vit_b`, `vit_l`, `vit_h`）
- `-checkpoint`: SAM 预训练权重路径
- `-work_dir`: 工作目录

#### 训练超参数
- `-batch_size`: 每个 GPU 的 batch size（**2080Ti 建议 2-4**）
- `-num_epochs`: 训练轮数（默认 100）
- `-lr`: 学习率（**仅训练 Mask Decoder，建议 5e-5 或 1e-4**）
- `-weight_decay`: 权重衰减（默认 0.01）
- `-num_workers`: 数据加载线程数（默认 4）

#### 显存优化参数
- `-use_amp`: 启用混合精度训练（**强烈推荐**）
- `--grad_acc_steps`: 梯度累积步数（**建议 4-8**）
  - 有效 batch size = `batch_size × grad_acc_steps × num_gpus`
  - 例如: 4 × 4 × 4 = 64

#### 分布式训练参数
- `--distributed`: 启用分布式训练
- `--world_size`: GPU 总数（例如 4）
- `--node_rank`: 节点排名（单机多卡设为 0）
- `--init_method`: 初始化方法（默认 `env://`）

#### 其他参数
- `--resume`: 从检查点恢复训练
- `-use_wandb`: 使用 Weights & Biases 记录训练

### 显存占用估算

针对 **4x 2080Ti (11GB)**，建议配置：

| 配置 | 每卡 Batch Size | 梯度累积 | 有效 Batch Size | 显存占用 |
|------|----------------|---------|-----------------|---------|
| 保守 | 2 | 8 | 64 | ~8-9 GB |
| 平衡 | 4 | 4 | 64 | ~9-10 GB |
| 激进 | 4 | 8 | 128 | ~9-10 GB |

**注意**: 使用 AMP 可节省约 30-40% 显存。

---

## 推理和评估

### 运行推理

使用训练好的模型在验证集上进行推理和评估：

```bash
python infer_bdd100k.py \
    --val_npy_path data/bdd100k_npy/val \
    --model_checkpoint work_dir/BDD100K-SAM-MaskDecoder-YYYYMMDD-HHMM/medsam_model_best.pth \
    --sam_checkpoint work_dir/SAM/sam_vit_b_01ec64.pth \
    --model_type vit_b \
    --output_dir ./inference_results \
    --batch_size 1 \
    --device cuda:0 \
    --visualize
```

### 参数说明

- `--val_npy_path`: 验证集路径
- `--model_checkpoint`: 微调后的模型检查点
- `--sam_checkpoint`: SAM 预训练权重（用于 Encoder）
- `--output_dir`: 输出目录
- `--visualize`: 可视化前 20 张结果
- `--device`: 推理设备

### 评估指标

推理脚本将输出以下指标：

1. **整体 mIoU**: 所有类别的平均 IoU
2. **每个类别的 IoU**: 19 个类别的详细 IoU

示例输出：
```
评估结果
============================================================
整体 mIoU: 0.6523

每个类别的 IoU:
------------------------------------------------------------
 0. road                : 0.9234
 1. sidewalk            : 0.7821
 2. building            : 0.8456
...
18. bicycle             : 0.5123
============================================================
```

### 可视化结果

可视化结果保存在 `output_dir/visualizations/` 目录下，包含：
- 原始图像
- 真值标注（彩色编码）
- 预测结果（彩色编码）

---

## 训练技巧和注意事项

### 1. 显存优化策略

#### ✅ 启用混合精度训练（AMP）
- 使用 `-use_amp` 参数
- 可节省 30-40% 显存
- 对精度影响极小

#### ✅ 使用梯度累积
- 使用 `--grad_acc_steps` 参数
- 模拟更大的 batch size
- 4-8 步是较好的选择

#### ✅ 冻结编码器
- **Image Encoder 已冻结** - 节省大量显存
- **Prompt Encoder 已冻结** - 保持提示编码稳定
- **仅训练 Mask Decoder** - 减少 90% 可训练参数

#### ⚠️ 调整 Batch Size
- 2080Ti (11GB): 建议 2-4 per GPU
- 3090 (24GB): 可以使用 8-16 per GPU
- 根据实际显存占用调整

### 2. 学习率调整

由于仅训练 Mask Decoder，建议使用较小的学习率：

- **初始学习率**: `5e-5` 或 `1e-4`
- **学习率调度**: 可添加 Cosine Annealing 或 Step Decay
- **Warmup**: 建议前 5-10 个 epoch 使用 warmup

### 3. 数据增强

当前实现包含以下数据增强：
- ✅ 边界框随机偏移（bbox_shift=20）
- ✅ 随机类别采样

可考虑添加：
- 随机水平翻转
- 随机亮度/对比度调整
- 随机裁剪

### 4. 分布式训练注意事项

#### 环境变量设置
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=29500
```

#### 使用 NCCL 后端
- 确保 NCCL 库已正确安装
- NCCL 是 GPU 间通信的最快后端

#### 同步批归一化（可选）
```python
# 在模型定义中使用
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
```

### 5. 监控训练过程

#### 使用 TensorBoard
```bash
# 在训练脚本中添加
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='./logs')
```

#### 使用 Weights & Biases
```bash
# 添加 -use_wandb 参数
python train_bdd100k.py ... -use_wandb True
```

### 6. 常见问题和解决方案

#### Q1: CUDA Out of Memory
**解决方案**:
- 减小 batch size
- 增加梯度累积步数
- 确保启用 AMP
- 减少 num_workers

#### Q2: 训练速度慢
**解决方案**:
- 增加 num_workers
- 使用 SSD 存储数据
- 检查 GPU 利用率（nvidia-smi）

#### Q3: 损失不下降
**解决方案**:
- 检查学习率（可能过大或过小）
- 确认数据预处理正确
- 查看数据加载是否有问题

#### Q4: 分布式训练同步问题
**解决方案**:
- 确保所有节点可以相互访问
- 检查防火墙设置
- 使用 `torch.distributed.barrier()` 同步

---

## 项目文件说明

### 核心文件

1. **`train_bdd100k.py`** - 主训练脚本
   - 支持单卡和多卡训练
   - 冻结 Image Encoder 和 Prompt Encoder
   - 混合精度训练和梯度累积

2. **`pre_bdd100k.py`** - 数据预处理脚本
   - 将 BDD100K 转换为 npy 格式
   - Resize 到 1024×1024
   - 归一化到 [0, 1]

3. **`infer_bdd100k.py`** - 推理和评估脚本
   - 计算 mIoU 和每类 IoU
   - 可视化分割结果

4. **`README_BDD100K.md`** - 本文档

### 代码修改要点

所有修改点在代码中用中文注释标注，主要包括：

1. **修改点 1**: 创建 BDD100K 数据集类
2. **修改点 2**: 冻结 Image Encoder 和 Prompt Encoder
3. **修改点 3**: 调整训练参数适配 2080Ti
4. **修改点 4**: 添加混合精度训练和梯度累积
5. **修改点 5**: 分布式训练参数
6. **修改点 6**: 仅优化 Mask Decoder 参数
7. **修改点 7**: 使用 AMP 混合精度训练
8. **修改点 8**: 梯度累积实现
9. **修改点 9**: 初始化分布式训练
10. **修改点 10**: 使用 DDP 包装模型
11. **修改点 11**: 使用分布式采样器
12. **修改点 12**: DDP + 梯度累积

---

## 实验结果（示例）

### 训练配置

| 参数 | 值 |
|------|-----|
| GPU | 4x 2080Ti (11GB) |
| Batch Size | 4 per GPU |
| 梯度累积 | 4 步 |
| 有效 Batch Size | 64 |
| 学习率 | 5e-5 |
| 训练轮数 | 100 |
| 混合精度 | 是 |

### 预期性能

| 指标 | 值 |
|------|-----|
| 整体 mIoU | 0.60-0.70 |
| 训练时间 | ~12-16 小时 |
| 显存占用 | ~9-10 GB per GPU |

**注意**: 实际结果可能因数据集、超参数和硬件配置而异。

---

## 引用

如果您使用本项目，请引用：

```bibtex
@article{MedSAM,
  title={Segment Anything in Medical Images},
  author={Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  journal={Nature Communications},
  volume={15},
  pages={654},
  year={2024}
}

@inproceedings{bdd100k,
  title={BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
  author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

---

## 致谢

- [Meta AI](https://github.com/facebookresearch/segment-anything) 提供的 Segment Anything 模型
- [BDD100K](https://bdd-data.berkeley.edu/) 数据集团队
- [MedSAM](https://github.com/bowang-lab/MedSAM) 项目提供的基础框架

---

## 许可证

本项目遵循 MIT 许可证。详情请参阅 [LICENSE](../LICENSE) 文件。

---

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件至项目维护者

---

**祝训练顺利！🚀**
