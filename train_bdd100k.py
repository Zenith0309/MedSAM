# -*- coding: utf-8 -*-
"""
在 BDD100K 数据集上微调 SAM 的 Mask Decoder
仅训练 Mask Decoder，冻结 Image Encoder 和 Prompt Encoder
支持多 GPU 分布式训练，针对 4x2080Ti (11GB) 优化
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob

# 设置随机种子
torch.manual_seed(2023)
torch.cuda.empty_cache()


def show_mask(mask, ax, random_color=False):
    """可视化分割掩码"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    """可视化边界框"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


# ========== 修改点 1: 创建 BDD100K 数据集类 ==========
# 适配 BDD100K 语义分割数据格式
class BDD100KDataset(Dataset):
    """
    BDD100K 数据集加载器
    支持语义分割标注，每个样本随机选择一个类别进行训练
    """
    def __init__(self, data_root, bbox_shift=20, num_classes=19):
        """
        参数:
            data_root: 数据根目录，包含 imgs 和 gts 子文件夹
            bbox_shift: 边界框随机偏移量（数据增强）
            num_classes: BDD100K 类别数量（默认19个）
        """
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        # 确保图像和标注文件一一对应
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        self.num_classes = num_classes
        print(f"数据集中的图像数量: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # 加载 npy 图像 (1024, 1024, 3), 范围 [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), allow_pickle=True
        )  # (1024, 1024, 3)
        
        # 转换形状为 (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "图像应归一化到 [0, 1]"
        
        # 加载语义分割标注 (1024, 1024)，类别ID: 0-18
        gt = np.load(
            self.gt_path_files[index], allow_pickle=True
        )  # (1024, 1024)
        
        assert img_name == os.path.basename(self.gt_path_files[index]), \
            "图像和标注文件名不匹配"
        
        # 获取所有存在的类别（排除背景类别255或其他无效值）
        label_ids = np.unique(gt)
        # 过滤掉背景和无效标签（假设背景为255或负值）
        label_ids = label_ids[(label_ids >= 0) & (label_ids < self.num_classes)]
        
        if len(label_ids) == 0:
            # 如果没有有效标签，使用全零掩码和默认边界框
            gt2D = np.zeros_like(gt, dtype=np.uint8)
            bboxes = np.array([0, 0, gt.shape[1]-1, gt.shape[0]-1])
        else:
            # 随机选择一个类别进行训练（SAM 单对象分割）
            selected_label = random.choice(label_ids.tolist())
            gt2D = np.uint8(gt == selected_label)  # 二值掩码 (1024, 1024)
            
            # 计算该类别的边界框
            y_indices, x_indices = np.where(gt2D > 0)
            if len(x_indices) == 0 or len(y_indices) == 0:
                # 如果掩码为空，使用默认边界框
                bboxes = np.array([0, 0, gt.shape[1]-1, gt.shape[0]-1])
            else:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                # 添加随机偏移（数据增强）
                H, W = gt2D.shape
                x_min = max(0, x_min - random.randint(0, self.bbox_shift))
                x_max = min(W, x_max + random.randint(0, self.bbox_shift))
                y_min = max(0, y_min - random.randint(0, self.bbox_shift))
                y_max = min(H, y_max + random.randint(0, self.bbox_shift))
                bboxes = np.array([x_min, y_min, x_max, y_max])
        
        assert np.max(gt2D) <= 1 and np.min(gt2D) >= 0, "真值应为 0 或 1"
        
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )


# ========== 修改点 2: 修改 MedSAM 模型类 ==========
# 冻结 Image Encoder 和 Prompt Encoder，仅训练 Mask Decoder
class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        
        # ========== 修改: 冻结 Image Encoder ==========
        # 原始 train_one_gpu.py 训练 Image Encoder，这里冻结以节省显存
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        print("Image Encoder 已冻结")
        
        # ========== 修改: 冻结 Prompt Encoder ==========
        # 与原始代码一致，冻结 Prompt Encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        print("Prompt Encoder 已冻结")

    def forward(self, image, box):
        # ========== 修改: Image Encoder 不计算梯度 ==========
        with torch.no_grad():
            image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        
        # Prompt Encoder 不计算梯度（与原始代码一致）
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        
        # ========== 只有 Mask Decoder 计算梯度 ==========
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


# %% 设置命令行参数
parser = argparse.ArgumentParser(description='在 BDD100K 上微调 SAM Mask Decoder')
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="data/bdd100k_npy/train",
    help="训练 npy 文件路径；包含两个子文件夹: gts 和 imgs",
)
parser.add_argument("-task_name", type=str, default="BDD100K-SAM-MaskDecoder")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth",
    help="SAM 预训练权重路径"
)
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="加载预训练模型"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")

# ========== 修改点 3: 调整训练参数以适配 2080Ti ==========
parser.add_argument("-num_epochs", type=int, default=100)
parser.add_argument(
    "-batch_size", type=int, default=4,
    help="每个 GPU 的 batch size（2080Ti 建议 2-4）"
)
parser.add_argument("-num_workers", type=int, default=4)

# 优化器参数
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="权重衰减 (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=5e-5, metavar="LR",
    help="学习率 (仅训练 Mask Decoder，使用较小学习率)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="使用 wandb 监控训练"
)

# ========== 修改点 4: 添加混合精度训练和梯度累积 ==========
parser.add_argument(
    "-use_amp", action="store_true", default=True,
    help="使用混合精度训练（AMP）节省显存"
)
parser.add_argument(
    "--grad_acc_steps",
    type=int,
    default=4,
    help="梯度累积步数（有效 batch size = batch_size * grad_acc_steps * num_gpus）"
)

# ========== 修改点 5: 分布式训练参数 ==========
parser.add_argument("--world_size", type=int, default=4, help="GPU 总数（默认4）")
parser.add_argument("--node_rank", type=int, default=0, help="节点排名")
parser.add_argument(
    "--bucket_cap_mb",
    type=int,
    default=25,
    help="DDP 梯度通信的内存限制（MB）"
)
parser.add_argument(
    "--resume", type=str, default="", help="从检查点恢复训练"
)
parser.add_argument("--init_method", type=str, default="env://")
parser.add_argument("--device", type=str, default="cuda:0", help="单卡训练时使用的设备")
parser.add_argument(
    "--distributed", action="store_true", default=False,
    help="是否使用分布式训练（多卡）"
)

args = parser.parse_args()

if args.use_wandb:
    import wandb
    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
            "grad_acc_steps": args.grad_acc_steps,
            "use_amp": args.use_amp,
        },
    )

run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)


# ========== 单卡训练主函数 ==========
def train_single_gpu():
    """单 GPU 训练函数"""
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    device = torch.device(args.device)
    
    # 加载 SAM 模型
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.train()

    print("=" * 60)
    print(
        "总参数数量: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )
    print(
        "可训练参数数量: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )
    print("=" * 60)

    # ========== 修改点 6: 仅优化 Mask Decoder 参数 ==========
    mask_dec_params = list(medsam_model.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(
        mask_dec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Mask Decoder 可训练参数数量: ",
        sum(p.numel() for p in mask_dec_params if p.requires_grad),
    )
    
    # 损失函数
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    
    # 加载数据集
    num_epochs = args.num_epochs
    losses = []
    best_loss = 1e10
    
    # ========== 使用 BDD100K 数据集 ==========
    train_dataset = BDD100KDataset(args.tr_npy_path)
    print("训练样本数量: ", len(train_dataset))
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        medsam_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"从 epoch {start_epoch} 恢复训练")
    
    # ========== 修改点 7: 使用 AMP 混合精度训练 ==========
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("使用混合精度训练（AMP）")

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            
            if args.use_amp:
                # 混合精度训练
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                        medsam_pred, gt2D.float()
                    )
                
                # ========== 修改点 8: 梯度累积 ==========
                loss = loss / args.grad_acc_steps
                scaler.scale(loss).backward()
                
                if (step + 1) % args.grad_acc_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                medsam_pred = medsam_model(image, boxes_np)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                
                # 梯度累积
                loss = loss / args.grad_acc_steps
                loss.backward()
                
                if (step + 1) % args.grad_acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_acc_steps

        epoch_loss /= (step + 1)
        losses.append(epoch_loss)
        
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        
        print(
            f'时间: {datetime.now().strftime("%Y%m%d-%H%M")}, '
            f'Epoch: {epoch}, Loss: {epoch_loss:.4f}'
        )
        
        # 保存最新模型
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        
        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))
            print(f"保存最佳模型，损失: {best_loss:.4f}")

        # 绘制损失曲线
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "_train_loss.png"))
        plt.close()


# ========== 多卡训练主函数 ==========
def main():
    """主函数：根据参数选择单卡或多卡训练"""
    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        print(f"检测到 {ngpus_per_node} 个 GPU，开始分布式训练")
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        print("单 GPU 训练模式")
        train_single_gpu()


def main_worker(gpu, ngpus_per_node, args):
    """多 GPU 训练的 worker 函数"""
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = args.world_size
    print(f"[Rank {rank}]: 使用 GPU: {gpu} 进行训练")
    is_main_host = rank == 0
    
    if is_main_host:
        os.makedirs(model_save_path, exist_ok=True)
        shutil.copyfile(
            __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
        )
    
    torch.cuda.set_device(gpu)
    
    # ========== 修改点 9: 初始化分布式训练 ==========
    torch.distributed.init_process_group(
        backend="nccl", init_method=args.init_method, rank=rank, world_size=world_size
    )

    # 加载模型
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder
    ).cuda()
    
    # 显存信息
    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[1] / (1024**3)
    print(f"[RANK {rank}: GPU {gpu}] DDP 初始化前: 总显存 {total_cuda_mem:.2f} GB, "
          f"空闲 {free_cuda_mem:.2f} GB")

    # ========== 修改点 10: 使用 DDP 包装模型 ==========
    medsam_model = nn.parallel.DistributedDataParallel(
        medsam_model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=False,  # 设置为 False 因为我们只训练 Mask Decoder
        bucket_cap_mb=args.bucket_cap_mb,
    )

    medsam_model.train()

    if is_main_host:
        print("=" * 60)
        print(
            "总参数数量: ",
            sum(p.numel() for p in medsam_model.parameters()),
        )
        print(
            "可训练参数数量: ",
            sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
        )
        print("=" * 60)

    # ========== 仅优化 Mask Decoder ==========
    mask_dec_params = list(medsam_model.module.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(
        mask_dec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    
    if is_main_host:
        print(
            "Mask Decoder 可训练参数数量: ",
            sum(p.numel() for p in mask_dec_params if p.requires_grad),
        )
    
    # 损失函数
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    
    # 数据集
    num_epochs = args.num_epochs
    losses = []
    best_loss = 1e10
    
    train_dataset = BDD100KDataset(args.tr_npy_path)
    # ========== 修改点 11: 使用分布式采样器 ==========
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    if is_main_host:
        print("训练样本数量: ", len(train_dataset))
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        if is_main_host:
            print(f"=> 加载检查点 '{args.resume}'")
        loc = f"cuda:{gpu}"
        checkpoint = torch.load(args.resume, map_location=loc)
        start_epoch = checkpoint["epoch"] + 1
        medsam_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if is_main_host:
            print(f"=> 从 epoch {start_epoch} 恢复训练")
    
    torch.distributed.barrier()

    # ========== 使用 AMP ==========
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        if is_main_host:
            print("使用混合精度训练（AMP）")

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        train_dataloader.sampler.set_epoch(epoch)
        
        for step, (image, gt2D, boxes, _) in enumerate(
            tqdm(train_dataloader, desc=f"[RANK {rank}: GPU {gpu}]", disable=not is_main_host)
        ):
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.cuda(), gt2D.cuda()
            
            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                        medsam_pred, gt2D.float()
                    )
                
                # ========== 修改点 12: DDP + 梯度累积 ==========
                loss = loss / args.grad_acc_steps
                scaler.scale(loss).backward()
                
                if (step + 1) % args.grad_acc_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                medsam_pred = medsam_model(image, boxes_np)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                    medsam_pred, gt2D.float()
                )
                
                # 梯度累积
                loss = loss / args.grad_acc_steps
                
                if (step + 1) % args.grad_acc_steps == 0:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    # 不同步梯度
                    with medsam_model.no_sync():
                        loss.backward()

            epoch_loss += loss.item() * args.grad_acc_steps

        epoch_loss /= (step + 1)
        losses.append(epoch_loss)
        
        if args.use_wandb and is_main_host:
            wandb.log({"epoch_loss": epoch_loss})
        
        if is_main_host:
            print(
                f'时间: {datetime.now().strftime("%Y%m%d-%H%M")}, '
                f'Epoch: {epoch}, Loss: {epoch_loss:.4f}'
            )
        
        # 保存模型（仅主进程）
        if is_main_host:
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))
                print(f"保存最佳模型，损失: {best_loss:.4f}")
        
        torch.distributed.barrier()

        # 绘制损失曲线（仅主进程）
        if is_main_host:
            plt.plot(losses)
            plt.title("Dice + Cross Entropy Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(join(model_save_path, args.task_name + "_train_loss.png"))
            plt.close()


if __name__ == "__main__":
    main()
