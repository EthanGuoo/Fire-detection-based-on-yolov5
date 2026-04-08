"""
RTX 5090 / A800 优化训练启动脚本
自动检测硬件配置并应用最优参数
支持: RTX 5090 (32GB) | A800-SXM4-40GB (40GB)
"""
import argparse
import subprocess
import sys
from pathlib import Path

import torch


def get_gpu_memory_gb():
    """获取 GPU 显存大小 (GB)"""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def get_optimal_batch_size(model_size, img_size, gpu_memory_gb, gpu_name=""):
    """根据模型大小和 GPU 显存推荐 batch size (仅支持 P6 模型)"""
    # 根据 GPU 类型选择基准配置
    if "A800" in gpu_name or "A100" in gpu_name:
        # A800/A100: 40GB/80GB 专业级 GPU
        base_memory = 40
        # P6 模型 (1280x1280)
        batch_map = {
            'n6': 96,   # nano-P6
            's6': 72,   # small-P6
            'm6': 48,   # medium-P6
            'l6': 32,   # large-P6
            'x6': 24,   # xlarge-P6
        }
    else:
        # RTX 5090: 32GB
        base_memory = 32
        # P6 模型 (1280x1280)
        batch_map = {
            'n6': 64,   # nano-P6
            's6': 48,   # small-P6
            'm6': 32,   # medium-P6
            'l6': 24,   # large-P6
            'x6': 16,   # xlarge-P6
        }

    scale = gpu_memory_gb / base_memory
    base_batch = batch_map.get(model_size, 32)

    # 根据图像尺寸调整
    img_scale = (1280 / img_size) ** 2

    optimal_batch = int(base_batch * scale * img_scale)
    return max(8, optimal_batch // 8 * 8)  # 向下取 8 的倍数


def get_optimal_workers(gpu_name=""):
    """推荐数据加载线程数"""
    import os
    cpu_count = os.cpu_count() or 8

    # A800 数据中心级 GPU 推荐更多线程
    if "A800" in gpu_name or "A100" in gpu_name:
        return min(32, max(16, cpu_count - 2))
    else:
        # RTX 5090 推荐线程数
        return min(24, max(12, cpu_count - 2))


def build_train_command(args):
    """构建训练命令"""
    # 基础命令
    cmd = [sys.executable, "train.py"]

    # 模型配置
    if args.weights:
        cmd.extend(["--weights", args.weights])
    if args.cfg:
        cmd.extend(["--cfg", args.cfg])

    # 数据配置
    cmd.extend(["--data", args.data])
    cmd.extend(["--imgsz", str(args.imgsz)])

    # 训练配置
    cmd.extend(["--epochs", str(args.epochs)])
    cmd.extend(["--batch", str(args.batch)])
    cmd.extend(["--val-batch", str(args.val_batch)])
    cmd.extend(["--workers", str(args.workers)])

    # 超参数配置
    cmd.extend(["--hyp", args.hyp])
    cmd.extend(["--optimizer", args.optimizer])
    cmd.extend(["--lr0", str(args.lr0)])

    # 设备配置
    cmd.extend(["--device", str(args.device)])

    # 缓存策略
    cmd.extend(["--cache", args.cache])

    # 输出配置
    cmd.extend(["--project", args.project])
    cmd.extend(["--name", args.name])

    # 优化选项
    if args.amp:
        cmd.append("--amp")
    if args.ema:
        cmd.append("--ema")
    if args.cos_lr:
        cmd.append("--cos-lr")
    if args.multi_scale:
        cmd.append("--multi-scale")
    if args.mixup:
        cmd.append("--mixup")
    if args.rect:
        cmd.append("--rect")
    if args.image_weights:
        cmd.append("--image-weights")

    # 其他选项
    cmd.extend(["--patience", str(args.patience)])
    cmd.extend(["--seed", str(args.seed)])

    if args.exist_ok:
        cmd.append("--exist-ok")
    if args.verbose:
        cmd.append("--verbose")

    return cmd


def print_system_info():
    """打印系统信息"""
    print("=" * 80)
    print("RTX 5090 / A800 优化训练脚本")
    print("=" * 80)

    # PyTorch 信息
    print(f"\n[系统信息]")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"  cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"  GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 显存: {get_gpu_memory_gb():.1f} GB")
        print(f"  GPU 数量: {torch.cuda.device_count()}")

        # 检测 GPU 类型
        gpu_name = torch.cuda.get_device_name(0)
        if "5090" in gpu_name:
            print(f"  ✓ 检测到 RTX 5090 (32GB)，将使用优化配置!")
        elif "A800" in gpu_name:
            print(f"  ✓ 检测到 A800 (40GB)，将使用数据中心级优化配置!")
        elif "A100" in gpu_name:
            print(f"  ✓ 检测到 A100，将使用数据中心级优化配置!")
        else:
            print(f"  ⚠ 未检测到 RTX 5090/A800，配置可能不是最优")

    # CPU 信息
    import os
    print(f"\n[CPU 信息]")
    print(f"  CPU 核心数: {os.cpu_count()}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="RTX 5090 / A800 优化训练启动脚本")

    # 配置预设 (P6 模型仅支持 optimal)
    parser.add_argument(
        "--preset",
        type=str,
        choices=["optimal"],
        default="optimal",
        help="配置预设 (P6 模型使用优化配置)"
    )

    # 模型配置
    parser.add_argument("--weights", type=str, default="yolov5m6.pt", help="预训练权重 (P6 模型)")
    parser.add_argument("--cfg", type=str, default="", help="模型配置文件")
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["n6", "s6", "m6", "l6", "x6"],
        default="m6",
        help="模型大小 (P6 模型，支持 1280x1280 高分辨率输入)"
    )

    # 数据配置
    parser.add_argument("--data", type=str, default="data/coco128.yaml", help="数据集配置")
    parser.add_argument("--imgsz", type=int, default=0, help="图像尺寸 (0=自动: P5用640, P6用1280)")

    # 训练配置
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--batch", type=int, default=-1, help="batch size (-1 自动)")
    parser.add_argument("--val-batch", type=int, default=-1, help="验证 batch size (-1 自动)")
    parser.add_argument("--workers", type=int, default=-1, help="数据加载线程 (-1 自动)")

    # 超参数配置
    parser.add_argument("--hyp", type=str, default="", help="超参数文件 (留空使用预设)")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["SGD", "Adam", "AdamW"])
    parser.add_argument("--lr0", type=float, default=0.025, help="初始学习率")

    # 设备配置
    parser.add_argument("--device", type=str, default="0", help="训练设备")

    # 缓存策略
    parser.add_argument("--cache", type=str, default="ram", choices=["ram", "disk", "val", "False"])

    # 输出配置
    parser.add_argument("--project", type=str, default="runs/train", help="项目目录")
    parser.add_argument("--name", type=str, default="rtx5090_exp", help="实验名称")
    parser.add_argument("--exist-ok", action="store_true", help="覆盖已存在的实验")

    # 优化选项
    parser.add_argument("--amp", action="store_true", default=True, help="混合精度训练")
    parser.add_argument("--ema", action="store_true", default=True, help="EMA")
    parser.add_argument("--cos-lr", action="store_true", default=True, help="余弦学习率")
    parser.add_argument("--multi-scale", action="store_true", help="多尺度训练")
    parser.add_argument("--mixup", action="store_true", help="Mixup 增强")
    parser.add_argument("--rect", action="store_true", help="矩形训练")
    parser.add_argument("--image-weights", action="store_true", help="图像加权采样")

    # 其他选项
    parser.add_argument("--patience", type=int, default=100, help="早停耐心值")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--dry-run", action="store_true", help="仅显示命令不执行")

    args = parser.parse_args()

    # 打印系统信息
    print_system_info()

    # 自动配置
    gpu_memory = get_gpu_memory_gb()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""

    # 根据 GPU 类型选择配置文件前缀
    if "A800" in gpu_name or "A100" in gpu_name:
        config_prefix = "a800"
    else:
        config_prefix = "rtx5090"

    # 自动设置图像尺寸 (P6 模型默认 1280)
    if args.imgsz <= 0:
        args.imgsz = 1280
        print(f"[自动配置] P6 模型，图像尺寸: {args.imgsz}")

    # 自动选择超参数文件 (仅 P6 配置)
    if not args.hyp:
        args.hyp = f"configs/{config_prefix}_optimal.yaml"
        print(f"[自动配置] 使用超参数文件: {args.hyp}")

    # 自动计算 batch size
    if args.batch <= 0:
        args.batch = get_optimal_batch_size(args.model_size, args.imgsz, gpu_memory, gpu_name)
        print(f"[自动配置] 训练 batch size: {args.batch}")

    # 自动计算验证 batch size
    if args.val_batch <= 0:
        args.val_batch = args.batch * 2
        print(f"[自动配置] 验证 batch size: {args.val_batch}")

    # 自动计算 workers
    if args.workers <= 0:
        args.workers = get_optimal_workers(gpu_name)
        print(f"[自动配置] 数据加载线程: {args.workers}")

    # 根据预设调整参数
    if args.preset == "extreme":
        args.optimizer = "AdamW"
        args.lr0 = 0.03
        args.multi_scale = False  # 极速模式不使用多尺度
        args.rect = True
        print(f"[预设: extreme] 追求最大训练速度")
    elif args.preset == "precision":
        args.optimizer = "AdamW"
        args.lr0 = 0.02
        args.multi_scale = True
        args.mixup = True
        args.image_weights = True
        args.patience = 100
        print(f"[预设: precision] 追求最高模型精度")
    else:  # optimal
        args.optimizer = "AdamW"
        args.lr0 = 0.025
        print(f"[预设: optimal] 平衡速度与精度")

    # 构建训练命令
    cmd = build_train_command(args)

    # 打印命令
    print("\n[训练命令]")
    print(" ".join(cmd))
    print("\n" + "=" * 80 + "\n")

    # 执行训练
    if args.dry_run:
        print("[Dry Run] 仅显示命令，不执行训练")
        return

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[错误] 训练失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[中断] 用户取消训练")
        sys.exit(0)


if __name__ == "__main__":
    main()
