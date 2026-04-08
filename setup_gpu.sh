#!/bin/bash
# YOLOv5 GPU 训练环境配置脚本
# 支持所有 NVIDIA GPU (RTX 系列、GTX 系列、Tesla 系列等)
# 用法: bash setup_gpu.sh

set -e

echo "=========================================="
echo "YOLOv5 GPU 训练环境配置"
echo "=========================================="
echo ""

# 检查 Python 版本
echo "[1/5] 检查 Python 版本..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"

required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ 错误: 需要 Python >= 3.8"
    exit 1
fi
echo "✅ Python 版本满足要求"
echo ""

# 检查 CUDA
echo "[2/5] 检查 CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo "✅ NVIDIA 驱动已安装"
else
    echo "⚠️  警告: nvidia-smi 不可用，请确认 NVIDIA 驱动已正确安装"
fi
echo ""

# 安装核心依赖
echo "[3/5] 安装核心依赖..."
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# PyTorch (CUDA 12.1) - 使用官方源，因为清华源可能没有 CUDA 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# YOLOv5 依赖 - 使用清华源加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "✅ 核心依赖安装完成"
echo ""

# 安装监控工具依赖
echo "[4/5] 安装监控工具依赖..."
pip install nvidia-ml-py3 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "✅ 监控工具依赖安装完成"
echo ""

# 验证安装
echo "[5/5] 验证安装..."
python -c "
import torch
import sys

print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'cuDNN 版本: {torch.backends.cudnn.version()}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
    print('')
    print('检测到的 GPU 设备:')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'         显存: {mem_gb:.1f} GB')
    print(f'✅ GPU 环境配置成功!')
else:
    print('❌ CUDA 不可用，请检查安装')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 环境配置完成!"
    echo "=========================================="
    echo ""
    echo "快速开始训练:"
    echo "  python train.py --data data/custom.yaml --weights yolov5s.pt --batch-size 16 --epochs 100"
    echo ""
    echo "运行检测:"
    echo "  python detect.py --weights runs/train/exp/weights/best.pt --source data/test/images"
    echo ""
    echo "查看使用指南:"
    echo "  cat 使用指南.md"
    echo ""
else
    echo ""
    echo "❌ 环境配置失败，请检查错误信息"
    exit 1
fi
