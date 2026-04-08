#!/bin/bash
# 修复 NumPy 和 PySide6 兼容性问题
# 用法: bash fix_dependencies.sh

set -e

echo "=========================================="
echo "修复依赖兼容性问题"
echo "=========================================="
echo ""

echo "[1/3] 卸载不兼容的 NumPy 2.x..."
pip uninstall numpy -y

echo ""
echo "[2/3] 安装兼容的 NumPy 1.x..."
pip install "numpy>=1.18.5,<2.0.0" -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "[3/3] 重新安装 PySide6 和 shiboken6..."
pip uninstall PySide6 shiboken6 -y
pip install "PySide6>=6.8.0" -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "=========================================="
echo "验证安装..."
echo "=========================================="
python -c "
import numpy
import PySide6

print(f'NumPy: {numpy.__version__}')
print(f'PySide6: {PySide6.__version__}')
print('✅ 修复成功!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 修复完成! 现在可以运行:"
    echo "  python wzq.py"
    echo "=========================================="
else
    echo ""
    echo "❌ 修复失败，请手动检查错误信息"
    exit 1
fi
