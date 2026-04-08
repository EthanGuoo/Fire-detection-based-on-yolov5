# PowerShell 脚本：修复 NumPy 和 PySide6 兼容性问题
# 用法: .\fix_dependencies.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "修复依赖兼容性问题" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/4] 检查当前版本..." -ForegroundColor Yellow
python -c "import numpy; print(f'当前 NumPy: {numpy.__version__}')" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "检测到 NumPy 已安装" -ForegroundColor Green
}

Write-Host ""
Write-Host "[2/4] 卸载不兼容的 NumPy 2.x..." -ForegroundColor Yellow
pip uninstall numpy -y

Write-Host ""
Write-Host "[3/4] 安装兼容的 NumPy 1.x..." -ForegroundColor Yellow
pip install "numpy>=1.18.5,<2.0.0" -i https://pypi.tuna.tsinghua.edu.cn/simple

Write-Host ""
Write-Host "[4/4] 重新安装 PySide6 和 shiboken6..." -ForegroundColor Yellow
pip uninstall PySide6 shiboken6 -y
pip install "PySide6>=6.8.0" -i https://pypi.tuna.tsinghua.edu.cn/simple

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "验证安装..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$result = python -c @"
import numpy
import PySide6

print(f'NumPy: {numpy.__version__}')
print(f'PySide6: {PySide6.__version__}')
"@ 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host $result -ForegroundColor Green
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "✓ 修复完成! 现在可以运行:" -ForegroundColor Green
    Write-Host "  python wzq.py" -ForegroundColor White
    Write-Host "==========================================" -ForegroundColor Green
} else {
    Write-Host $result -ForegroundColor Red
    Write-Host ""
    Write-Host "× 修复失败，请检查错误信息" -ForegroundColor Red
}

Write-Host ""
Write-Host "按任意键退出..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
