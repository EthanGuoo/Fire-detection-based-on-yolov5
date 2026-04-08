"""
RTX 5090 训练性能监控脚本
实时显示 GPU 利用率、显存占用、训练速度等关键指标
"""
import time
import subprocess
import sys
from pathlib import Path

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("警告: pynvml 未安装，部分功能不可用")
    print("安装命令: pip install nvidia-ml-py3")


class GPUMonitor:
    def __init__(self, device_id=0):
        self.device_id = device_id
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self.enabled = True
            except Exception as e:
                print(f"NVML 初始化失败: {e}")
                self.enabled = False
        else:
            self.enabled = False

    def get_info(self):
        """获取 GPU 信息"""
        if not self.enabled:
            return None

        try:
            # 温度
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)

            # 利用率
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = util.gpu
            mem_util = util.memory

            # 显存
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            mem_used = mem_info.used / 1024**3  # GB
            mem_total = mem_info.total / 1024**3  # GB
            mem_percent = (mem_used / mem_total) * 100

            # 功耗
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            power_w = power_mw / 1000

            # 时钟频率
            clock_sm = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
            clock_mem = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)

            return {
                'temp': temp,
                'gpu_util': gpu_util,
                'mem_util': mem_util,
                'mem_used': mem_used,
                'mem_total': mem_total,
                'mem_percent': mem_percent,
                'power': power_w,
                'clock_sm': clock_sm,
                'clock_mem': clock_mem,
            }
        except Exception as e:
            print(f"获取 GPU 信息失败: {e}")
            return None

    def __del__(self):
        if NVML_AVAILABLE and self.enabled:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


def parse_training_log(log_file):
    """解析训练日志获取性能指标"""
    if not Path(log_file).exists():
        return None

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 查找最后一行包含训练信息的日志
        for line in reversed(lines):
            if 'Epoch' in line and 'mem=' in line:
                # 解析: Epoch 1/300 mem=22.5G box=0.0500 obj=0.0700 cls=0.0300 img=640
                parts = line.split()
                info = {}

                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        info[key] = value

                return info

        return None
    except Exception as e:
        print(f"解析日志失败: {e}")
        return None


def print_dashboard(gpu_info, train_info=None):
    """打印监控面板"""
    print("\033[2J\033[H", end="")  # 清屏

    print("=" * 80)
    print("RTX 5090 训练性能监控".center(80))
    print("=" * 80)
    print()

    # GPU 信息
    if gpu_info:
        print(f"[GPU 状态]")
        print(f"  温度:       {gpu_info['temp']}°C")
        print(f"  GPU 利用率: {gpu_info['gpu_util']}%")
        print(f"  显存利用率: {gpu_info['mem_util']}%")
        print(f"  显存占用:   {gpu_info['mem_used']:.1f} / {gpu_info['mem_total']:.1f} GB ({gpu_info['mem_percent']:.1f}%)")
        print(f"  功耗:       {gpu_info['power']:.0f} W")
        print(f"  SM 频率:    {gpu_info['clock_sm']} MHz")
        print(f"  显存频率:   {gpu_info['clock_mem']} MHz")

        # 性能条
        print()
        print(f"  GPU: [{'█' * int(gpu_info['gpu_util']/5):<20}] {gpu_info['gpu_util']}%")
        print(f"  MEM: [{'█' * int(gpu_info['mem_percent']/5):<20}] {gpu_info['mem_percent']:.1f}%")
    else:
        print("[GPU 状态] 无法获取 (需要安装 nvidia-ml-py3)")

    print()

    # 训练信息
    if train_info:
        print(f"[训练状态]")
        epoch = train_info.get('Epoch', 'N/A')
        mem = train_info.get('mem', 'N/A')
        box = train_info.get('box', 'N/A')
        obj = train_info.get('obj', 'N/A')
        cls = train_info.get('cls', 'N/A')
        img = train_info.get('img', 'N/A')

        print(f"  当前轮次:   {epoch}")
        print(f"  显存占用:   {mem}")
        print(f"  Box Loss:   {box}")
        print(f"  Obj Loss:   {obj}")
        print(f"  Cls Loss:   {cls}")
        print(f"  图像尺寸:   {img}")
    else:
        print(f"[训练状态] 未检测到训练日志")

    print()

    # 性能建议
    if gpu_info:
        print(f"[性能建议]")

        if gpu_info['gpu_util'] < 70:
            print(f"  ⚠️  GPU 利用率较低 ({gpu_info['gpu_util']}%)")
            print(f"      建议: 增加 batch size 或 workers 数量")

        if gpu_info['mem_percent'] < 60:
            print(f"  💡 显存占用较低 ({gpu_info['mem_percent']:.1f}%)")
            print(f"      建议: 可以增加 batch size 以充分利用显存")

        if gpu_info['mem_percent'] > 90:
            print(f"  ⚠️  显存占用过高 ({gpu_info['mem_percent']:.1f}%)")
            print(f"      警告: 可能面临 OOM 风险，建议降低 batch size")

        if gpu_info['temp'] > 80:
            print(f"  🔥 GPU 温度较高 ({gpu_info['temp']}°C)")
            print(f"      建议: 检查散热，可能需要降低功耗限制")

        if gpu_info['gpu_util'] >= 95 and gpu_info['mem_percent'] >= 80:
            print(f"  ✅ GPU 利用率优秀! 当前配置已接近最优")

    print()
    print("=" * 80)
    print(f"按 Ctrl+C 退出监控")
    print("=" * 80)


def monitor(device_id=0, log_file=None, interval=2):
    """主监控循环"""
    monitor = GPUMonitor(device_id)

    print(f"开始监控 GPU {device_id}...")
    if log_file:
        print(f"监控日志文件: {log_file}")
    print(f"刷新间隔: {interval} 秒")
    print()

    try:
        while True:
            gpu_info = monitor.get_info()
            train_info = parse_training_log(log_file) if log_file else None

            print_dashboard(gpu_info, train_info)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n监控已停止")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RTX 5090 训练性能监控")
    parser.add_argument("--device", type=int, default=0, help="GPU 设备 ID")
    parser.add_argument("--log", type=str, default=None, help="训练日志文件路径")
    parser.add_argument("--interval", type=float, default=2.0, help="刷新间隔 (秒)")

    args = parser.parse_args()

    monitor(args.device, args.log, args.interval)


if __name__ == "__main__":
    main()
