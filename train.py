import argparse
import csv
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.dataloaders import create_dataloader
from utils.general import (
    check_dataset,
    check_file,
    check_img_size,
    colorstr,
    increment_path,
    init_seeds,
    labels_to_class_weights,
    methods,
    print_args,
    strip_optimizer,
)
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_images, plot_labels, plot_results
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_optimizer,
    smart_resume,
)
import val as validate


def safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return torch.load(*args, **kwargs)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def seed_everything(seed=0):
    init_seeds(seed, deterministic=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_results_csv(csv_path, epoch, train_loss_items, results, lr):
    headers = [
        "epoch",
        "train/box_loss",
        "train/obj_loss",
        "train/cls_loss",
        "metrics/precision",
        "metrics/recall",
        "metrics/mAP_0.5",
        "metrics/mAP_0.5:0.95",
        "val/box_loss",
        "val/obj_loss",
        "val/cls_loss",
        "x/lr0",
        "x/lr1",
        "x/lr2",
    ]
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)

        p, r, map50, map5095, vbox, vobj, vcls = results
        row = [
            epoch,
            float(train_loss_items[0]),
            float(train_loss_items[1]),
            float(train_loss_items[2]),
            float(p),
            float(r),
            float(map50),
            float(map5095),
            float(vbox),
            float(vobj),
            float(vcls),
            float(lr[0] if len(lr) > 0 else 0.0),
            float(lr[1] if len(lr) > 1 else 0.0),
            float(lr[2] if len(lr) > 2 else (lr[-1] if len(lr) else 0.0)),
        ]
        writer.writerow(row)


def build_optimizer(model, opt):
    if opt.optimizer.lower() == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=opt.lr0,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
            nesterov=True,
        )
    elif opt.optimizer.lower() == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=opt.lr0,
            betas=(opt.momentum, 0.999),
            weight_decay=opt.weight_decay,
        )
    elif opt.optimizer.lower() == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=opt.lr0,
            betas=(opt.momentum, 0.999),
            weight_decay=opt.weight_decay,
        )
    else:
        raise ValueError(f"不支持的 optimizer: {opt.optimizer}")
    return optimizer


def build_scheduler(optimizer, opt, epochs):
    if opt.cos_lr:
        lf = one_cycle(1, opt.lrf, epochs)
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - opt.lrf) + opt.lrf
    scheduler = LambdaLR(optimizer, lr_lambda=lf)
    return scheduler, lf


def load_pretrained(model, weights, device, nc, names):
    ckpt = safe_torch_load(weights, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        if hasattr(ckpt["model"], "yaml"):
            yaml_cfg = ckpt["model"].yaml
        else:
            yaml_cfg = model.yaml
        pretrained_state = ckpt["model"].float().state_dict()
    else:
        raise ValueError("加载权重失败：pt 文件中未找到 model")

    model_state = model.state_dict()
    filtered_state = {
        k: v for k, v in pretrained_state.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    model.load_state_dict(filtered_state, strict=False)

    model.nc = nc
    model.names = names

    print(f"成功加载 {len(filtered_state)}/{len(model_state)} 个匹配参数，已自动跳过不匹配层")
    return ckpt


def save_checkpoint(path, epoch, best_fitness, model, ema, optimizer, opt):
    ckpt = {
        "epoch": epoch,
        "best_fitness": best_fitness,
        "model": deepcopy(de_parallel(ema.ema if ema else model)).half(),
        "ema": deepcopy(ema.ema).half() if ema else None,
        "updates": ema.updates if ema else 0,
        "optimizer": optimizer.state_dict(),
        "opt": vars(opt),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(ckpt, path)


def train(opt):
    seed_everything(opt.seed)

    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    last_path = wdir / "last.pt"
    best_path = wdir / "best.pt"
    results_csv = save_dir / "results.csv"

    device = select_device(opt.device, batch_size=opt.batch if opt.batch > 0 else 16)
    cuda = device.type != "cpu"
    device_name = torch.cuda.get_device_name(device) if cuda else "CPU"
    is_rtx_5090 = "5090" in device_name

    if cuda:
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    data_dict = check_dataset(opt.data)
    train_path = data_dict["train"]
    val_path = data_dict["val"]
    if opt.single_cls:
        nc = 1
        names = ["item"]
    else:
        nc = int(data_dict["nc"])
        names = data_dict["names"]
        assert len(names) == nc, f"names 数量 {len(names)} 与 nc={nc} 不一致"

    imgsz = check_img_size(opt.imgsz, s=32)
    model_ckpt = None
    model_yaml = None
    if opt.cfg:
        model_yaml = check_file(opt.cfg)
    elif opt.weights and str(opt.weights).endswith(".pt"):
        model_ckpt = safe_torch_load(opt.weights, map_location="cpu")
        model_yaml = model_ckpt["model"].yaml if "model" in model_ckpt else None
    if model_yaml is None:
        raise ValueError("请提供 --cfg 或包含 model.yaml 的 --weights")

    model = Model(model_yaml, ch=3, nc=nc).to(device)
    if opt.weights and str(opt.weights).endswith(".pt"):
        _ = load_pretrained(model, opt.weights, device, nc, names)

    if opt.freeze:
        freeze = [f"model.{x}." for x in opt.freeze]
        for k, v in model.named_parameters():
            if any(f in k for f in freeze):
                v.requires_grad = False

    hyp = {
        "lr0": opt.lr0,
        "lrf": opt.lrf,
        "momentum": opt.momentum,
        "weight_decay": opt.weight_decay,
        "warmup_epochs": opt.warmup_epochs,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 0.05,
        "cls": 0.5,
        "obj": 1.0,
        "cls_pw": 1.0,
        "obj_pw": 1.0,
        "fl_gamma": 0.0,
        "label_smoothing": opt.label_smoothing,
        "anchor_t": 4.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0 if not opt.no_mosaic else 0.0,
        "mixup": 0.1 if opt.mixup else 0.0,
        "copy_paste": 0.0,
    }
    if opt.hyp:
        with open(check_file(opt.hyp), "r", encoding="utf-8") as f:
            user_hyp = yaml.safe_load(f) or {}
        hyp.update(user_hyp)

    model.hyp = hyp
    model.nc = nc
    model.names = names

    cpu_count = os.cpu_count() or 8
    if opt.workers <= 0:
        if is_rtx_5090:
            opt.workers = min(20, max(8, cpu_count - 2))
        else:
            opt.workers = min(8, max(1, cpu_count - 1))

    batch_size = opt.batch
    if batch_size <= 0:
        if is_rtx_5090:
            batch_size = 64
        else:
            batch_size = 32 if cuda else 8
        print(f"未指定有效 batch，自动使用 batch={batch_size}")

    if opt.val_batch <= 0:
        if is_rtx_5090:
            opt.val_batch = max(64, batch_size * 2)
        else:
            opt.val_batch = max(batch_size, 8)

    if is_rtx_5090:
        print(f"检测到 RTX 5090，启用高吞吐默认参数：batch={batch_size}, val_batch={opt.val_batch}, workers={opt.workers}")

    train_loader, train_dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size,
        32,
        single_cls=opt.single_cls,
        hyp=hyp,
        augment=True,
        cache=False if opt.cache == "val" else opt.cache,
        rect=opt.rect,
        rank=-1,
        workers=opt.workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr("train: "),
        shuffle=not opt.rect,
    )

    val_loader = create_dataloader(
        val_path,
        imgsz,
        batch_size if opt.val_batch <= 0 else opt.val_batch,
        32,
        single_cls=opt.single_cls,
        hyp=hyp,
        augment=False,
        cache=(opt.cache if not opt.noval else False),
        rect=True,
        rank=-1,
        workers=max(1, opt.workers // 2),
        pad=0.5,
        prefix=colorstr("val: "),
        shuffle=False,
    )[0]

    labels = np.concatenate(train_dataset.labels, 0) if len(train_dataset.labels) else np.zeros((0, 5))
    if len(train_dataset.labels):
        model.class_weights = labels_to_class_weights(train_dataset.labels, nc).to(device) * nc
    else:
        # 数据集无标签时回退为均匀权重，避免 labels_to_class_weights([]) 触发索引异常
        model.class_weights = torch.ones(nc, device=device)

    if len(labels) and not opt.noplots:
        plot_labels(labels, names, save_dir)

    if not opt.noautoanchor:
        check_anchors(train_dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)

    optimizer = build_optimizer(model, opt)
    scheduler, lf = build_scheduler(optimizer, opt, opt.epochs)
    ema = ModelEMA(model) if opt.ema else None
    scaler = amp.GradScaler(enabled=opt.amp and cuda)
    compute_loss = ComputeLoss(model)
    stopper = EarlyStopping(patience=opt.patience)

    nb = len(train_loader)
    nw = max(round(hyp["warmup_epochs"] * nb), 100)
    best_fitness = -1.0
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)

    print(f"训练输出目录: {save_dir}")
    print(f"device={device}, name={device_name}, batch={batch_size}, workers={opt.workers}, amp={opt.amp}, cache={opt.cache}")
    print("开始训练...")

    t0 = time.time()
    for epoch in range(opt.epochs):
        model.train()

        mloss = torch.zeros(3, device=device)
        optimizer.zero_grad(set_to_none=True)

        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=nb, ncols=140, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")

        if hasattr(train_loader.dataset, "mosaic"):
            train_loader.dataset.mosaic = not (opt.close_mosaic and epoch >= opt.epochs - opt.close_mosaic)

        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)

            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5 + 32)) // 32 * 32
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / 32) * 32 for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, max(1, round(64 / batch_size))]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])
            else:
                accumulate = max(1, round(64 / batch_size))

            with amp.autocast(enabled=opt.amp and cuda):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets)

            scaler.scale(loss).backward()

            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)

            mloss = (mloss * i + loss_items) / (i + 1)

            mem = f"{torch.cuda.memory_reserved() / 1e9:.2f}G" if cuda else "CPU"
            pbar.set_description(
                f"Epoch {epoch + 1}/{opt.epochs} "
                f"mem={mem} "
                f"box={mloss[0]:.4f} obj={mloss[1]:.4f} cls={mloss[2]:.4f} "
                f"img={imgs.shape[-1]}"
            )

            if epoch == 0 and i < 3 and not opt.noplots:
                plot_images(imgs, targets, paths, save_dir / f"train_batch{i}.jpg", names)

        scheduler.step()

        # EMA attributes sync
        if ema:
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])

        # validate every epoch
        if not opt.noval:
            val_model = ema.ema if ema else model
            results, maps, _ = validate.run(
                data=data_dict,
                batch_size=opt.val_batch if opt.val_batch > 0 else max(batch_size, 1),
                imgsz=imgsz,
                model=val_model,
                iou_thres=0.65,
                single_cls=opt.single_cls,
                dataloader=val_loader,
                save_dir=save_dir,
                plots=not opt.noplots,
                compute_loss=compute_loss,
                half=opt.amp and cuda,
                verbose=opt.verbose,
            )

        lr = [x["lr"] for x in optimizer.param_groups]
        write_results_csv(results_csv, epoch, mloss.tolist(), results, lr)

        fi = fitness(np.array(results).reshape(1, -1))
        if fi > best_fitness:
            best_fitness = fi

        if not opt.nosave:
            save_checkpoint(last_path, epoch, best_fitness, model, ema, optimizer, opt)
            if fi >= best_fitness:
                save_checkpoint(best_path, epoch, best_fitness, model, ema, optimizer, opt)

        if epoch == 0 and not opt.noplots and not opt.noval:
            try:
                val_batch_loader = iter(val_loader)
                imgs, targets, paths, _ = next(val_batch_loader)
                plot_images(imgs, targets, paths, save_dir / "val_batch0.jpg", names)
            except Exception as e:
                print(f"生成 val_batch0.jpg 失败: {e}")

        print(
            f"epoch {epoch + 1}/{opt.epochs} | "
            f"P={results[0]:.4f} R={results[1]:.4f} mAP50={results[2]:.4f} mAP50-95={results[3]:.4f}"
        )

        stop = stopper(epoch=epoch, fitness=fi)
        if stop:
            print(f"EarlyStopping: 提前停止于 epoch {epoch + 1}")
            break

    # final plots
    if not opt.noplots:
        try:
            plot_results(file=results_csv)
        except Exception as e:
            print(f"生成 results.png 失败: {e}")

    for f in last_path, best_path:
        if f.exists():
            try:
                strip_optimizer(f)
            except Exception as e:
                print(f"strip_optimizer 失败: {f.name}, {e}")

    print("\n训练完成")
    print(f"耗时: {(time.time() - t0) / 3600:.2f} 小时")
    print(f"输出目录: {save_dir}")
    print(f"last.pt: {last_path}")
    print(f"best.pt: {best_path}")
    print("\n应生成的文件通常包括：")
    print("- weights/last.pt")
    print("- weights/best.pt")
    print("- results.csv")
    print("- results.png")
    print("- PR_curve.png")
    print("- P_curve.png")
    print("- R_curve.png")
    print("- F1_curve.png")
    print("- confusion_matrix.png")
    print("- labels.jpg")
    print("- labels_correlogram.jpg")
    print("- train_batch0.jpg / train_batch1.jpg / train_batch2.jpg")
    print("- val_batch0.jpg")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov5s.pt", help="预训练权重（建议手动指定 yolov5x.pt 获取最强效果）")
    parser.add_argument("--cfg", type=str, default="", help="模型配置 yaml")
    parser.add_argument("--hyp", type=str, default="", help="超参文件 yaml")
    parser.add_argument("--data", type=str, default="data.yaml", help="数据集 yaml")
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--batch", type=int, default=-1, help="训练 batch，<=0 自动设置（RTX 5090 默认 64）")
    parser.add_argument("--batch-size", type=int, default=None, help="官方兼容参数，等价于 --batch")
    parser.add_argument("--val-batch", type=int, default=-1, help="验证 batch，<=0 自动设置（RTX 5090 默认 128）")
    parser.add_argument("--img", "--imgsz", dest="imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--device", default="0", help="设备，如 0 或 cpu")
    parser.add_argument("--workers", type=int, default=-1, help="数据加载线程，<=0 自动设置（RTX 5090 默认更高）")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam", "AdamW"])
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终 lr 系数")
    parser.add_argument("--momentum", type=float, default=0.937, help="momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="权重衰减")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="warmup epochs")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--project", default=ROOT / "runs" / "train", help="输出根目录")
    parser.add_argument("--name", type=str, default="fire_train_full", help="实验名")
    parser.add_argument("--exist-ok", action="store_true", help="已存在目录则继续使用")
    parser.add_argument("--cache", type=str, default="ram", choices=["ram", "disk", "val", "False", "false"], help="缓存图片")
    parser.add_argument("--amp", action="store_true", default=True, help="开启混合精度")
    parser.add_argument("--ema", action="store_true", default=True, help="开启 EMA")
    parser.add_argument("--cos-lr", action="store_true", default=True, help="余弦学习率")
    parser.add_argument("--single-cls", action="store_true", help="单类别训练")
    parser.add_argument("--rect", action="store_true", help="矩形训练")
    parser.add_argument("--image-weights", action="store_true", help="按图像权重采样")
    parser.add_argument("--multi-scale", action="store_true", help="多尺度训练")
    parser.add_argument("--noval", action="store_true", help="只训练不验证")
    parser.add_argument("--nosave", action="store_true", help="不保存 checkpoint")
    parser.add_argument("--noplots", action="store_true", help="不生成可视化图")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="冻结层，如 --freeze 10")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--mixup", action="store_true", help="开启 mixup")
    parser.add_argument("--no-mosaic", action="store_true", help="关闭 mosaic")
    parser.add_argument("--close-mosaic", type=int, default=10, help="最后 N 个 epoch 关闭 mosaic")
    parser.add_argument("--noautoanchor", action="store_true", help="关闭自动锚框检查")
    parser.add_argument("--patience", type=int, default=50, help="早停 patience")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="验证时打印各类指标")
    opt = parser.parse_args()

    if opt.batch_size is not None:
        opt.batch = opt.batch_size
    if str(opt.cache).lower() == "false":
        opt.cache = False

    return opt


def main(opt):
    print_args(vars(opt))
    train(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)