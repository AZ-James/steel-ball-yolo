#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 YOLO 训练脚本 — 在小钢珠数据集上微调 yolo11n
=============================================================================

 [前提]
   1. 已用 collect_data.py 采集图片  →  data/images/train/
   2. 已用 labelImg 标注边界框      →  data/labels/train/
   3. 数据集配置 steel_ball.yaml 已就绪

 [用法]
   cd steel_ball
   python scripts/train.py                      # 默认参数训练
   python scripts/train.py --epochs 200         # 自定义轮数
   python scripts/train.py --model yolo11s.pt   # 用更大模型 (更准但更慢)

 [训练产出]
   runs/detect/steel_ball/weights/best.pt   →  最佳模型 (验证集 mAP 最高)
   runs/detect/steel_ball/weights/last.pt   →  最后一轮
=============================================================================
"""

import sys, os, argparse

# 确保能找到 .venv 中的库
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".venv", "Lib", "site-packages"))

from ultralytics import YOLO

# ── 项目根目录 (steel_ball/) ────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Train YOLO for steel ball detection")
    parser.add_argument("--model",  default="yolo11n.pt", help="Base model (yolo11n/yolo11s/yolov8n...)")
    parser.add_argument("--epochs", type=int, default=100,   help="Training epochs (建议 100~300)")
    parser.add_argument("--imgsz",  type=int, default=640,   help="Input image size")
    parser.add_argument("--batch",  type=int, default=16,    help="Batch size (CPU 建议 8~16)")
    parser.add_argument("--lr0",    type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--device", default="cpu",            help="Device: cpu / cuda / 0")
    args = parser.parse_args()

    # ── 配置文件 (steel_ball.yaml) ──────────────────────────────────
    yaml_path = os.path.join(PROJECT_ROOT, "steel_ball.yaml")
    if not os.path.exists(yaml_path):
        print(f"[ERROR] Dataset config not found: {yaml_path}")
        sys.exit(1)

    # ── 输出目录 ────────────────────────────────────────────────────
    run_dir = os.path.join(PROJECT_ROOT, "runs")

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Config:       {yaml_path}")
    print(f"[INFO] Output:       {run_dir}")
    print(f"[INFO] Model:        {args.model}")
    print(f"[INFO] Epochs:       {args.epochs}")
    print(f"[INFO] Image size:   {args.imgsz}")
    print(f"[INFO] Batch size:   {args.batch}")
    print(f"[INFO] Device:       {args.device}")

    # ── 加载预训练模型 ──────────────────────────────────────────────
    model = YOLO(args.model)

    # ── 训练 ────────────────────────────────────────────────────────
    results = model.train(
        data=yaml_path,              # 数据集配置 (绝对路径)
        epochs=args.epochs,          # 训练轮数
        imgsz=args.imgsz,            # 输入尺寸
        batch=args.batch,            # 批次大小
        lr0=args.lr0,                # 初始学习率

        # 数据增强 (针对小目标钢珠优化)
        hsv_h=0.015,                 # 色调抖动 (钢珠颜色固定，调小)
        hsv_s=0.7,                   # 饱和度抖动
        hsv_v=0.4,                   # 亮度抖动 (补偿反光变化)
        degrees=10.0,                # 随机旋转 ±10°
        translate=0.1,               # 随机平移 10%
        scale=0.5,                   # 随机缩放 50% (模拟远近)
        shear=0.0,                   # 剪切 (圆形不变，关掉)
        flipud=0.0,                  # 上下翻转 (关)
        fliplr=0.5,                  # 左右翻转 50%
        mosaic=1.0,                  # Mosaic 4图拼接
        mixup=0.0,                   # MixUp (小数据集关)

        # 训练配置
        device=args.device,
        workers=4,
        project=run_dir,             # 输出根目录
        name="steel_ball",           # 子目录名 → runs/detect/steel_ball/
        exist_ok=True,               # 覆盖已有同名目录
        pretrained=True,             # 使用预训练权重
        optimizer="auto",            # 自动选择优化器
        verbose=True,
        seed=42,                     # 随机种子，保证可复现
    )

    best = os.path.join(run_dir, "detect", "steel_ball", "weights", "best.pt")
    print(f"\n{'='*60}")
    print(f"[INFO] Training complete!")
    print(f"[INFO] Best model: {best}")
    print(f"[INFO] Test:  python scripts/detect_yolo.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
