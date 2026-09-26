#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 YOLO 检测脚本 — 用训练好的钢珠模型进行推理
=============================================================================

 [前提]
   已运行 train.py 训练完成，得到 best.pt

 [用法]
   python scripts/detect_yolo.py                              # 摄像头实时
   python scripts/detect_yolo.py --image test.jpg             # 单张图片
   python scripts/detect_yolo.py --model path/to/best.pt      # 指定模型
=============================================================================
"""

import sys, os, time, argparse, ssl
from datetime import datetime

# SSL 修复
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".venv", "Lib", "site-packages"))

import cv2
from ultralytics import YOLO

# ── 项目根目录 ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── 默认模型路径 (训练后的 best.pt) ─────────────────────────────────
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "runs", "steel_ball", "weights", "best.pt")


def detect_camera(model_path, conf=0.3):
    """摄像头实时 YOLO 检测"""
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path, verbose=False)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        sys.exit(1)

    print("[INFO] Camera opened. Q=Quit  S=Screenshot")

    prev_time = time.time()
    fps = 0.0
    fc = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转灰度并保持3通道 (与训练时一致)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # YOLO 推理
        results = model(frame, verbose=False, conf=conf)
        annotated = results[0].plot()  # ultralytics 内置绘制 (边框+标签+置信度)

        # FPS
        fc += 1
        now = time.time()
        if now - prev_time >= 1.0:
            fps = fc / (now - prev_time)
            fc = 0
            prev_time = now

        # 顶部信息面板
        h, w = annotated.shape[:2]
        count = len(results[0].boxes) if results[0].boxes is not None else 0
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), cv2.FILLED)
        annotated = cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0)
        cv2.putText(annotated, f"Steel Balls: {count}  |  FPS: {fps:.1f}",
                    (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Steel Ball - YOLO", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"yolo_steel_{ts}.jpg", annotated)
            print(f"[INFO] Screenshot saved: yolo_steel_{ts}.jpg")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


def detect_image(model_path, img_path, conf=0.3):
    """单张图片检测，打印每颗钢珠的坐标和置信度"""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("        Run train.py first or use --model to specify path.")
        sys.exit(1)

    model = YOLO(model_path, verbose=False)
    results = model(img_path, conf=conf)
    annotated = results[0].plot()

    count = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"Detected {count} steel balls in {img_path}")
    if results[0].boxes is not None:
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            c = float(box.conf[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            print(f"  #{i+1}: center=({cx},{cy})  box=({x1},{y1})-({x2},{y2})  conf={c:.3f}")

    cv2.imshow("Steel Ball - YOLO", annotated)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Steel Ball Detection")
    parser.add_argument("--image", "-i", type=str, help="Single image path")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help="Model .pt path")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (0~1)")
    args = parser.parse_args()

    if args.image:
        detect_image(args.model, args.image, args.conf)
    else:
        detect_camera(args.model, args.conf)




