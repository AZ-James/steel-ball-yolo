#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 小钢珠检测 — 传统 CV 方案 (霍夫圆检测 HoughCircles)
=============================================================================

 [原理]
   霍夫圆检测通过边缘梯度 + 投票机制在图像中寻找圆形。
   适合钢珠检测的原因：钢珠形状固定（圆形）、颜色单一（反光金属）。

 [检测流程]
   1. 高斯模糊去噪               →  减少噪点干扰
   2. 转灰度图                   →  简化处理
   3. 自适应阈值 / 固定阈值      →  分离钢珠和背景
   4. 霍夫圆检测                 →  找到圆形
   5. 过滤假阳性 (按尺寸/圆度)    →  只保留真正的钢珠

 [运行]
   python scripts/detect_cv.py                     # 摄像头实时检测
   python scripts/detect_cv.py --image test.jpg    # 单张图片检测
   python scripts/detect_cv.py --tune              # 调参模式 (显示滑块)

 [按键]
   Q / ESC  =  退出
   S        =  截图
   +/-      =  调整二值化阈值
   [/]      =  调整最小半径
=============================================================================
"""

import sys, os, time, argparse
from datetime import datetime

# ── 确保 .venv ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".venv", "Lib", "site-packages"))

import cv2
import numpy as np


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    可调参数（根据实际场景微调）                     ║
# ╚══════════════════════════════════════════════════════════════════╝

class Config:
    """集中管理所有可调参数，方便调参"""

    # ── 摄像头 ───────────────────────────────────────────────────
    CAMERA_ID      = 0            # 摄像头编号

    # ── 预处理 ───────────────────────────────────────────────────
    BLUR_SIZE      = (9, 9)       # 高斯模糊核大小 (奇数), 越大越模糊
    BLUR_SIGMA     = 2            # 高斯模糊标准差

    # ── 阈值（二值化） ────────────────────────────────────────────
    # strategy = "adaptive": 自适应阈值, 适合光照不均匀
    # strategy = "fixed":    固定阈值, 适合光照稳定
    THRESH_STRATEGY = "fixed"

    # 固定阈值参数
    THRESH_VALUE    = 120         # 阈值 (0~255), 大于此值的像素变白

    # 自适应阈值参数
    ADAPTIVE_BLOCK  = 25          # 邻域大小 (奇数)
    ADAPTIVE_C      = 10          # 常数偏移

    # ── 形态学操作 ────────────────────────────────────────────────
    MORPH_OPEN_SIZE  = (3, 3)     # 开运算核大小 (去噪点)
    MORPH_CLOSE_SIZE = (5, 5)     # 闭运算核大小 (填空洞)

    # ── 霍夫圆检测参数 ────────────────────────────────────────────
    # HoughCircles 的核心 4 个参数:
    HOUGH_DP        = 1.0         # 累加器分辨率 (1=原图, 2=一半)
    HOUGH_MIN_DIST  = 20          # 两个圆心之间的最小距离 (px)
    HOUGH_PARAM1    = 100         # Canny 边缘检测高阈值 (低阈值=高/2)
    HOUGH_PARAM2    = 25          # 圆心检测累加器阈值, 越小越多圆
    HOUGH_MIN_R     = 2           # 最小钢珠半径 (px)
    HOUGH_MAX_R     = 30          # 最大钢珠半径 (px)

    # ── 结果显示 ─────────────────────────────────────────────────
    DRAW_CIRCLES    = True         # 是否画圆
    DRAW_CROSSHAIR  = True         # 是否画十字准心
    DRAW_NUMBER     = True         # 是否编号

    # ── 颜色 ─────────────────────────────────────────────────────
    CIRCLE_COLOR    = (0, 255, 0)  # 圆边框 (BGR)
    CENTER_COLOR    = (0, 0, 255)  # 圆心
    TEXT_COLOR      = (255, 255, 255)
    PANEL_BG        = (0, 0, 0)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    检测核心函数                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def preprocess(img, cfg):
    """
    图像预处理流水线:
      BGR → 灰度 → 高斯模糊 → 阈值二值化 → 形态学开闭运算

    返回: 处理后的二值图像
    """
    # Step 1: 转灰度 - cv2.COLOR_BGR2GRAY = 6
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: 高斯模糊 - 消除高频噪声，让边缘更干净
    #   核大小必须是奇数 (3,5,7,9...)
    #   标准差越大模糊越厉害
    blurred = cv2.GaussianBlur(gray, cfg.BLUR_SIZE, cfg.BLUR_SIGMA)

    # Step 3: 二值化 - 把图像变成黑白，分离物体和背景
    if cfg.THRESH_STRATEGY == "adaptive":
        # 自适应阈值：每个像素的阈值由周围 ADAPTIVE_BLOCK 个像素的均值决定
        # 适合光照不均的场景 (如台灯从侧面照)
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # 高斯加权邻域
            cv2.THRESH_BINARY,                 # 高于阈值→白
            cfg.ADAPTIVE_BLOCK,
            cfg.ADAPTIVE_C
        )
    else:
        # 固定阈值：全局统一阈值
        # 反光钢珠通常比背景亮 → 用 THRESH_BINARY 把亮的变成白色
        _, binary = cv2.threshold(blurred, cfg.THRESH_VALUE, 255, cv2.THRESH_BINARY)

    # Step 4: 形态学操作 - 清理噪点和空洞
    #   开运算 = 先腐蚀再膨胀 → 去除小白点噪声
    #   闭运算 = 先膨胀再腐蚀 → 填充小空洞
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cfg.MORPH_OPEN_SIZE)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cfg.MORPH_CLOSE_SIZE)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel_open)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    return binary, gray, blurred


def detect_circles_hough(blurred, cfg):
    """
    霍夫圆检测。

    cv2.HoughCircles 内部流程:
      1. Canny 边缘检测 (param1=高阈值, 低阈值=param1/2)
      2. 对每个边缘点计算梯度方向
      3. 沿梯度方向在累加器中"投票"
      4. param2 控制需要多少票才算一个圆心
      5. 返回通过阈值的所有圆

    HOUGH_GRADIENT 是 OpenCV 唯一稳定可用的方法。
    """
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=cfg.HOUGH_DP,
        minDist=cfg.HOUGH_MIN_DIST,
        param1=cfg.HOUGH_PARAM1,
        param2=cfg.HOUGH_PARAM2,
        minRadius=cfg.HOUGH_MIN_R,
        maxRadius=cfg.HOUGH_MAX_R,
    )
    return circles  # shape=(1, N, 3) → [[x, y, r], ...]  或 None


def validate_circles(circles, binary, cfg):
    """
    后处理：过滤假阳性。

    过滤策略:
      1. 检查圆心处是否为白色 (二值图中)

    返回: 过滤后的 circles
    """
    if circles is None:
        return None

    valid = []
    h, w = binary.shape

    for (x, y, r) in circles[0]:
        x, y, r = int(x), int(y), int(r)

        # 边界检查 — 圆心必须在图像范围内
        if x < 0 or y < 0 or x >= w or y >= h:
            continue

        # 检查二值图中圆心处是否为白色
        # 真正的钢珠圆心应该在二值化后是白色区域
        if binary[y, x] < 128:
            continue

        valid.append([x, y, r])

    if not valid:
        return None

    return np.array([valid], dtype=np.uint16)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    绘制函数                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

def draw_results(img, circles, cfg):
    """在图像上绘制检测到的钢珠"""
    if circles is None:
        return img

    for i, (x, y, r) in enumerate(circles[0]):
        x, y, r = int(x), int(y), int(r)

        if cfg.DRAW_CIRCLES:
            # 画圆，线宽 2px
            cv2.circle(img, (x, y), r, cfg.CIRCLE_COLOR, 2)

        if cfg.DRAW_CROSSHAIR:
            # 圆心十字准心
            cv2.drawMarker(img, (x, y), cfg.CENTER_COLOR,
                           cv2.MARKER_CROSS, 10, 2)

        if cfg.DRAW_NUMBER:
            # 编号
            cv2.putText(img, f"#{i+1}", (x + r + 5, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cfg.CIRCLE_COLOR, 2)

    return img


def draw_panel(img, count, fps, binary):
    """绘制顶部信息面板 + 右侧预览"""
    h, w = img.shape[:2]

    # 半透明顶部面板
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (0, 0, 0), cv2.FILLED)
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    cv2.putText(img, "Steel Ball Detection | HoughCircles",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, f"Balls: {count} | FPS: {fps:.1f}",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "Q=Quit S=Shot +/-=Thresh []=Radius",
                (w - 330, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # 右侧小窗：二值化预览 (帮助调参)
    if binary is not None:
        preview = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        preview = cv2.resize(preview, (200, 150))
        ph, pw = preview.shape[:2]
        img[h-ph-10:h-10, w-pw-10:w-10] = preview
        cv2.putText(img, "binary preview", (w-pw-10, h-ph-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return img


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    单图检测 (调试用)                             ║
# ╚══════════════════════════════════════════════════════════════════╝

def detect_image(image_path, cfg):
    """对单张图片进行检测，显示结果并等待按键"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read: {image_path}")
        return

    binary, gray, blurred = preprocess(img, cfg)
    circles = detect_circles_hough(blurred, cfg)
    circles = validate_circles(circles, binary, cfg)

    count = len(circles[0]) if circles is not None else 0
    print(f"Detected {count} steel balls in {image_path}")
    if circles is not None:
        for i, (x, y, r) in enumerate(circles[0]):
            print(f"  #{i+1}: center=({x},{y})  radius={r}px")

    img = draw_results(img, circles, cfg)
    img = draw_panel(img, count, 0, binary)

    # 缩放显示（如果太大）
    h, w = img.shape[:2]
    max_w = 1200
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)))

    cv2.imshow("Steel Ball Detection", img)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    摄像头实时检测                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def detect_camera(cfg, tune_mode=False):
    """打开摄像头，实时检测钢珠"""
    cap = cv2.VideoCapture(cfg.CAMERA_ID)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera #{cfg.CAMERA_ID}")
        sys.exit(1)

    print("[INFO] Camera opened. Controls:")
    print("        Q/ESC  = Quit")
    print("        S      = Screenshot")
    print("        +/-    = Adjust threshold")
    print("        [/]    = Adjust min/max radius")

    prev_time = time.time()
    fps = 0.0
    fc = 0

    if tune_mode:
        cv2.namedWindow("Steel Ball Detection")
        cv2.createTrackbar("Thresh", "Steel Ball Detection",
                           cfg.THRESH_VALUE, 255, lambda x: None)
        cv2.createTrackbar("Min R", "Steel Ball Detection",
                           cfg.HOUGH_MIN_R, 100, lambda x: None)
        cv2.createTrackbar("Max R", "Steel Ball Detection",
                           cfg.HOUGH_MAX_R, 200, lambda x: None)
        cv2.createTrackbar("Param2", "Steel Ball Detection",
                           cfg.HOUGH_PARAM2, 200, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 如果是调参模式，从滑块读取参数
        if tune_mode:
            cfg.THRESH_VALUE = cv2.getTrackbarPos("Thresh", "Steel Ball Detection")
            cfg.HOUGH_MIN_R  = cv2.getTrackbarPos("Min R", "Steel Ball Detection")
            cfg.HOUGH_MAX_R  = cv2.getTrackbarPos("Max R", "Steel Ball Detection")
            cfg.HOUGH_PARAM2 = cv2.getTrackbarPos("Param2", "Steel Ball Detection")
            cfg.HOUGH_MIN_R  = max(1, cfg.HOUGH_MIN_R)
            cfg.HOUGH_MAX_R  = max(cfg.HOUGH_MIN_R + 1, cfg.HOUGH_MAX_R)

        # 预处理
        binary, gray, blurred = preprocess(frame, cfg)

        # 检测
        circles = detect_circles_hough(blurred, cfg)
        circles = validate_circles(circles, binary, cfg)

        # 计数
        count = len(circles[0]) if circles is not None else 0

        # 绘制
        frame = draw_results(frame, circles, cfg)
        frame = draw_panel(frame, count, fps, binary)

        cv2.imshow("Steel Ball Detection", frame)

        # FPS
        fc += 1
        now = time.time()
        if now - prev_time >= 1.0:
            fps = fc / (now - prev_time)
            fc = 0
            prev_time = now

        # 按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"steel_ball_{ts}.jpg", frame)
            print(f"[INFO] Screenshot: steel_ball_{ts}.jpg")
        elif key in [ord('+'), ord('=')]:
            cfg.THRESH_VALUE = min(255, cfg.THRESH_VALUE + 5)
            print(f"[INFO] Threshold: {cfg.THRESH_VALUE}")
        elif key == ord('-'):
            cfg.THRESH_VALUE = max(0, cfg.THRESH_VALUE - 5)
            print(f"[INFO] Threshold: {cfg.THRESH_VALUE}")
        elif key == ord(']'):
            cfg.HOUGH_MAX_R = min(200, cfg.HOUGH_MAX_R + 2)
            print(f"[INFO] Max Radius: {cfg.HOUGH_MAX_R}")
        elif key == ord('['):
            cfg.HOUGH_MIN_R = max(1, cfg.HOUGH_MIN_R - 1)
            print(f"[INFO] Min Radius: {cfg.HOUGH_MIN_R}")

    cap.release()
    cv2.destroyAllWindows()


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    入口                                          ║
# ╚══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steel Ball Detector")
    parser.add_argument("--image", "-i", type=str, help="Single image path")
    parser.add_argument("--tune", "-t", action="store_true", help="Tune mode with sliders")
    args = parser.parse_args()

    cfg = Config()

    if args.image:
        detect_image(args.image, cfg)
    else:
        detect_camera(cfg, tune_mode=args.tune)
