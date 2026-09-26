#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 钢珠标注工具 (OpenCV 版)  — 替代 labelImg，稳定不闪退
=============================================================================

 [用法]
   python scripts/annotate.py

 [操作]
   鼠标拖拽画框  → 框住钢珠
   松开自动保存  → YOLO 格式 .txt
   D / Delete    → 删除当前图所有框
   空格 / →      → 下一张
   ←             → 上一张
   S             → 跳过 (无钢珠的图)
   ESC           → 退出 (自动保存进度)
   +/-           → 缩放图片

 [输出]
   data/labels/train/steel_ball_XXXX.txt  (YOLO 格式)
   data/progress.txt                       (标注进度，支持断点续标)
=============================================================================
"""

import os, sys, cv2, json, numpy as np
from pathlib import Path

# ── 路径配置 ───────────────────────────────────────────────────
IMG_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "images", "train")
LABEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "labels", "train")
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "progress.txt")
os.makedirs(LABEL_DIR, exist_ok=True)

# ── 颜色 ───────────────────────────────────────────────────────
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
WHITE  = (255, 255, 255)
BLUE   = (255, 0, 0)
BLACK  = (0, 0, 0)


class Annotator:
    def __init__(self):
        self.images = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))])
        if not self.images:
            print("[ERROR] No images found!")
            sys.exit(1)
        self.idx = self._load_progress()
        self.boxes = []           # 当前图的所有框 [(x1,y1,x2,y2), ...]
        self.drawing = False      # 是否正在画框
        self.start_pt = None      # 框起点
        self.img_raw = None       # 原图
        self.img_display = None   # 显示图
        self.scale = 1.0          # 缩放比例
        self.win_name = "Annotator - Steel Ball"
        self.modified = False     # 当前图是否被修改过
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win_name, self._mouse_callback)

    def _load_progress(self):
        """读取上次标注进度"""
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE) as f:
                idx = f.read().strip()
                if idx.isdigit():
                    i = int(idx)
                    if 0 <= i < len(self.images):
                        print(f"[INFO] Resuming from #{i}: {self.images[i]}")
                        return i
        return 0

    def _save_progress(self):
        with open(PROGRESS_FILE, "w") as f:
            f.write(str(self.idx))

    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调: 左键按下开始画框, 拖动显示预览, 松开保存框"""
        if self.img_display is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # 实时预览
            tmp = self.img_display.copy()
            cv2.rectangle(tmp, self.start_pt, (x, y), GREEN, 2)
            cv2.imshow(self.win_name, tmp)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_pt is None:
                return
            x1, y1 = self.start_pt
            x2, y2 = x, y
            # 规范化：确保左上→右下
            x1, x2 = min(x1,x2), max(x1,x2)
            y1, y2 = min(y1,y2), max(y1,y2)
            if x2 - x1 > 5 and y2 - y1 > 5:   # 太小忽略
                self.boxes.append((x1, y1, x2, y2))
                self.modified = True
            self.start_pt = None
            self._render()

    def _render(self):
        """绘制当前图 + 所有框 + 信息"""
        if self.img_raw is None:
            return
        self.img_display = self.img_raw.copy()
        # 画框
        for (x1, y1, x2, y2) in self.boxes:
            cv2.rectangle(self.img_display, (x1, y1), (x2, y2), GREEN, 2)
            cv2.putText(self.img_display, "steel_ball", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
        # 信息栏
        h, w = self.img_display.shape[:2]
        total = len(self.images)
        progress = f"[{self.idx+1}/{total}]"
        name = self.images[self.idx]
        cv2.putText(self.img_display, f"{progress}  {name}  Boxes:{len(self.boxes)}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
        cv2.putText(self.img_display, "Drag=Lbox  Space=Next  Del=Clear  S=Skip  ESC=Save&Quit",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)
        cv2.imshow(self.win_name, self.img_display)

    def _load_image(self):
        """加载当前索引的图片"""
        path = os.path.join(IMG_DIR, self.images[self.idx])
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Cannot read: {path}")
            return False
        # 缩放
        if self.scale != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w*self.scale), int(h*self.scale)))
        self.img_raw = img
        self.boxes = []
        self.modified = False
        # 读取已有标注
        label_path = self._get_label_path()
        if os.path.exists(label_path):
            img_h, img_w = img.shape[:2]
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, bw, bh = map(float, parts)
                        x1 = int((cx - bw/2) * img_w)
                        y1 = int((cy - bh/2) * img_h)
                        x2 = int((cx + bw/2) * img_w)
                        y2 = int((cy + bh/2) * img_h)
                        self.boxes.append((x1, y1, x2, y2))
        return True

    def _get_label_path(self):
        name = os.path.splitext(self.images[self.idx])[0] + ".txt"
        return os.path.join(LABEL_DIR, name)

    def _save_labels(self):
        """保存当前图片的标注为 YOLO 格式"""
        if not self.boxes:
            # 无框则删除已有标注文件
            lp = self._get_label_path()
            if os.path.exists(lp):
                os.remove(lp)
            return
        img_h, img_w = self.img_raw.shape[:2]
        lines = []
        for (x1, y1, x2, y2) in self.boxes:
            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(self._get_label_path(), "w") as f:
            f.write("\n".join(lines))

    def _next(self):
        if self.modified:
            self._save_labels()
        self._save_progress()
        if self.idx < len(self.images) - 1:
            self.idx += 1
            self._load_image()
            self._render()

    def _prev(self):
        if self.modified:
            self._save_labels()
        self._save_progress()
        if self.idx > 0:
            self.idx -= 1
            self._load_image()
            self._render()

    def run(self):
        if not self._load_image():
            print("[ERROR] Failed to load first image")
            return
        self._render()
        print(f"[INFO] Total: {len(self.images)} images. Start annotating...")
        print("        Drag mouse to draw box. Space=Next. D=Delete all. S=Skip. ESC=Exit")

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == 27:  # ESC
                if self.modified:
                    self._save_labels()
                self._save_progress()
                print(f"[INFO] Progress saved at #{self.idx+1}. Bye!")
                break
            elif key == 32 or key == ord('d'):  # Space or → = next
                self._next()
            elif key == ord('a') or key == 81:   # ← = prev
                self._prev()
            elif key == ord('s') or key == ord('S'):  # Skip
                self._next()
            elif key == 8 or key == 127:  # Delete / Backspace
                self.boxes = []
                self.modified = True
                self._render()
            elif key == ord('=') or key == ord('+'):
                self.scale = min(2.0, self.scale + 0.1)
                self._load_image()
                self._render()
            elif key == ord('-'):
                self.scale = max(0.3, self.scale - 0.1)
                self._load_image()
                self._render()
            elif key == ord('z'):  # Undo - remove last box
                if self.boxes:
                    self.boxes.pop()
                    self.modified = True
                    self._render()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    a = Annotator()
    a.run()

