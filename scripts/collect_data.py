#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 数据集采集脚本 — 用摄像头拍照，用于后续标注和训练 YOLO
=============================================================================

 [用法]
   python scripts/collect_data.py                 # 交互式采集
   python scripts/collect_data.py --interval 2    # 每 2 秒自动拍一张
   python scripts/collect_data.py --count 100     # 拍 100 张后停止

 [按键]
   SPACE  =  拍一张照片
   Q/ESC  =  退出
   D      =  删除上一张

 [标注建议]
   拍照后用 labelImg 或 label-studio 标注钢珠位置。
   pip install labelImg
   labelImg ../data/images/train ../data/classes.txt
=============================================================================
"""

import sys, os, time, argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".venv", "Lib", "site-packages"))
import cv2

# ── 配置 ──────────────────────────────────────────────────────────
CAMERA_ID = 0
SAVE_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "images", "train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0, help="Auto-capture interval (seconds)")
    parser.add_argument("--count", type=int, default=0, help="Stop after N captures")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        sys.exit(1)

    print(f"[INFO] Saving to: {SAVE_DIR}")
    print("[INFO] Controls: SPACE=Capture  Q=Quit  D=Delete last")
    print("[INFO] Tip: Vary lighting, angles, and ball positions for better training")

    count = 0
    last_img = None
    last_auto = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 显示信息
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Captured: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 十字辅助线
        cv2.line(frame, (w//2, 0), (w//2, h), (100, 100, 100), 1)
        cv2.line(frame, (0, h//2), (w, h//2), (100, 100, 100), 1)

        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        # 退出
        if key == ord('q') or key == 27:
            break

        # 拍照
        capture = False
        if key == 32:  # SPACE
            capture = True
        elif args.interval > 0 and time.time() - last_auto >= args.interval:
            capture = True
            last_auto = time.time()

        if capture:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = os.path.join(SAVE_DIR, f"steel_ball_{ts}.jpg")
            cv2.imwrite(fname, frame)
            count += 1
            last_img = fname
            print(f"[{count}] Saved: {os.path.basename(fname)}")

            if args.count > 0 and count >= args.count:
                print(f"[INFO] Reached target count ({args.count}), exiting.")
                break

        # 删除上一张
        if key == ord('d') and last_img:
            os.remove(last_img)
            count -= 1
            print(f"[INFO] Deleted: {os.path.basename(last_img)}")
            last_img = None

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. {count} images saved to {SAVE_DIR}")
    print(f"[INFO] Next step: label images with labelImg, then run train.py")


if __name__ == "__main__":
    main()

