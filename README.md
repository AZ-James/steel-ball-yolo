# 小钢珠检测工程 (Steel Ball Detection)

## 项目结构

```
steel_ball/
├── steel_ball.yaml           # YOLO 数据集配置
├── README.md
├── data/
│   ├── classes.txt           # 类别名
│   ├── images/
│   │   ├── train/            # 104 张训练图 (灰度)
│   │   └── val/              # 26 张验证图 (灰度)
│   └── labels/
│       ├── train/            # 104 个标注文件
│       └── val/              # 26 个标注文件
├── scripts/
│   ├── annotate.py           # 标注工具 (OpenCV 版, 替代 labelImg)
│   ├── detect_cv.py          # [备用] 传统 CV 霍夫圆检测
│   ├── detect_yolo.py        # YOLO 实时检测 (训练后使用)
│   ├── collect_data.py       # 采集图片
│   ├── train.py              # YOLO 训练
│   └── split_data.py         # 划分数据集
├── runs/
│   └── steel_ball/weights/
│       ├── best.pt           # ⭐ 最佳模型 (mAP50=0.995)
│       └── last.pt           # 最后一轮模型
└── steel_ball.yaml
```

## 训练结果

| 指标 | 数值 | 说明 |
|------|------|------|
| 训练集 | 104 张 / 704 个框 | 80% 划分 |
| 验证集 | 26 张 / 103 个框 | 20% 划分 |
| 基础模型 | yolo11n.pt | Ultralytics 预训练 |
| 训练轮数 | 150 epochs | CPU 约 50 分钟 |
| **mAP50** | **0.995** | 接近完美 |
| 精确率 (P) | 0.997 | 几乎零误检 |
| 召回率 (R) | 1.000 | 零漏检 |
| 推理速度 | ~31 ms/张 | CPU 约 32 FPS，实时流畅 |
| 模型大小 | 5.5 MB | 轻量 |

## 环境

| 项目 | 版本/路径 |
|------|-----------|
| Python | `D:\个人的数据\PyCharm\电赛\.venv\Scripts\python.exe` |
| ultralytics | 8.4.108 |
| torch | 2.13.0 (CPU) |
| opencv | 5.0.0 |

## 运行命令

```bash
# 切到项目目录
cd /d D:\个人的数据\PyCharm\电赛\steel_ball

# 实时检测 (默认)
D:\个人的数据\PyCharm\电赛\.venv\Scripts\python.exe scripts/detect_yolo.py

# 提高置信度阈值
D:\个人的数据\PyCharm\电赛\.venv\Scripts\python.exe scripts/detect_yolo.py --conf 0.5

# 单张图片检测
D:\个人的数据\PyCharm\电赛\.venv\Scripts\python.exe scripts/detect_yolo.py --image test.jpg
```

| 按键 | 功能 |
|------|------|
| Q / ESC | 退出 |
| S | 截图 |

## 关键说明

### 为什么摄像头输入要转灰度？
模型训练时用的是灰度图，推理时如果不转灰度，彩色输入和训练数据分布不一致，会导致置信度低、误检多。`detect_yolo.py` 已内置自动灰度转换。

### best.pt vs yolo11n.pt
- `yolo11n.pt` — 官方预训练模型，识别 80 类通用物体（无钢珠）
- `best.pt` — 在你的 130 张钢珠数据上微调后，专用于钢珠检测

### 标注工具
labelImg 和 PyQt5 有兼容性问题，使用 `scripts/annotate.py` 替代：
```bash
D:\个人的数据\PyCharm\电赛\.venv\Scripts\python.exe scripts/annotate.py
```
操作：鼠标拖拽画框 | 空格下一张 | A上一张 | Delete清框 | Z撤销 | ESC保存退出

### OpenCV 中文路径问题
OpenCV 的 `imread` 不支持中文路径，统一使用 `np.fromfile + cv2.imdecode` 替代。

## 完整流程回顾

```
拍照(手机130张) → 灰度预处理 → annotate.py标注(704框)
→ 划分训练/验证(8:2) → train.py训练(150轮)
→ best.pt(99.5% mAP) → detect_yolo.py实时检测 ✅
```
