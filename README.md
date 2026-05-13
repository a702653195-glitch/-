# 女装电商图片智能裁切 (2:3)

一个面向女装电商图片的批量智能裁切 Windows 软件。使用 MediaPipe 识别人物，
按严格 2:3 比例输出，保证人物水平居中、头顶留白充足，原图不足以裁出 2:3 时自动用白色背景填充（绝不拉伸）。

## 功能特性

- **严格 2:3**：输出图像 `w * 3 == h * 2`，像素级精确。
- **视觉美感**：
  - MediaPipe Pose（主）+ FaceDetection（备）双重检测。
  - 人物水平居中（以头部中心为锚点）。
  - 头顶上方保留适度留白（默认 8%，可调）。
- **智能填白**：原图无法裁出 2:3 时，自动用白色背景在顶部/两侧填充，无任何拉伸。
- **多格式支持**：JPG, JPEG, PNG, WebP, BMP。输入什么格式，输出什么格式。
- **多线程批处理**：默认 CPU 核数的一半，可在 GUI 中调节 1~16。
- **Windows 中文路径兼容**：读写全部使用 `numpy.fromfile` + `cv2.imdecode` 双通道。
- **进度条 + 实时日志**：tkinter 图形界面，支持中途取消。

## 安装

要求 Python 3.10+（推荐 3.11）。

```bash
pip install -r requirements.txt
```

依赖说明：

| 包 | 用途 |
| -- | -- |
| `opencv-python` | 图像编解码、数组操作 |
| `mediapipe` | Pose / Face 检测 |
| `numpy` | 像素运算 |
| `Pillow` | 格式兼容辅助（PyInstaller 打包时减少 plugins 问题） |

## 运行

```bash
python main.py
```

主界面操作：

1. **输入文件夹**：选择待处理图片所在目录。
2. **输出文件夹**：选择输出目录（默认在输入目录旁生成 `_cropped`）。
3. **参数**：
   - **并发线程**：默认按 CPU 自动选取 2~8。
   - **模型精度**：Lite（最快）/ Full（默认）/ Heavy（高精度）。
   - **头顶留白比例**：占输出高度的百分比，默认 `0.08`（8%）。
   - **输出质量**：JPEG / WebP 质量 60~100，默认 95。
4. 点击 **开始处理**。可以随时取消。

处理完成后可以点击 **打开输出目录** 直接查看结果。

## 裁切策略

```
输入图像
  ├── 原图宽≥2:3 → 保留全高，裁左右（人物头心居中）
  │    └── 若头顶留白不足 → 白底画布补白
  ├── 原图<2:3   → 保留全宽，裁上下（留出头顶空间）
  │    └── 若无法兼顾头顶留白 → 白底画布补白
  └── 未检测到人物 → 几何中心裁切 / 两侧居中补白
```

关键保证：
- 输出分辨率恒为 `(2k, 3k)` 形式，不存在 ±1 像素的舍入误差。
- 任何情况下都**不**调用 `resize` / 插值，像素要么来自原图，要么为纯白填充。

## 项目结构

```
.
├── main.py                 # GUI 入口 (tkinter)
├── app/
│   ├── __init__.py
│   ├── utils.py            # Unicode 路径下的图像读写、遍历
│   ├── detector.py         # MediaPipe 人物检测封装 → PersonBox
│   ├── cropper.py          # 2:3 几何规划 + 执行
│   └── processor.py        # 多线程批处理调度
├── tests/
│   └── test_cropper_logic.py  # 纯 Python 几何单元测试
├── requirements.txt
├── build.bat               # PyInstaller 一键打包脚本
└── README.md
```

## 打包为 .exe

需要在 Windows 上执行：

```bat
pip install pyinstaller
build.bat
```

成品位于 `dist\CropTool.exe`。双击即可运行，首次启动会稍慢（MediaPipe 模型加载）。

## 命令行使用（可选）

虽然主交互是 GUI，也可以直接调用处理函数：

```python
from app.processor import process_batch

stats = process_batch(
    "D:/photos/raw",
    "D:/photos/out",
    max_workers=6,
    head_margin_ratio=0.10,
)
print(stats.ok, stats.failed)
```

## 测试

不依赖 MediaPipe 的纯几何测试：

```bash
python tests/test_cropper_logic.py
```

应输出 `14 passed, 0 failed`。

## 已知限制

- MediaPipe 的 Pose 模型首次加载需 1~3 秒，首张图较慢。
- 极端倾斜、背身或多人场景，人物定位可能以最显眼的一位为准。
- GPU 加速未启用（如需，可改 `model_complexity=2` 并启用 MediaPipe GPU 分发）。
