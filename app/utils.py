"""图像 I/O 工具模块。

负责 JPG / PNG / WebP 的统一读写，保留原格式与合理质量参数。
使用 numpy 读取以避免 OpenCV 在非 ASCII 路径下读写失败的问题（Windows 中文路径常见问题）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def imread_unicode(path: str | Path) -> np.ndarray | None:
    """读取图像，兼容 Windows 非 ASCII 路径。

    返回 BGR 顺序的 numpy 数组；失败时返回 None。
    """
    path = str(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_unicode(path: str | Path, img: np.ndarray, quality: int = 95) -> bool:
    """写入图像，按扩展名选择编码参数，兼容非 ASCII 路径。"""
    path = str(path)
    ext = os.path.splitext(path)[1].lower()

    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
    elif ext == ".png":
        # PNG 压缩级别 0-9，9 最高压缩
        level = max(0, min(9, int((100 - quality) / 11)))
        params = [cv2.IMWRITE_PNG_COMPRESSION, level]
    else:
        params = []

    try:
        ok, buf = cv2.imencode(ext if ext else ".jpg", img, params)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


def iter_images(folder: str | Path, recursive: bool = True) -> List[Path]:
    """遍历文件夹中的所有受支持图片。"""
    folder = Path(folder)
    if not folder.exists():
        return []
    pattern = "**/*" if recursive else "*"
    return [
        p for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def derive_output_path(
    input_path: Path, input_root: Path, output_root: Path,
    preserve_tree: bool = True,
) -> Path:
    """根据输入路径推导输出路径，可选保留子目录结构。"""
    if preserve_tree:
        try:
            rel = input_path.relative_to(input_root)
        except ValueError:
            rel = Path(input_path.name)
    else:
        rel = Path(input_path.name)
    return output_root / rel
