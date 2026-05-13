"""2:3 智能裁切核心模块。

设计目标（严格遵循需求）:
  1. 输出严格 2:3 (宽:高)。
  2. 人物水平居中。
  3. 模特头顶上方保留适度空白，不得切到头。
  4. 原图尺寸不足以裁出 2:3 时，自动用白色背景填充（两侧/顶部），绝不拉伸。
  5. 未检测到人物时，回退到全图居中 + 必要填白。

架构:
  * `plan_crop()` —— 纯 Python 几何规划，返回一个 CropPlan 指令；零依赖、易测。
  * `apply_plan()` —— 用 numpy 把 Plan 执行为一张严格 2:3 的输出图像。
  * `crop_to_2x3()` —— 对外一体化入口。

所有尺寸计算均不做 resize / 插值，保证像素级无损。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .detector import PersonBox

# 目标宽高比
TARGET_W = 2
TARGET_H = 3


@dataclass
class CropConfig:
    # 头顶到裁切窗口顶部的最小留白，按输出高度的比例
    head_margin_ratio: float = 0.08
    # 白色画布颜色（BGR）
    pad_color: Tuple[int, int, int] = (255, 255, 255)


@dataclass
class CropPlan:
    """几何规划结果。

    策略:
      1) 若 canvas_w/canvas_h 等于 src 中被裁出的窗口，则等价于 "纯裁切"
         （src_box 覆盖整个画布，paste_x = paste_y = 0）。
      2) 否则为 "白底画布"：先创建 canvas_w × canvas_h 白色画布，
         再把 src 中的 src_box 区域粘贴到画布的 (paste_x, paste_y) 处。
    """
    canvas_w: int
    canvas_h: int
    # src 中截取的区域（左闭右开）
    src_x1: int
    src_y1: int
    src_x2: int
    src_y2: int
    # 截取区域在 canvas 上的落点
    paste_x: int
    paste_y: int
    # 供调试 / 日志使用
    strategy: str  # 'crop_wide' | 'crop_tall' | 'pad_canvas' | 'fallback_center'

    @property
    def src_w(self) -> int:
        return self.src_x2 - self.src_x1

    @property
    def src_h(self) -> int:
        return self.src_y2 - self.src_y1

    def is_pure_crop(self) -> bool:
        return (
            self.canvas_w == self.src_w
            and self.canvas_h == self.src_h
            and self.paste_x == 0
            and self.paste_y == 0
        )

    def validate(self) -> None:
        """自检：严格 2:3 + 尺寸合法。"""
        assert self.canvas_w > 0 and self.canvas_h > 0
        assert self.canvas_w * TARGET_H == self.canvas_h * TARGET_W, (
            f"canvas {self.canvas_w}x{self.canvas_h} 非 2:3"
        )
        assert 0 <= self.src_x1 < self.src_x2
        assert 0 <= self.src_y1 < self.src_y2
        assert 0 <= self.paste_x
        assert 0 <= self.paste_y
        assert self.paste_x + self.src_w <= self.canvas_w
        assert self.paste_y + self.src_h <= self.canvas_h


# --------------------------------------------------------------------------- #
# 核心几何规划（纯 Python，无 numpy 依赖）
# --------------------------------------------------------------------------- #
def plan_crop(
    img_w: int,
    img_h: int,
    person: Optional[PersonBox],
    config: Optional[CropConfig] = None,
) -> CropPlan:
    """根据图像尺寸和人物位置，计算 2:3 输出的几何方案。"""
    if config is None:
        config = CropConfig()
    if img_w <= 0 or img_h <= 0:
        raise ValueError("图像尺寸非法")

    if person is None:
        return _plan_fallback(img_w, img_h, config)

    img_ratio_num = img_w * TARGET_H  # img_w / img_h vs TARGET_W / TARGET_H
    img_ratio_den = img_h * TARGET_W
    if img_ratio_num >= img_ratio_den:
        # 原图宽于或等于 2:3 → 尝试保留全高裁左右
        plan = _plan_crop_wide(img_w, img_h, person, config)
        if plan is not None:
            return plan
    else:
        # 原图偏瘦 → 尝试保留全宽裁上下
        plan = _plan_crop_tall(img_w, img_h, person, config)
        if plan is not None:
            return plan

    # 纯裁切不可行 → 白底补白
    return _plan_pad_canvas(img_w, img_h, person, config)


def _largest_2x3(w_limit: int, h_limit: int) -> Tuple[int, int]:
    """返回满足 canvas_w=2k, canvas_h=3k, 2k<=w_limit, 3k<=h_limit 的最大 (w,h)。
    保证 w*3 == h*2 严格成立。"""
    k = min(w_limit // TARGET_W, h_limit // TARGET_H)
    return k * TARGET_W, k * TARGET_H


def _plan_crop_wide(
    img_w: int, img_h: int, person: PersonBox, config: CropConfig,
) -> Optional[CropPlan]:
    """原图 ≥ 2:3：保留全高，裁左右。严格 2:3 的最大矩形 = (2k, 3k)。"""
    out_w, out_h = _largest_2x3(img_w, img_h)
    if out_w <= 0 or out_h <= 0:
        return None

    # 留白判断
    if person.head_top_y < out_h * config.head_margin_ratio:
        return None

    # 水平以人物头心居中
    x1 = person.head_center_x - out_w // 2
    x1 = max(0, min(img_w - out_w, x1))
    # 垂直：因为 out_h <= img_h 可能小 1~2 像素，从 0 开始；
    # 如果 out_h < img_h，优先保留头顶部分
    y1 = 0
    if out_h < img_h:
        # 把少的那几行留给底部（鞋子/裙摆）通常更合理
        pass  # y1 保持 0

    return CropPlan(
        canvas_w=out_w, canvas_h=out_h,
        src_x1=x1, src_y1=y1, src_x2=x1 + out_w, src_y2=y1 + out_h,
        paste_x=0, paste_y=0,
        strategy="crop_wide",
    )


def _plan_crop_tall(
    img_w: int, img_h: int, person: PersonBox, config: CropConfig,
) -> Optional[CropPlan]:
    """原图 < 2:3：保留全宽，裁上下。严格 2:3 的最大矩形 = (2k, 3k)。"""
    out_w, out_h = _largest_2x3(img_w, img_h)
    if out_w <= 0 or out_h <= 0:
        return None

    required_space = out_h * config.head_margin_ratio
    ideal_top = person.head_top_y - required_space

    if ideal_top < 0:
        return None  # 头顶贴边，裁会切到头
    if ideal_top + out_h > img_h:
        return None  # 人物偏下，底部会被切

    top = int(round(ideal_top))
    top = max(0, min(img_h - out_h, top))

    # 水平居中（此时画面 = 全宽，无平移空间，x1=0）
    x1 = 0
    if out_w < img_w:
        # 很罕见的情况：把少的像素留给两侧
        x1 = (img_w - out_w) // 2

    return CropPlan(
        canvas_w=out_w, canvas_h=out_h,
        src_x1=x1, src_y1=top, src_x2=x1 + out_w, src_y2=top + out_h,
        paste_x=0, paste_y=0,
        strategy="crop_tall",
    )


def _plan_pad_canvas(
    img_w: int, img_h: int, person: PersonBox, config: CropConfig,
) -> CropPlan:
    """白底画布方案：严格 2:3，人物水平居中，头顶留白充足。"""
    # 顶部需要补的像素：不动点迭代（因为 new_h 增大时 required 也增大）
    top_pad = 0
    for _ in range(6):
        new_h = img_h + top_pad
        required = new_h * config.head_margin_ratio
        actual = person.head_top_y + top_pad
        if actual >= required:
            break
        delta = int(math.ceil(required - actual))
        top_pad += max(1, delta)

    content_h = img_h + top_pad

    # 给定 content_h 和 img_w，计算能容纳它们的最小 2:3 画布 (canvas_w=2k, canvas_h=3k)
    k_by_w = math.ceil(img_w / TARGET_W)
    k_by_h = math.ceil(content_h / TARGET_H)
    k = max(k_by_w, k_by_h)
    canvas_w = k * TARGET_W
    canvas_h = k * TARGET_H

    # canvas_h 可能 > content_h，多出来的像素放底部（裙摆下方白边）
    # 也可能 canvas_h == content_h
    assert canvas_h >= content_h, f"{canvas_h} < {content_h}"
    assert canvas_w >= img_w

    # 水平粘贴位置：人物头心对齐到 canvas_w / 2
    paste_x = canvas_w // 2 - person.head_center_x
    paste_x = max(0, min(canvas_w - img_w, paste_x))

    # 垂直粘贴位置：原图顶部在 top_pad 处
    paste_y = top_pad

    plan = CropPlan(
        canvas_w=canvas_w, canvas_h=canvas_h,
        src_x1=0, src_y1=0, src_x2=img_w, src_y2=img_h,
        paste_x=paste_x, paste_y=paste_y,
        strategy="pad_canvas",
    )
    return plan


def _plan_fallback(img_w: int, img_h: int, config: CropConfig) -> CropPlan:
    """未检测到人物时：几何中心裁切 / 两侧居中补白。"""
    if img_w * TARGET_H == img_h * TARGET_W:
        return CropPlan(
            canvas_w=img_w, canvas_h=img_h,
            src_x1=0, src_y1=0, src_x2=img_w, src_y2=img_h,
            paste_x=0, paste_y=0,
            strategy="fallback_center",
        )

    # 尝试纯裁切：找到最大 (2k, 3k) 矩形在原图内的情况
    out_w, out_h = _largest_2x3(img_w, img_h)
    if out_w > 0 and out_h > 0 and out_w <= img_w and out_h <= img_h:
        x1 = (img_w - out_w) // 2
        y1 = (img_h - out_h) // 2
        return CropPlan(
            canvas_w=out_w, canvas_h=out_h,
            src_x1=x1, src_y1=y1, src_x2=x1 + out_w, src_y2=y1 + out_h,
            paste_x=0, paste_y=0,
            strategy="fallback_center",
        )

    # 理论不可达：补白兜底
    k_by_w = math.ceil(img_w / TARGET_W)
    k_by_h = math.ceil(img_h / TARGET_H)
    k = max(k_by_w, k_by_h)
    canvas_w = k * TARGET_W
    canvas_h = k * TARGET_H
    paste_x = (canvas_w - img_w) // 2
    paste_y = (canvas_h - img_h) // 2
    return CropPlan(
        canvas_w=canvas_w, canvas_h=canvas_h,
        src_x1=0, src_y1=0, src_x2=img_w, src_y2=img_h,
        paste_x=paste_x, paste_y=paste_y,
        strategy="fallback_center",
    )


# --------------------------------------------------------------------------- #
# 执行层（需要 numpy）
# --------------------------------------------------------------------------- #
def apply_plan(img, plan: CropPlan, pad_color: Tuple[int, int, int] = (255, 255, 255)):
    """按照 plan 把原图转成严格 2:3 输出。"""
    import numpy as np  # 延迟导入，保持 plan_crop 的纯 Python 可测性

    plan.validate()

    if plan.is_pure_crop():
        return img[plan.src_y1:plan.src_y2, plan.src_x1:plan.src_x2].copy()

    canvas = np.empty((plan.canvas_h, plan.canvas_w, 3), dtype=np.uint8)
    canvas[:] = pad_color

    crop = img[plan.src_y1:plan.src_y2, plan.src_x1:plan.src_x2]
    canvas[
        plan.paste_y:plan.paste_y + plan.src_h,
        plan.paste_x:plan.paste_x + plan.src_w,
    ] = crop
    return canvas


def crop_to_2x3(img, person: Optional[PersonBox], config: Optional[CropConfig] = None):
    """一体化入口。"""
    if img is None or img.size == 0:
        raise ValueError("输入图像为空")
    if config is None:
        config = CropConfig()
    h, w = img.shape[:2]
    plan = plan_crop(w, h, person, config)
    return apply_plan(img, plan, pad_color=config.pad_color)


def verify_ratio(img, tolerance: int = 1) -> bool:
    """检查图像是否为严格 2:3（允许 ±tolerance 像素误差）。"""
    h, w = img.shape[:2]
    # 严格整数判断：w*3 == h*2
    return abs(w * TARGET_H - h * TARGET_W) <= tolerance * max(TARGET_H, TARGET_W)
