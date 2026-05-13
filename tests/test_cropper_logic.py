"""纯 Python 几何测试：验证 cropper.plan_crop 的 2:3 + 居中 + 留白 + 补白正确性。

无需 numpy / cv2 / mediapipe。
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 把 app.detector 替换成一个仅提供 PersonBox 的最小存根，绕过 mediapipe 依赖
stub = types.ModuleType("app.detector")

from dataclasses import dataclass  # noqa: E402


@dataclass
class PersonBox:
    x1: int
    y1: int
    x2: int
    y2: int
    head_top_y: int
    head_center_x: int
    source: str = "pose"


stub.PersonBox = PersonBox
sys.modules["app.detector"] = stub

from app.cropper import CropConfig, plan_crop  # noqa: E402


def _person(head_center_x, head_top, img_w=1000, img_h=1500):
    return PersonBox(
        x1=max(0, head_center_x - 100), y1=head_top,
        x2=min(img_w, head_center_x + 100), y2=int(img_h * 0.9),
        head_top_y=head_top, head_center_x=head_center_x,
    )


def _check_2x3(plan):
    assert plan.canvas_w * 3 == plan.canvas_h * 2, (
        f"{plan.canvas_w}x{plan.canvas_h} 非严格 2:3"
    )
    plan.validate()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_wide_image_crops_sides():
    # 2000x1000 (2:1) → 裁为 ~666x1000（需 w*3==h*2）→ 实际应为 666x999
    plan = plan_crop(2000, 1000, _person(1000, 200, 2000, 1000))
    _check_2x3(plan)
    assert plan.strategy == "crop_wide"
    # 画布高 <= 原图高
    assert plan.canvas_h <= 1000


def test_wide_image_centers_on_person_left():
    plan = plan_crop(1800, 900, _person(400, 100, 1800, 900))
    _check_2x3(plan)
    assert plan.strategy == "crop_wide"
    # 人物偏左 → src_x1 应较小
    assert plan.src_x1 < 400


def test_wide_image_centers_on_person_right():
    plan = plan_crop(1800, 900, _person(1500, 100, 1800, 900))
    _check_2x3(plan)
    assert plan.strategy == "crop_wide"
    # 人物偏右 → src_x1 较大
    assert plan.src_x1 > 600


def test_tall_image_crops_top():
    plan = plan_crop(1000, 2000, _person(500, 400, 1000, 2000))
    _check_2x3(plan)
    assert plan.strategy == "crop_tall"
    assert plan.canvas_w == 1000
    assert plan.canvas_h == 1500
    # 头顶留白 >= 1500 * 0.08 = 120
    head_top_in_canvas = 400 - plan.src_y1
    assert head_top_in_canvas >= 120


def test_tall_image_head_too_high_falls_back_to_padding():
    plan = plan_crop(1000, 2000, _person(500, 50, 1000, 2000))
    _check_2x3(plan)
    assert plan.strategy == "pad_canvas"
    # 顶部有白边
    assert plan.paste_y > 0


def test_square_image_crops_to_largest_2x3():
    # 1000x1000 方形图：可裁成最大 2:3 矩形 = (666, 999)
    # 人物在中间且头顶留白充足 (150 > 999*0.08=79.92)
    plan = plan_crop(1000, 1000, _person(500, 150, 1000, 1000))
    _check_2x3(plan)
    # 能纯裁切
    assert plan.strategy in ("crop_wide", "crop_tall")
    assert plan.canvas_w == 666
    assert plan.canvas_h == 999


def test_no_person_square_pads():
    plan = plan_crop(500, 500, None)
    _check_2x3(plan)
    assert plan.strategy == "fallback_center"


def test_no_person_exact_ratio_noop():
    plan = plan_crop(600, 900, None)
    _check_2x3(plan)
    assert plan.canvas_w == 600 and plan.canvas_h == 900


def test_strict_ratio_across_many_sizes():
    cases = [
        (600, 400), (1200, 800), (800, 1200), (1080, 1920),
        (1500, 1500), (3000, 1000), (300, 1000),
        (1, 1), (2, 3), (3, 2), (4000, 6000),
    ]
    for w, h in cases:
        p = _person(w // 2, max(10, h // 5), w, h)
        plan = plan_crop(w, h, p)
        assert plan.canvas_w * 3 == plan.canvas_h * 2, (
            f"输入 {w}x{h} → 输出 {plan.canvas_w}x{plan.canvas_h} 非 2:3"
        )
        plan.validate()


def test_paste_x_horizontal_centering():
    # 正方形输入，人物头心在 x=300
    plan = plan_crop(1000, 1000, _person(300, 150, 1000, 1000))
    _check_2x3(plan)
    # 画布中心应对齐人物头心：paste_x + 人物头心_in_src = canvas_w // 2
    head_in_canvas = plan.paste_x + 300
    assert abs(head_in_canvas - plan.canvas_w // 2) <= 1 or plan.paste_x == 0 or plan.paste_x == plan.canvas_w - 1000


def test_head_margin_respected_in_padding():
    # 头顶紧贴图像顶部，必须补白
    plan = plan_crop(600, 1000, _person(300, 5, 600, 1000))
    _check_2x3(plan)
    cfg = CropConfig()
    # 人物在 canvas 中的头顶位置
    head_y_in_canvas = plan.paste_y + 5
    required = plan.canvas_h * cfg.head_margin_ratio
    assert head_y_in_canvas >= required - 1, (
        f"head_y={head_y_in_canvas} < required={required}"
    )


def test_plan_does_not_stretch():
    # src 区域大小必定 <= 原图，且粘贴区域 <= 画布
    plan = plan_crop(800, 1200, _person(400, 200, 800, 1200))
    _check_2x3(plan)
    assert plan.src_w <= 800 and plan.src_h <= 1200
    assert plan.paste_x + plan.src_w <= plan.canvas_w
    assert plan.paste_y + plan.src_h <= plan.canvas_h


def test_very_wide_pano_image():
    # 3000x1000：人物在中间
    plan = plan_crop(3000, 1000, _person(1500, 200, 3000, 1000))
    _check_2x3(plan)
    assert plan.strategy == "crop_wide"
    assert plan.src_w < 3000  # 确实裁了左右


def test_very_tall_portrait():
    # 1000x4000：人物头顶留白充足
    plan = plan_crop(1000, 4000, _person(500, 600, 1000, 4000))
    _check_2x3(plan)
    assert plan.strategy == "crop_tall"
    assert plan.canvas_h == 1500


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import traceback
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
