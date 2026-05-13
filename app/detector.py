"""人物检测模块。

策略（按优先级尝试）:
  1. MediaPipe Pose：最稳健，可直接拿到头顶附近关键点（鼻尖、耳、眼）与身体躯干。
  2. MediaPipe FaceDetection：当 Pose 未检测到人时的回退（例如半身特写、只有头）。
  3. 若全部失败，返回 None，让上层做全图居中裁切 + 白底填充。

对外只暴露一个简单的结果结构 PersonBox，供 cropper 使用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None


@dataclass
class PersonBox:
    """检测到的人物区域（像素坐标，均为 int）。"""
    x1: int
    y1: int
    x2: int
    y2: int
    head_top_y: int      # 估算的头顶 y 坐标（越小越靠上）
    head_center_x: int   # 头部中心 x 坐标
    source: str          # 'pose' | 'face' | 'fallback'

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class PersonDetector:
    """对单个 worker 线程安全的检测器封装。

    MediaPipe 的 solution 对象不是完全线程安全的，因此每个实例只应在单线程中使用；
    多线程场景请为每个线程各自实例化一次。
    """

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        if mp is None:
            raise ImportError("mediapipe 未安装，请先 pip install mediapipe")

        self._mp_pose = mp.solutions.pose
        self._mp_face = mp.solutions.face_detection

        self._pose = self._mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
        )
        # model_selection=1 适合全身距离较远的场景
        self._face = self._mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence,
        )

    def close(self) -> None:
        try:
            self._pose.close()
        except Exception:
            pass
        try:
            self._face.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ------------------------------------------------------------------ #
    # 核心检测入口
    # ------------------------------------------------------------------ #
    def detect(self, bgr_image: np.ndarray) -> Optional[PersonBox]:
        """检测图像中最显著的人物，返回 PersonBox。找不到时返回 None。"""
        if bgr_image is None or bgr_image.size == 0:
            return None

        # MediaPipe 要求 RGB
        rgb = bgr_image[:, :, ::-1]

        box = self._detect_pose(rgb)
        if box is not None:
            return box
        return self._detect_face(rgb)

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    def _detect_pose(self, rgb: np.ndarray) -> Optional[PersonBox]:
        h, w = rgb.shape[:2]
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None

        lms = result.pose_landmarks.landmark

        # 仅保留 visibility 较高的点，避免离散离群点
        visible_xy = [
            (lm.x * w, lm.y * h) for lm in lms if lm.visibility >= 0.3
        ]
        if not visible_xy:
            return None

        xs = np.array([p[0] for p in visible_xy])
        ys = np.array([p[1] for p in visible_xy])

        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())

        # 估算头顶：使用鼻子、双眼、双耳中最靠上的点，再向上外推一定比例
        # 因为 Pose 模型无法直接给出头顶，需要补偿头发/头皮高度
        head_candidate_ids = [
            self._mp_pose.PoseLandmark.NOSE.value,
            self._mp_pose.PoseLandmark.LEFT_EYE.value,
            self._mp_pose.PoseLandmark.RIGHT_EYE.value,
            self._mp_pose.PoseLandmark.LEFT_EAR.value,
            self._mp_pose.PoseLandmark.RIGHT_EAR.value,
        ]
        head_pts = [
            (lms[i].x * w, lms[i].y * h)
            for i in head_candidate_ids
            if lms[i].visibility >= 0.3
        ]

        shoulder_ids = [
            self._mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self._mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        ]
        shoulder_ys = [
            lms[i].y * h for i in shoulder_ids if lms[i].visibility >= 0.3
        ]

        if head_pts and shoulder_ys:
            head_y_min = min(p[1] for p in head_pts)
            shoulder_y = float(np.mean(shoulder_ys))
            # 头高 ≈ 鼻/眼 到 肩的距离
            face_span = max(20.0, shoulder_y - head_y_min)
            # 向上外推 ~0.55 个脸长，覆盖额头 + 发顶
            head_top_y = head_y_min - face_span * 0.55
            head_center_x = float(np.mean([p[0] for p in head_pts]))
        elif head_pts:
            head_y_min = min(p[1] for p in head_pts)
            head_top_y = head_y_min - 40.0  # 兜底偏移
            head_center_x = float(np.mean([p[0] for p in head_pts]))
        else:
            head_top_y = y1
            head_center_x = (x1 + x2) / 2.0

        # 修正整体 bbox 的上边界（别漏掉头发）
        y1 = min(y1, head_top_y)

        return PersonBox(
            x1=int(max(0, x1)),
            y1=int(max(0, y1)),
            x2=int(min(w, x2)),
            y2=int(min(h, y2)),
            head_top_y=int(max(0, head_top_y)),
            head_center_x=int(max(0, min(w, head_center_x))),
            source="pose",
        )

    def _detect_face(self, rgb: np.ndarray) -> Optional[PersonBox]:
        h, w = rgb.shape[:2]
        result = self._face.process(rgb)
        if not result.detections:
            return None

        # 选面积最大的人脸
        best = max(
            result.detections,
            key=lambda d: d.location_data.relative_bounding_box.width
            * d.location_data.relative_bounding_box.height,
        )
        rb = best.location_data.relative_bounding_box
        fx = rb.xmin * w
        fy = rb.ymin * h
        fw = rb.width * w
        fh = rb.height * h

        # 脸框上方预留 0.35 个脸高的头发/发型空间
        head_top_y = max(0.0, fy - fh * 0.35)
        head_center_x = fx + fw / 2.0

        # 身体范围无法知道，用脸框粗略外推：宽 3 倍脸宽，高向下 6 倍脸高
        body_x1 = max(0.0, head_center_x - fw * 1.5)
        body_x2 = min(float(w), head_center_x + fw * 1.5)
        body_y2 = min(float(h), fy + fh * 7.0)

        return PersonBox(
            x1=int(body_x1),
            y1=int(head_top_y),
            x2=int(body_x2),
            y2=int(body_y2),
            head_top_y=int(head_top_y),
            head_center_x=int(head_center_x),
            source="face",
        )
