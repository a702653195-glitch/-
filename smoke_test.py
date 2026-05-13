"""用于 CI 烟雾测试：PyInstaller 打包后仍能正常访问 mediapipe.solutions.

单独打成一个 console exe，成功则退出 0 并打印 SMOKE_OK。
"""
import sys


def main() -> int:
    try:
        import mediapipe as mp
        import mediapipe.python.solutions.pose  # noqa: F401
        import mediapipe.python.solutions.face_detection  # noqa: F401

        if not hasattr(mp, "solutions"):
            from mediapipe.python import solutions as _solutions
            mp.solutions = _solutions  # type: ignore[attr-defined]

        # 访问 API，确认 solutions namespace 真正可用
        _ = mp.solutions.pose.Pose
        _ = mp.solutions.face_detection.FaceDetection

        import cv2
        import numpy as np

        # 顺带验证 cv2 基础函数
        _ = cv2.imdecode(np.zeros(0, dtype=np.uint8), cv2.IMREAD_COLOR)

        # 还验证一下我们自己的模块
        from app.cropper import plan_crop  # noqa: F401
        from app.detector import PersonDetector  # noqa: F401

        print("SMOKE_OK: mediapipe.solutions accessible, cv2 works, app modules import")
        return 0
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"SMOKE_FAIL: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
