"""批量处理调度器：多线程 + 进度回调。"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .cropper import CropConfig, crop_to_2x3, verify_ratio
from .detector import PersonDetector
from .utils import (
    derive_output_path,
    ensure_dir,
    imread_unicode,
    imwrite_unicode,
    iter_images,
)

# 每个线程持有一个独立的 PersonDetector 实例（MediaPipe 非线程安全）
_thread_local = threading.local()


def _get_detector(model_complexity: int) -> PersonDetector:
    det: Optional[PersonDetector] = getattr(_thread_local, "detector", None)
    if det is None:
        det = PersonDetector(model_complexity=model_complexity)
        _thread_local.detector = det
    return det


@dataclass
class JobResult:
    source: Path
    target: Path
    ok: bool
    message: str = ""
    detect_source: str = ""     # 'pose' / 'face' / 'fallback'
    elapsed_ms: float = 0.0


@dataclass
class BatchStats:
    total: int = 0
    done: int = 0
    ok: int = 0
    failed: int = 0
    results: List[JobResult] = field(default_factory=list)


def process_single(
    src: Path,
    dst: Path,
    crop_config: CropConfig,
    model_complexity: int = 1,
    jpeg_quality: int = 95,
    overwrite: bool = True,
) -> JobResult:
    """处理单张图片。线程安全。"""
    t0 = time.perf_counter()

    if dst.exists() and not overwrite:
        return JobResult(src, dst, False, "目标已存在，已跳过", elapsed_ms=0)

    img = imread_unicode(src)
    if img is None:
        return JobResult(
            src, dst, False, "无法读取（文件损坏或格式不支持）",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    try:
        detector = _get_detector(model_complexity)
        person = detector.detect(img)
        cropped = crop_to_2x3(img, person, crop_config)
    except Exception as e:  # noqa: BLE001
        return JobResult(
            src, dst, False, f"处理失败: {e}",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    if not verify_ratio(cropped):
        return JobResult(
            src, dst, False,
            f"输出比例校验失败 {cropped.shape[1]}x{cropped.shape[0]}",
            detect_source=(person.source if person else "fallback"),
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    ensure_dir(dst.parent)
    ok = imwrite_unicode(dst, cropped, quality=jpeg_quality)
    elapsed = (time.perf_counter() - t0) * 1000

    return JobResult(
        src, dst, ok,
        "" if ok else "写入失败",
        detect_source=(person.source if person else "fallback"),
        elapsed_ms=elapsed,
    )


def process_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    recursive: bool = True,
    max_workers: int = 4,
    model_complexity: int = 1,
    jpeg_quality: int = 95,
    overwrite: bool = True,
    head_margin_ratio: float = 0.08,
    progress_cb: Optional[Callable[[int, int, JobResult], None]] = None,
    cancel_flag: Optional[threading.Event] = None,
) -> BatchStats:
    """批量处理入口。

    Args:
        progress_cb: 每完成一张触发，签名 (done, total, result)。
        cancel_flag: 设置该 Event 可中途取消。
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    files = iter_images(input_dir, recursive=recursive)

    stats = BatchStats(total=len(files))
    if not files:
        return stats

    crop_config = CropConfig(head_margin_ratio=head_margin_ratio)

    def _worker(src: Path) -> JobResult:
        if cancel_flag and cancel_flag.is_set():
            return JobResult(src, src, False, "已取消")
        dst = derive_output_path(src, input_dir, output_dir, preserve_tree=recursive)
        return process_single(
            src, dst, crop_config,
            model_complexity=model_complexity,
            jpeg_quality=jpeg_quality,
            overwrite=overwrite,
        )

    # max_workers 上限：避免在 CPU 受限机器上过度抢占 MediaPipe 内部线程池
    max_workers = max(1, min(max_workers, 16))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, f): f for f in files}
        for fut in as_completed(futures):
            result = fut.result()
            stats.done += 1
            if result.ok:
                stats.ok += 1
            else:
                stats.failed += 1
            stats.results.append(result)
            if progress_cb:
                try:
                    progress_cb(stats.done, stats.total, result)
                except Exception:
                    pass
            if cancel_flag and cancel_flag.is_set():
                # 剩下的未完成任务仍会跑完当前 worker 线程中的工作
                # （线程池没有强行中断，但新任务会很快返回"已取消"）
                break

    return stats


def cleanup_thread_local_detectors() -> None:
    """在程序退出前清理线程本地的 MediaPipe 资源。
    注意：只能清理当前线程的实例，工作线程的资源会随进程退出而释放。
    """
    det: Optional[PersonDetector] = getattr(_thread_local, "detector", None)
    if det is not None:
        det.close()
        _thread_local.detector = None
