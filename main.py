"""女装电商图片批量智能裁切工具 - Windows GUI 入口。

运行方式:
  python main.py

界面功能:
  - 选择输入 / 输出文件夹
  - 设置并发线程数、MediaPipe 模型复杂度、头顶留白比例、输出质量
  - 进度条 + 实时日志
  - 开始 / 取消 / 打开输出目录
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from app.processor import BatchStats, JobResult, process_batch
from app.utils import iter_images

APP_TITLE = "女装电商图片智能裁切 (2:3)"
APP_SIZE = "780x620"


class CropperApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(APP_SIZE)
        self.minsize(720, 560)

        # --- State --- #
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.workers = tk.IntVar(value=max(2, min(8, (os.cpu_count() or 4) // 2)))
        self.model_complexity = tk.IntVar(value=1)  # 0=Lite 1=Full 2=Heavy
        self.head_margin = tk.DoubleVar(value=0.08)
        self.quality = tk.IntVar(value=95)
        self.recursive = tk.BooleanVar(value=True)
        self.overwrite = tk.BooleanVar(value=True)

        self._cancel_flag: threading.Event | None = None
        self._worker_thread: threading.Thread | None = None
        self._log_queue: queue.Queue[str] = queue.Queue()

        self._build_ui()
        self.after(100, self._drain_log_queue)

    # ------------------------------------------------------------------ #
    # UI 布局
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # ---- 目录选择 ---- #
        frm_dir = ttk.LabelFrame(self, text="文件夹")
        frm_dir.pack(fill="x", **pad)

        ttk.Label(frm_dir, text="输入文件夹:").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        ttk.Entry(frm_dir, textvariable=self.input_dir).grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(frm_dir, text="浏览…", command=self._choose_input).grid(row=0, column=2, padx=6)

        ttk.Label(frm_dir, text="输出文件夹:").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        ttk.Entry(frm_dir, textvariable=self.output_dir).grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(frm_dir, text="浏览…", command=self._choose_output).grid(row=1, column=2, padx=6)

        frm_dir.columnconfigure(1, weight=1)

        # ---- 参数 ---- #
        frm_opt = ttk.LabelFrame(self, text="参数")
        frm_opt.pack(fill="x", **pad)

        ttk.Label(frm_opt, text="并发线程:").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        ttk.Spinbox(frm_opt, from_=1, to=16, width=6, textvariable=self.workers).grid(row=0, column=1, sticky="w")

        ttk.Label(frm_opt, text="模型精度:").grid(row=0, column=2, sticky="e", padx=6)
        ttk.Combobox(
            frm_opt, width=10, state="readonly",
            values=["Lite (快)", "Full (默认)", "Heavy (高精度)"],
            textvariable=self._model_name_var(),
        ).grid(row=0, column=3, sticky="w")

        ttk.Label(frm_opt, text="头顶留白比例:").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        ttk.Spinbox(
            frm_opt, from_=0.02, to=0.25, increment=0.01,
            width=6, textvariable=self.head_margin, format="%.2f",
        ).grid(row=1, column=1, sticky="w")

        ttk.Label(frm_opt, text="输出质量:").grid(row=1, column=2, sticky="e", padx=6)
        ttk.Spinbox(frm_opt, from_=60, to=100, width=6, textvariable=self.quality).grid(row=1, column=3, sticky="w")

        ttk.Checkbutton(frm_opt, text="包含子目录", variable=self.recursive).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=6, pady=6,
        )
        ttk.Checkbutton(frm_opt, text="覆盖已存在文件", variable=self.overwrite).grid(
            row=2, column=2, columnspan=2, sticky="w", padx=6, pady=6,
        )

        # ---- 控制按钮 ---- #
        frm_ctrl = ttk.Frame(self)
        frm_ctrl.pack(fill="x", **pad)

        self.btn_start = ttk.Button(frm_ctrl, text="开始处理", command=self._on_start)
        self.btn_start.pack(side="left", padx=4)

        self.btn_cancel = ttk.Button(frm_ctrl, text="取消", command=self._on_cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=4)

        self.btn_open = ttk.Button(frm_ctrl, text="打开输出目录", command=self._open_output)
        self.btn_open.pack(side="left", padx=4)

        # ---- 进度 ---- #
        frm_prog = ttk.Frame(self)
        frm_prog.pack(fill="x", **pad)
        self.progress = ttk.Progressbar(frm_prog, mode="determinate")
        self.progress.pack(fill="x", expand=True, padx=4, pady=2)
        self.lbl_status = ttk.Label(frm_prog, text="就绪")
        self.lbl_status.pack(anchor="w", padx=4)

        # ---- 日志 ---- #
        frm_log = ttk.LabelFrame(self, text="日志")
        frm_log.pack(fill="both", expand=True, **pad)
        self.log_text = tk.Text(frm_log, wrap="none", height=14, state="disabled")
        scroll_y = ttk.Scrollbar(frm_log, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll_y.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll_y.pack(side="right", fill="y")

    # ------------------------------------------------------------------ #
    # 参数绑定辅助
    # ------------------------------------------------------------------ #
    def _model_name_var(self) -> tk.StringVar:
        """把 Combobox 的字符串双向绑定到 self.model_complexity (0/1/2)。"""
        names = {0: "Lite (快)", 1: "Full (默认)", 2: "Heavy (高精度)"}
        rev = {v: k for k, v in names.items()}
        var = tk.StringVar(value=names[self.model_complexity.get()])

        def _on_change(*_):
            self.model_complexity.set(rev.get(var.get(), 1))

        var.trace_add("write", _on_change)
        return var

    # ------------------------------------------------------------------ #
    # 事件处理
    # ------------------------------------------------------------------ #
    def _choose_input(self) -> None:
        d = filedialog.askdirectory(title="选择输入文件夹")
        if d:
            self.input_dir.set(d)
            if not self.output_dir.get():
                # 默认输出到 <input>_cropped
                self.output_dir.set(str(Path(d).with_name(Path(d).name + "_cropped")))

    def _choose_output(self) -> None:
        d = filedialog.askdirectory(title="选择输出文件夹")
        if d:
            self.output_dir.set(d)

    def _open_output(self) -> None:
        out = self.output_dir.get()
        if not out or not Path(out).exists():
            messagebox.showinfo("提示", "输出目录尚未生成")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(out)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{out}"')
            else:
                os.system(f'xdg-open "{out}"')
        except Exception as e:  # noqa: BLE001
            messagebox.showwarning("打开失败", str(e))

    def _on_start(self) -> None:
        in_dir = self.input_dir.get().strip()
        out_dir = self.output_dir.get().strip()

        if not in_dir or not Path(in_dir).is_dir():
            messagebox.showwarning("提示", "请选择有效的输入文件夹")
            return
        if not out_dir:
            messagebox.showwarning("提示", "请选择输出文件夹")
            return
        if Path(out_dir).resolve() == Path(in_dir).resolve():
            messagebox.showwarning("提示", "输入和输出文件夹不能相同")
            return

        files = iter_images(in_dir, recursive=self.recursive.get())
        if not files:
            messagebox.showinfo("提示", "输入文件夹中没有可处理的图片 (jpg/png/webp/bmp)")
            return

        self._set_running(True)
        self.progress.configure(maximum=len(files), value=0)
        self._log_clear()
        self._log(f"共发现 {len(files)} 张图片，开始处理…")

        self._cancel_flag = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._run_batch, args=(in_dir, out_dir), daemon=True,
        )
        self._worker_thread.start()

    def _on_cancel(self) -> None:
        if self._cancel_flag is not None:
            self._cancel_flag.set()
            self._log("已请求取消…等待当前任务收尾")
            self.btn_cancel.configure(state="disabled")

    # ------------------------------------------------------------------ #
    # 后台任务
    # ------------------------------------------------------------------ #
    def _run_batch(self, in_dir: str, out_dir: str) -> None:
        def on_progress(done: int, total: int, result: JobResult) -> None:
            # 从后台线程安全地传递到 UI 线程
            self.after(0, self._update_progress, done, total, result)

        try:
            stats: BatchStats = process_batch(
                in_dir, out_dir,
                recursive=self.recursive.get(),
                max_workers=int(self.workers.get()),
                model_complexity=int(self.model_complexity.get()),
                jpeg_quality=int(self.quality.get()),
                overwrite=bool(self.overwrite.get()),
                head_margin_ratio=float(self.head_margin.get()),
                progress_cb=on_progress,
                cancel_flag=self._cancel_flag,
            )
        except Exception as e:  # noqa: BLE001
            self.after(0, self._log, f"[错误] {e}")
            self.after(0, self._set_running, False)
            return

        self.after(0, self._finish_batch, stats)

    def _update_progress(self, done: int, total: int, result: JobResult) -> None:
        self.progress.configure(value=done)
        self.lbl_status.configure(text=f"进度 {done}/{total}")
        status = "OK" if result.ok else "FAIL"
        tag = f"[{result.detect_source or '-'}]" if result.ok else "[x]"
        msg = result.message or ""
        self._log(
            f"{status} {tag} {result.source.name} "
            f"({result.elapsed_ms:.0f}ms){' - ' + msg if msg else ''}"
        )

    def _finish_batch(self, stats: BatchStats) -> None:
        self._log("-" * 50)
        self._log(f"完成: 总 {stats.total} / 成功 {stats.ok} / 失败 {stats.failed}")
        self.lbl_status.configure(
            text=f"完成 - 成功 {stats.ok}, 失败 {stats.failed}",
        )
        self._set_running(False)
        if stats.failed == 0 and stats.total > 0:
            messagebox.showinfo("完成", f"全部 {stats.ok} 张图片处理成功")
        elif stats.total > 0:
            messagebox.showwarning(
                "完成（部分失败）",
                f"成功 {stats.ok} 张，失败 {stats.failed} 张，详情见日志",
            )

    # ------------------------------------------------------------------ #
    # 状态 & 日志
    # ------------------------------------------------------------------ #
    def _set_running(self, running: bool) -> None:
        state_start = "disabled" if running else "normal"
        state_cancel = "normal" if running else "disabled"
        self.btn_start.configure(state=state_start)
        self.btn_cancel.configure(state=state_cancel)

    def _log(self, msg: str) -> None:
        self._log_queue.put(msg)

    def _log_clear(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self.log_text.configure(state="normal")
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)


def main() -> None:
    app = CropperApp()
    app.mainloop()


if __name__ == "__main__":
    main()
