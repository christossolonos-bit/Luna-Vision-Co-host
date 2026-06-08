from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from luna.config import ScreenConfig
from luna.screen import CaptureResult, ScreenCapture


class ScreenVideoBuffer:
    """Records short silent screen clips and serves the latest frame for vision."""

    def __init__(self, screen: ScreenCapture, config: ScreenConfig) -> None:
        self.screen = screen
        self.config = config
        self._source = f"monitor:{config.monitor}"
        self._source_label = ""
        self._segments: deque[Path] = deque()
        self._work_dir = Path(tempfile.gettempdir()) / "luna-vision-stream"
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._segment_index = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_source(self, source: str) -> None:
        with self._lock:
            if source != self._source:
                self._source = source
                self._source_label = ""

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._record_loop,
            name="luna-video-buffer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def get_latest_frame(self) -> CaptureResult | None:
        video_path = self._newest_segment()
        if video_path and video_path.exists():
            try:
                image = self._read_last_frame(video_path)
                label = self._source_label or video_path.name
                return self.screen.from_image(image, f"Video stream — {label}")
            except Exception:  # noqa: BLE001
                pass

        try:
            return self.screen.capture_source(self._source)
        except Exception:  # noqa: BLE001
            return None

    def _newest_segment(self) -> Path | None:
        with self._lock:
            if not self._segments:
                return None
            return self._segments[-1]

    def _record_loop(self) -> None:
        while not self._stop.is_set():
            started = time.time()
            try:
                self._record_segment()
            except Exception:  # noqa: BLE001
                pass
            elapsed = time.time() - started
            sleep_for = max(0.05, self.config.video_segment_seconds - elapsed)
            if self._stop.wait(sleep_for):
                break

    def _record_segment(self) -> None:
        source = self._source
        fps = max(2, self.config.video_fps)
        duration = max(0.5, self.config.video_segment_seconds)
        frame_count = max(2, int(round(fps * duration)))
        frames: list[Image.Image] = []

        interval = 1.0 / fps
        for _ in range(frame_count):
            if self._stop.is_set():
                return
            result = self.screen.capture_source(source)
            self._source_label = result.source_label
            frames.append(result.preview.copy())
            if self._stop.wait(interval):
                return

        if len(frames) < 2:
            return

        path = self._work_dir / f"segment_{self._segment_index:06d}.mp4"
        self._segment_index += 1
        self._write_segment(frames, path, fps)
        if path.exists() and path.stat().st_size > 1024:
            self._register_segment(path)

    def _write_segment(self, frames: list[Image.Image], path: Path, fps: int) -> None:
        width, height = frames[0].size
        bitrate = max(256_000, self.config.video_bytes_per_second * 8)

        if shutil.which("ffmpeg"):
            self._write_segment_ffmpeg(frames, path, fps, width, height, bitrate)
            return

        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            return
        for frame in frames:
            bgr = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()

    def _write_segment_ffmpeg(
        self,
        frames: list[Image.Image],
        path: Path,
        fps: int,
        width: int,
        height: int,
        bitrate: int,
    ) -> None:
        duration = len(frames) / fps
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-b:v",
            str(bitrate),
            "-maxrate",
            str(bitrate),
            "-bufsize",
            str(max(bitrate // 2, 256_000)),
            "-t",
            f"{duration:.3f}",
            str(path),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert proc.stdin is not None
        try:
            for frame in frames:
                proc.stdin.write(frame.convert("RGB").tobytes())
        finally:
            proc.stdin.close()
        proc.wait(timeout=30)

    def _register_segment(self, path: Path) -> None:
        with self._lock:
            self._segments.append(path)
            while len(self._segments) > self.config.video_max_segments:
                old = self._segments.popleft()
                old.unlink(missing_ok=True)
            keep = set(self._segments)
            for candidate in self._work_dir.glob("segment_*.mp4"):
                if candidate not in keep:
                    candidate.unlink(missing_ok=True)

    @staticmethod
    def _read_last_frame(path: Path) -> Image.Image:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Cannot open video: {path}")

        frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_total > 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_total - 1)

        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        capture.release()

        if not ok or frame is None:
            raise ValueError(f"No frames in video: {path}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
