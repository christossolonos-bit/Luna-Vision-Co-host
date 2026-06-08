from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import mss
import win32gui
import win32ui
from ctypes import windll
from PIL import Image

from luna.config import ScreenConfig
from luna.windows import list_game_windows

PW_RENDERFULLCONTENT = 2


@dataclass
class CaptureResult:
    image_b64: str
    preview: Image.Image
    source_label: str
    is_blank: bool = False
    preview_b64: str = ""


class ScreenCapture:
    def __init__(self, config: ScreenConfig) -> None:
        self.config = config

    def list_monitors(self) -> list[str]:
        with mss.mss() as sct:
            labels: list[str] = []
            for index, monitor in enumerate(sct.monitors):
                if index == 0:
                    continue
                labels.append(
                    f"Screen {index} ({monitor['width']}x{monitor['height']})"
                )
            return labels

    def list_capture_sources(self) -> list[dict[str, str]]:
        sources: list[dict[str, str]] = []
        with mss.mss() as sct:
            for index in range(1, len(sct.monitors)):
                monitor = sct.monitors[index]
                sources.append(
                    {
                        "id": f"monitor:{index}",
                        "label": f"Screen {index} ({monitor['width']}x{monitor['height']})",
                    }
                )

        for window in list_game_windows():
            sources.append(
                {
                    "id": str(window["id"]),
                    "label": f"Game — {window['title']}",
                }
            )
        return sources

    def capture(self, source: str | None = None, monitor_index: int | None = None) -> CaptureResult:
        if source:
            return self.capture_source(source)
        index = monitor_index if monitor_index is not None else self.config.monitor
        return self.capture_source(f"monitor:{index}")

    def capture_source(self, source: str) -> CaptureResult:
        kind, _, value = source.partition(":")
        if kind == "monitor":
            return self._capture_monitor(int(value))
        if kind == "window":
            return self._capture_window(int(value))
        raise ValueError(f"Unknown capture source: {source}")

    def _capture_monitor(self, monitor_index: int) -> CaptureResult:
        with mss.mss() as sct:
            if monitor_index < 0 or monitor_index >= len(sct.monitors):
                raise ValueError(f"Monitor index {monitor_index} is out of range.")

            shot = sct.grab(sct.monitors[monitor_index])
            image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            label = self.list_monitors()[monitor_index - 1]
            return self._finalize(image, label)

    def _capture_window(self, hwnd: int) -> CaptureResult:
        if not win32gui.IsWindow(hwnd):
            raise ValueError("Selected game window is no longer open.")

        title = win32gui.GetWindowText(hwnd).strip() or f"Window {hwnd}"
        image = self._capture_window_image(hwnd)
        return self._finalize(image, f"Game — {title}")

    def _window_rect(self, hwnd: int) -> tuple[int, int, int, int]:
        try:
            rect = win32gui.GetWindowRect(hwnd)
            left, top, right, bottom = rect
            if right > left and bottom > top:
                return left, top, right, bottom
        except Exception:  # noqa: BLE001
            pass
        return win32gui.GetWindowRect(hwnd)

    def _capture_window_image(self, hwnd: int) -> Image.Image:
        left, top, right, bottom = self._window_rect(hwnd)
        width = max(1, right - left)
        height = max(1, bottom - top)

        best_image: Image.Image | None = None
        best_score = -1.0

        for flag in (PW_RENDERFULLCONTENT, 0):
            try:
                image = self._capture_window_printwindow(hwnd, width, height, flag)
                if image is None:
                    continue
                score = self._content_score(image)
                if score > best_score:
                    best_score = score
                    best_image = image
                if score > 0.2:
                    return image
            except Exception:  # noqa: BLE001
                continue

        with mss.mss() as sct:
            shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
            mss_image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            score = self._content_score(mss_image)
            if score > best_score:
                best_image = mss_image

        if best_image is not None:
            return best_image

        with mss.mss() as sct:
            shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    def _capture_window_printwindow(
        self,
        hwnd: int,
        width: int,
        height: int,
        flag: int,
    ) -> Image.Image | None:
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        if not hwnd_dc:
            return None

        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bitmap)

        result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), flag)
        if result != 1:
            win32gui.DeleteObject(bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            return None

        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)
        image = Image.frombuffer(
            "RGB",
            (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
            bmpstr,
            "raw",
            "BGRX",
            0,
            1,
        )

        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        return image

    def _content_score(self, image: Image.Image) -> float:
        sample = image.convert("L")
        sample = sample.resize((min(320, sample.width), min(180, sample.height)))
        pixels = list(sample.getdata())
        if not pixels:
            return 0.0
        dark = sum(1 for value in pixels if value < 24)
        flat = sum(1 for value in pixels if 24 <= value < 40)
        usable = len(pixels) - dark - flat
        return usable / len(pixels)

    def is_mostly_blank(self, image: Image.Image) -> bool:
        return self._content_score(image) < 0.08

    def from_image(self, image: Image.Image, label: str) -> CaptureResult:
        return self._finalize(image, label)

    def _finalize(self, image: Image.Image, label: str) -> CaptureResult:
        preview = image.copy()
        is_blank = self.is_mostly_blank(preview)
        resized = self._resize(image)
        buffer = io.BytesIO()
        resized.save(
            buffer,
            format="JPEG",
            quality=self.config.jpeg_quality,
            optimize=True,
        )
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        thumb = preview.copy()
        thumb.thumbnail((320, 180), Image.Resampling.BILINEAR)
        thumb_buffer = io.BytesIO()
        thumb.save(thumb_buffer, format="JPEG", quality=55)
        preview_b64 = base64.b64encode(thumb_buffer.getvalue()).decode("ascii")

        return CaptureResult(
            image_b64=encoded,
            preview=preview,
            source_label=label,
            is_blank=is_blank,
            preview_b64=preview_b64,
        )

    def _resize(self, image: Image.Image) -> Image.Image:
        max_width = self.config.max_width
        if image.width <= max_width:
            return image

        ratio = max_width / image.width
        new_size = (max_width, max(1, int(image.height * ratio)))
        return image.resize(new_size, Image.Resampling.BILINEAR)


def shrink_image_b64(image_b64: str, max_width: int = 640, jpeg_quality: int = 60) -> str:
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, max(1, int(image.height * ratio))), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
