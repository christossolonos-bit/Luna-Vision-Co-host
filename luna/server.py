from __future__ import annotations

import base64
import io
import threading
import time
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from luna.brain import CohostBrain
from luna.config import AppConfig, load_config
from luna.screen import CaptureResult, ScreenCapture
from luna.speech import SpeechRecognizer
from luna.video_buffer import ScreenVideoBuffer
from luna.voice import EdgeVoice

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    message: str
    capture_source: str = "monitor:1"
    use_screen: bool = True
    style: str = "energetic"
    speak: bool = True


class AnalyzeRequest(BaseModel):
    capture_source: str = "monitor:1"
    style: str = "energetic"
    speak: bool = True


class TTSRequest(BaseModel):
    text: str


class ObsTTSRequest(BaseModel):
    audio_b64: str


class ObsRelay:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_heartbeat = 0.0
        self._pending_audio_b64: str | None = None

    def heartbeat(self) -> None:
        with self._lock:
            self._last_heartbeat = time.time()

    def disconnect(self) -> None:
        with self._lock:
            self._last_heartbeat = 0.0
            self._pending_audio_b64 = None

    def is_active(self) -> bool:
        with self._lock:
            return time.time() - self._last_heartbeat < 5.0

    def queue_tts(self, audio_b64: str) -> None:
        with self._lock:
            self._pending_audio_b64 = audio_b64

    def poll_tts(self) -> str | None:
        with self._lock:
            audio = self._pending_audio_b64
            self._pending_audio_b64 = None
            return audio


class LunaService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.brain = CohostBrain(config)
        self.screen = ScreenCapture(config.screen)
        self.video_buffer = ScreenVideoBuffer(self.screen, config.screen)
        self.voice = EdgeVoice(config.voice)
        self.speech = SpeechRecognizer(config.speech)
        self._watch_lock = threading.Lock()
        self._watch_running = False
        self._last_frame_b64: str | None = None

    def status(self) -> dict[str, str | bool]:
        ok, message = self.brain.ping()
        return {"ok": ok, "message": message}

    def capture_sources(self) -> list[dict[str, str]]:
        return self.screen.list_capture_sources()

    def capture_preview(self, source: str) -> dict[str, str | bool]:
        self.video_buffer.set_source(source)
        result = self.video_buffer.get_latest_frame()
        if result is None:
            result = self.screen.capture_source(source)
        self._last_frame_b64 = result.image_b64
        return {
            "image_b64": result.preview_b64,
            "label": result.source_label,
            "is_blank": result.is_blank,
        }

    def _vision_capture(self, capture_source: str) -> CaptureResult:
        self.video_buffer.set_source(capture_source)
        if not self.video_buffer.running:
            self.video_buffer.start()
        result = self.video_buffer.get_latest_frame()
        if result is None:
            result = self.screen.capture_source(capture_source)
        self._last_frame_b64 = result.image_b64
        return result

    def _vision_prefix(self, result: CaptureResult) -> str:
        if result.is_blank:
            return (
                f"[WARNING: Capture from '{result.source_label}' is black or unreadable. "
                "Windows often cannot capture fullscreen games by window handle. "
                "Ask the player to switch to borderless windowed and capture the monitor "
                "that shows the game, or verify the correct screen is selected. "
                "Do not invent game details.]\n"
            )
        return f"[Live capture: {result.source_label}]\n"

    def _capture_response_meta(self, result: CaptureResult) -> dict[str, str | bool]:
        return {
            "capture_label": result.source_label,
            "capture_warning": result.is_blank,
            "capture_preview_b64": result.preview_b64,
        }

    def chat(
        self,
        message: str,
        capture_source: str,
        use_screen: bool,
        style: str,
        speak: bool,
    ) -> dict[str, str | bool | None]:
        self.config.cohost.style = style
        image_b64 = None
        meta: dict[str, str | bool] = {}
        prompt = message.strip()

        if use_screen:
            result = self._vision_capture(capture_source)
            image_b64 = result.image_b64
            meta = self._capture_response_meta(result)
            prompt = self._vision_prefix(result) + prompt

        try:
            reply = self.brain.ask(prompt, image_b64)
            audio_b64, audio_format = self._speak_if_needed(reply, speak)
            return {
                "reply": reply,
                "audio_b64": audio_b64,
                "audio_format": audio_format,
                **meta,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "reply": self.brain.friendly_error(exc),
                "audio_b64": None,
                "audio_format": None,
                "error": True,
                **meta,
            }

    def analyze(self, capture_source: str, style: str, speak: bool) -> dict[str, str | bool | None]:
        self.config.cohost.style = style
        result = self._vision_capture(capture_source)
        prompt = self._vision_prefix(result) + (
            "Look at this frame from the live video stream taken just now. "
            "Describe only what is clearly visible."
        )
        try:
            reply = self.brain.ask(prompt, result.image_b64)
            audio_b64, audio_format = self._speak_if_needed(reply, speak)
            return {
                "reply": reply,
                "audio_b64": audio_b64,
                "audio_format": audio_format,
                **self._capture_response_meta(result),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "reply": self.brain.friendly_error(exc),
                "audio_b64": None,
                "audio_format": None,
                "error": True,
                **self._capture_response_meta(result),
            }

    def watch_tick(
        self,
        capture_source: str,
        style: str,
        speak: bool,
    ) -> dict[str, str | bool | None]:
        with self._watch_lock:
            if self._watch_running:
                return {"reply": "", "audio_b64": None, "skipped": True}
            self._watch_running = True

        try:
            self.config.cohost.style = style
            result: CaptureResult | None = None
            result = self._vision_capture(capture_source)
            prompt = self._vision_prefix(result) + (
                "You are watching live gameplay from a rolling video stream. "
                "Give a short co-host line about what is clearly visible. "
                "Do not invent details. Avoid repeating your last comment."
            )
            reply = self.brain.ask(prompt, result.image_b64)
            audio_b64, audio_format = self._speak_if_needed(reply, speak)
            return {
                "reply": reply,
                "audio_b64": audio_b64,
                "audio_format": audio_format,
                "timestamp": time.strftime("%H:%M:%S"),
                **self._capture_response_meta(result),
            }
        except Exception as exc:  # noqa: BLE001
            meta: dict[str, str | bool] = (
                self._capture_response_meta(result) if result is not None else {}
            )
            return {
                "reply": self.brain.friendly_error(exc),
                "audio_b64": None,
                "audio_format": None,
                "error": True,
                "timestamp": time.strftime("%H:%M:%S"),
                **meta,
            }
        finally:
            with self._watch_lock:
                self._watch_running = False

    def transcribe(self, payload: bytes, suffix: str) -> str:
        return self.speech.transcribe_bytes(payload, suffix=suffix)

    def tts(self, text: str) -> bytes:
        return self.voice.synthesize(text)

    def reset(self) -> None:
        self.brain.reset()

    def _speak_if_needed(self, text: str, speak: bool) -> tuple[str | None, str | None]:
        if not speak or not text.strip():
            return None, None
        audio = self.voice.synthesize(text)
        if not audio:
            return None, None
        return base64.b64encode(audio).decode("ascii"), self.voice.audio_format


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    service = LunaService(cfg)
    obs_relay = ObsRelay()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        service.video_buffer.start()
        yield
        service.video_buffer.stop()
        service.voice.shutdown()

    app = FastAPI(title="Luna Gaming Co-Host", lifespan=lifespan)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/obs")
    async def obs_overlay() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    async def api_status() -> dict[str, str | bool]:
        return service.status()

    @app.get("/api/capture-sources")
    async def api_capture_sources() -> list[dict[str, str]]:
        return service.capture_sources()

    @app.get("/api/preview")
    async def api_preview(source: str = "monitor:1") -> dict[str, str | bool]:
        return service.capture_preview(source)

    @app.post("/api/chat")
    async def api_chat(body: ChatRequest) -> dict[str, str | bool | None]:
        try:
            return service.chat(
                body.message,
                body.capture_source,
                body.use_screen,
                body.style,
                body.speak,
            )
        except Exception as exc:  # noqa: BLE001
            return {"reply": f"Error: {exc}", "audio_b64": None, "error": True}

    @app.post("/api/analyze")
    async def api_analyze(body: AnalyzeRequest) -> dict[str, str | bool | None]:
        try:
            return service.analyze(body.capture_source, body.style, body.speak)
        except Exception as exc:  # noqa: BLE001
            return {"reply": f"Error: {exc}", "audio_b64": None, "error": True}

    @app.post("/api/watch")
    async def api_watch(body: AnalyzeRequest) -> dict[str, str | bool | None]:
        try:
            return service.watch_tick(body.capture_source, body.style, body.speak)
        except Exception as exc:  # noqa: BLE001
            return {"reply": f"Error: {exc}", "audio_b64": None, "error": True}

    @app.post("/api/reset")
    async def api_reset() -> dict[str, bool]:
        service.reset()
        return {"ok": True}

    @app.post("/api/tts")
    async def api_tts(body: TTSRequest) -> Response:
        audio = service.voice.synthesize(body.text)
        return Response(content=audio, media_type="audio/mpeg")

    @app.post("/api/stt")
    async def api_stt(file: UploadFile = File(...)) -> dict[str, str]:
        suffix = Path(file.filename or "audio.webm").suffix or ".webm"
        payload = await file.read()
        text = service.transcribe(payload, suffix)
        return {"text": text}

    @app.get("/api/assets/vrm")
    async def api_vrm() -> FileResponse:
        path = Path(cfg.vrm.model_path)
        if not path.exists():
            raise FileNotFoundError(f"VRM not found: {path}")
        return FileResponse(path, media_type="model/gltf-binary", filename=path.name)

    @app.get("/api/assets/vrma")
    async def api_vrma() -> FileResponse:
        path = Path(cfg.vrm.idle_animation_path)
        if not path.exists():
            raise FileNotFoundError(f"VRMA not found: {path}")
        return FileResponse(path, media_type="application/octet-stream", filename=path.name)

    @app.get("/api/config")
    async def api_config() -> dict[str, str | float | bool]:
        return {
            "name": cfg.cohost.name,
            "player_name": cfg.cohost.player_name,
            "style": cfg.cohost.style,
            "voice": cfg.voice.edge_voice,
            "watch_interval_sec": cfg.screen.capture_interval_sec,
        }

    @app.post("/api/obs/heartbeat")
    async def obs_heartbeat() -> dict[str, bool]:
        obs_relay.heartbeat()
        return {"ok": True}

    @app.post("/api/obs/disconnect")
    async def obs_disconnect() -> dict[str, bool]:
        obs_relay.disconnect()
        return {"ok": True}

    @app.get("/api/obs/active")
    async def obs_active() -> dict[str, bool]:
        return {"active": obs_relay.is_active()}

    @app.post("/api/obs/tts")
    async def obs_queue_tts(body: ObsTTSRequest) -> dict[str, bool]:
        obs_relay.queue_tts(body.audio_b64)
        return {"ok": True}

    @app.get("/api/obs/tts")
    async def obs_poll_tts() -> dict[str, str | None]:
        return {"audio_b64": obs_relay.poll_tts()}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.luna_service = service
    return app


def launch(config_path: str | None = None) -> None:
    import webbrowser

    import uvicorn

    config = load_config(config_path)
    app = create_app(config)
    service: LunaService = app.state.luna_service
    url = f"http://{config.ui.host}:{config.ui.port}/"
    webbrowser.open(url)
    uvicorn.run(app, host=config.ui.host, port=config.ui.port, log_level="info")
