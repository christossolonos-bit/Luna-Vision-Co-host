from __future__ import annotations

import base64
import io
import logging
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
from luna.league_client import LeagueClient
from luna.screen import CaptureResult, ScreenCapture
from luna.speech import SpeechRecognizer
from luna.twitch_persona import build_idle_user_message, build_twitch_system_prompt, is_creator_login, pick_idle_topic
from luna.video_buffer import ScreenVideoBuffer
from luna.voice_tags import strip_tags_for_display
from luna.twitch_chat import TwitchChatBot
from luna.voice import create_voice
from luna.voice_mood import list_moods, normalize_mood

STATIC_DIR = Path(__file__).resolve().parent / "static"
logger = logging.getLogger(__name__)


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
    mood: str | None = None


class MoodRequest(BaseModel):
    mood: str


class ObsTTSRequest(BaseModel):
    audio_b64: str


class ScreenCaptureRequest(BaseModel):
    enabled: bool


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
        twitch_prompt = build_twitch_system_prompt(config.twitch) if config.twitch.enabled else None
        self.twitch_brain = CohostBrain(
            config,
            system_prompt_override=twitch_prompt,
        )
        self.league = LeagueClient(config.league, config.cohost.player_name)
        self.screen = ScreenCapture(config.screen)
        self.video_buffer = ScreenVideoBuffer(self.screen, config.screen)
        self.voice = create_voice(config.voice)
        self.speech = SpeechRecognizer(config.speech)
        self._watch_lock = threading.Lock()
        self._watch_running = False
        self._idle_topic_index = 0
        self._last_frame_b64: str | None = None
        self._capture_source = f"monitor:{config.screen.monitor}"
        self._twitch_status = "Twitch: off"
        self._screen_capture_enabled = config.screen.capture_enabled

    @property
    def screen_capture_enabled(self) -> bool:
        return self._screen_capture_enabled

    def set_screen_capture(self, enabled: bool) -> None:
        self._screen_capture_enabled = enabled
        if enabled:
            self.video_buffer.start()
            logger.info("Screen capture enabled — background vision buffer started")
        else:
            self.video_buffer.stop()
            logger.info("Screen capture disabled — background vision buffer stopped")

    def status(self) -> dict[str, str | bool]:
        ok, message = self.brain.ping()
        league_line = self.league.status_line()
        if league_line:
            message = f"{message} | {league_line}"
        if self.config.twitch.enabled:
            message = f"{message} | {self._twitch_status}"
        snap = self.league.snapshot()
        return {
            "ok": ok,
            "message": message,
            "league_active": snap.active,
            "league_phase": snap.phase,
            "twitch_enabled": self.config.twitch.enabled,
            "screen_capture_enabled": self.screen_capture_enabled,
        }

    def set_twitch_status(self, message: str) -> None:
        self._twitch_status = message

    def twitch_chat_reply(self, username: str, message: str) -> str:
        league_block = self._league_block()
        league_context = bool(league_block)
        creator_note = ", stream creator" if is_creator_login(username, self.config.twitch) else ""

        image_b64 = None
        prompt_parts: list[str] = []
        if self.config.twitch.use_screen and self.screen_capture_enabled:
            image_b64, vision_prefix = self._latest_vision_for_chat()
            if vision_prefix:
                prompt_parts.append(vision_prefix.rstrip())
        elif league_block:
            prompt_parts.append(league_block)

        speaker = f"{username}{creator_note}"
        use_vision = self.config.twitch.use_screen and self.screen_capture_enabled
        prompt_parts.append(
            self.brain.wrap_vision_user_message(message.strip(), speaker=speaker)
            if use_vision
            else f"[Twitch chat — {speaker}]: {message.strip()}"
        )
        prompt = "\n\n".join(prompt_parts)

        return self.twitch_brain.ask(
            prompt,
            image_b64,
            remember=True,
            include_history=True,
            history_user_text=f"{username}: {message.strip()}",
            league_context=league_context,
        )

    def twitch_idle_talk(self) -> str:
        topic, self._idle_topic_index = pick_idle_topic(self._idle_topic_index)
        league_block = self._league_block()
        league_context = bool(league_block)
        image_b64 = None
        prompt_parts: list[str] = []

        use_screen = (
            self.screen_capture_enabled
            and self.config.twitch.idle_talk_use_screen
            and self.config.twitch.use_screen
        )
        if use_screen:
            image_b64, vision_prefix = self._latest_vision_for_chat()
            if vision_prefix:
                prompt_parts.append(vision_prefix.rstrip())
        elif league_block:
            prompt_parts.append(league_block)

        prompt_parts.append(build_idle_user_message(topic))
        prompt = "\n\n".join(prompt_parts)

        return self.twitch_brain.ask(
            prompt,
            image_b64,
            remember=True,
            include_history=True,
            history_user_text="[Chat quiet — Luna speaks unprompted]",
            league_context=league_context,
        )

    def _remember_capture_source(self, capture_source: str) -> None:
        if capture_source.strip():
            self._capture_source = capture_source.strip()

    def _latest_vision_for_chat(self) -> tuple[str | None, str]:
        try:
            result = self._vision_capture(self._capture_source)
            league_block = self._league_block()
            prefix = self._prompt_prefix(result, league_block=league_block)
            return result.image_b64, prefix
        except Exception:  # noqa: BLE001
            return None, ""

    def speak_text_for_obs(self, text: str) -> tuple[str | None, str | None]:
        return self._speak_if_needed(text, speak=True)

    def capture_sources(self) -> list[dict[str, str]]:
        return self.screen.list_capture_sources()

    def capture_preview(self, source: str) -> dict[str, str | bool]:
        if not self.screen_capture_enabled:
            return {"image_b64": "", "label": "Screen capture off", "is_blank": True}
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
        if not self.screen_capture_enabled:
            raise RuntimeError("Screen capture is off. Turn it on to use vision.")
        self.video_buffer.set_source(capture_source)
        if not self.video_buffer.running:
            self.video_buffer.start()
        result = self.video_buffer.get_latest_frame()
        if result is None:
            result = self.screen.capture_source(capture_source)
        self._last_frame_b64 = result.image_b64
        return result

    def _vision_instruction(self) -> str:
        return (
            "Use this screenshot as evidence for your answer. Focus on the main central "
            "area. Do not narrate the whole screen — only cite details relevant to what "
            "the player asked or said."
        )

    def _vision_prefix(self, result: CaptureResult) -> str:
        if result.is_blank:
            return (
                f"[WARNING: Capture from '{result.source_label}' is black or unreadable. "
                "Windows often cannot capture fullscreen games by window handle. "
                "Ask the player to switch to borderless windowed and capture the monitor "
                "that shows the game, or verify the correct screen is selected. "
                "Do not invent game details.]\n"
            )
        return f"[Live capture: {result.source_label}]\n{self._vision_instruction()}\n"

    def _prompt_prefix(
        self,
        result: CaptureResult | None = None,
        *,
        league_block: str | None = None,
    ) -> str:
        parts: list[str] = []
        block = league_block if league_block is not None else self.league.context_block_for_vision()
        if block:
            parts.append(block)
        if result is not None:
            parts.append(self._vision_prefix(result).rstrip())
        return "\n\n".join(part for part in parts if part) + ("\n" if parts else "")

    def _league_block(self) -> str:
        return self.league.context_block_for_vision()

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
        self._remember_capture_source(capture_source)
        league_block = self._league_block()
        league_context = bool(league_block)

        if use_screen and self.screen_capture_enabled:
            result = self._vision_capture(capture_source)
            image_b64 = result.image_b64
            meta = self._capture_response_meta(result)
            prompt = (
                self._prompt_prefix(result, league_block=league_block)
                + self.brain.wrap_vision_user_message(message.strip())
            )
        else:
            prefix = self._prompt_prefix(league_block=league_block)
            if prefix:
                prompt = prefix + prompt

        try:
            reply = self.brain.ask(
                prompt,
                image_b64,
                history_user_text=message.strip(),
                league_context=league_context,
            )
            audio_b64, audio_format = self._speak_if_needed(reply, speak)
            return {
                "reply": strip_tags_for_display(reply, enabled=self.config.voice.use_delivery_tags),
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
        if not self.screen_capture_enabled:
            return {
                "reply": "Screen capture is off — turn it on in the dock to analyze your screen.",
                "audio_b64": None,
                "audio_format": None,
                "error": True,
            }
        self.config.cohost.style = style
        self._remember_capture_source(capture_source)
        result = self._vision_capture(capture_source)
        league_block = self._league_block()
        league_context = bool(league_block)
        prompt = self._prompt_prefix(result, league_block=league_block) + self.brain.screen_commentary_prompt(
            mode="analyze",
            league_context=league_context,
        )
        try:
            reply = self.brain.ask(
                prompt,
                result.image_b64,
                history_user_text="[Analyze screen]",
                league_context=league_context,
            )
            audio_b64, audio_format = self._speak_if_needed(reply, speak)
            return {
                "reply": strip_tags_for_display(reply, enabled=self.config.voice.use_delivery_tags),
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
        if not self.screen_capture_enabled:
            return {"reply": "", "audio_b64": None, "skipped": True}
        with self._watch_lock:
            if self._watch_running:
                return {"reply": "", "audio_b64": None, "skipped": True}
            self._watch_running = True

        try:
            self.config.cohost.style = style
            self._remember_capture_source(capture_source)
            result: CaptureResult | None = None
            result = self._vision_capture(capture_source)
            league_block = self._league_block()
            league_context = bool(league_block)
            prompt = self._prompt_prefix(result, league_block=league_block) + self.brain.screen_commentary_prompt(
                mode="watch",
                league_context=league_context,
            )
            reply = self.brain.ask(
                prompt,
                result.image_b64,
                remember=False,
                include_history=True,
                league_context=league_context,
            )
            audio_b64, audio_format = self._speak_if_needed(reply, speak)
            return {
                "reply": strip_tags_for_display(reply, enabled=self.config.voice.use_delivery_tags),
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

    def tts(self, text: str, mood: str | None = None) -> bytes:
        return self.voice.synthesize(text, mood=mood)

    def set_voice_mood(self, mood: str) -> str:
        self.voice.set_mood(mood)
        return normalize_mood(mood, self.config.voice.default_mood)

    def get_voice_mood(self) -> str:
        getter = getattr(self.voice, "get_mood", None)
        if callable(getter):
            return getter()
        return self.config.voice.default_mood

    def reset(self) -> None:
        self.brain.reset()
        self.twitch_brain.reset()

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
    twitch_bot = TwitchChatBot(
        cfg.twitch,
        service,
        obs_relay,
        on_status=service.set_twitch_status,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if service.screen_capture_enabled:
            service.video_buffer.start()
        service.voice.start()
        await twitch_bot.start()
        yield
        await twitch_bot.stop()
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

    @app.post("/api/screen-capture")
    async def api_screen_capture(body: ScreenCaptureRequest) -> dict[str, bool]:
        service.set_screen_capture(body.enabled)
        return {"enabled": service.screen_capture_enabled}

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
        audio = service.tts(body.text, mood=body.mood)
        media_type = "audio/wav" if service.voice.audio_format == "wav" else "audio/mpeg"
        return Response(content=audio, media_type=media_type)

    @app.get("/api/voice/moods")
    async def api_voice_moods() -> dict[str, list[str] | str | bool]:
        return {
            "moods": list_moods(),
            "current": service.get_voice_mood(),
            "provider": cfg.voice.provider,
            "auto_mood": cfg.voice.auto_mood,
        }

    @app.post("/api/voice/mood")
    async def api_voice_mood(body: MoodRequest) -> dict[str, str]:
        mood = service.set_voice_mood(body.mood)
        return {"mood": mood}

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
    async def api_config() -> dict[str, str | float | bool | list[str]]:
        return {
            "name": cfg.cohost.name,
            "player_name": cfg.cohost.player_name,
            "style": cfg.cohost.style,
            "voice": cfg.voice.edge_voice,
            "voice_provider": cfg.voice.provider,
            "voice_mood": service.get_voice_mood(),
            "voice_moods": list_moods(),
            "watch_interval_sec": cfg.screen.capture_interval_sec,
            "screen_capture_enabled": service.screen_capture_enabled,
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
