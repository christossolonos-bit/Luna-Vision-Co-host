from __future__ import annotations

import concurrent.futures
import logging
import os
import tempfile
import threading
from pathlib import Path

from luna.config import VoiceConfig
from luna.voice import clean_for_speech
from luna.voice_fx import apply_wav_fx
from luna.voice_mood import detect_mood, mood_params, normalize_mood
from luna.voice_tags import (
    TaggedSegment,
    concat_wav_chunks,
    has_delivery_tags,
    parse_tagged_speech,
    silence_wav,
)

logger = logging.getLogger(__name__)


class XTTSVoice:
    audio_format = "wav"

    def __init__(self, config: VoiceConfig) -> None:
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        self.config = config
        self._reference = Path(config.reference_audio)
        self._mood = normalize_mood(config.default_mood)
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._load_error: str | None = None
        self._load_started = False
        self._tts = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="xtts",
        )

    def set_mood(self, mood: str) -> None:
        self._mood = normalize_mood(mood, self.config.default_mood)

    def get_mood(self) -> str:
        return self._mood

    def start(self) -> None:
        with self._lock:
            if self._load_started:
                return
            self._load_started = True
            self._executor.submit(self._load_model)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _load_model(self) -> None:
        try:
            if not self._reference.exists():
                raise FileNotFoundError(f"XTTS reference audio not found: {self._reference}")

            import torch
            from TTS.api import TTS

            use_gpu = self.config.gpu and torch.cuda.is_available()
            device = "cuda" if use_gpu else "cpu"
            logger.info("Loading XTTS on %s from %s", device, self.config.model_name)
            model = TTS(self.config.model_name, gpu=use_gpu, progress_bar=False)
            if use_gpu:
                model = model.to(device)
            self._tts = model
            logger.info("XTTS ready — reference voice: %s", self._reference.name)
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            logger.exception("XTTS failed to load: %s", exc)
        finally:
            self._ready.set()

    def _ensure_model(self) -> None:
        if not self._load_started:
            self.start()
        if not self._ready.wait(timeout=600):
            raise TimeoutError("Timed out waiting for XTTS to load.")
        if self._load_error:
            raise RuntimeError(f"XTTS is not available: {self._load_error}")
        if self._tts is None:
            raise RuntimeError("XTTS model failed to load.")

    def _resolve_mood(self, text: str, mood: str | None) -> str:
        if mood:
            return normalize_mood(mood, self.config.default_mood)
        if self.config.auto_mood:
            return detect_mood(text, default=self._mood)
        return self._mood

    def _segments_for(self, text: str, mood: str | None) -> list[TaggedSegment]:
        base_mood = self._resolve_mood(text, mood)
        if self.config.use_delivery_tags and has_delivery_tags(text):
            return parse_tagged_speech(text, default_mood=base_mood)
        cleaned = clean_for_speech(text)
        if not cleaned:
            return []
        return [TaggedSegment(text=cleaned, mood=base_mood, speed_mult=1.0)]

    def _render_segment(self, segment: TaggedSegment, output_path: Path) -> bytes | None:
        if segment.pause_sec > 0:
            return None

        cleaned = clean_for_speech(segment.text)
        if not cleaned:
            return None

        params = mood_params(segment.mood, self.config.default_mood)
        speed = max(0.5, min(2.0, params.speed * self.config.speed * segment.speed_mult))

        assert self._tts is not None
        self._tts.tts_to_file(
            text=cleaned,
            speaker_wav=str(self._reference),
            language=self.config.language,
            file_path=str(output_path),
            split_sentences=False,
            temperature=params.temperature,
            speed=speed,
            top_p=params.top_p,
            repetition_penalty=params.repetition_penalty,
        )
        payload = output_path.read_bytes()
        if (
            abs(segment.pitch_semitones) >= 0.05
            or abs(segment.gain_db) >= 0.05
            or abs(segment.speed_mult - 1.0) >= 0.02
        ):
            payload = apply_wav_fx(
                payload,
                pitch_semitones=segment.pitch_semitones,
                gain_db=segment.gain_db,
                speed=1.0,
            )
        return payload

    def _synthesize_blocking(self, text: str, mood: str | None = None) -> bytes:
        self._ensure_model()
        segments = self._segments_for(text, mood)
        if not segments:
            return b""

        chunks: list[bytes] = []
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)

        try:
            for segment in segments:
                if segment.pause_sec > 0:
                    if chunks:
                        chunks.append(silence_wav(segment.pause_sec, chunks[-1]))
                    continue

                payload = self._render_segment(segment, temp_path)
                if payload:
                    chunks.append(payload)

            if not chunks:
                return b""
            return concat_wav_chunks(chunks)
        finally:
            temp_path.unlink(missing_ok=True)

    def synthesize(self, text: str, mood: str | None = None) -> bytes:
        with self._lock:
            future = self._executor.submit(self._synthesize_blocking, text, mood)
            return future.result(timeout=300)
