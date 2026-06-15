from __future__ import annotations

import concurrent.futures
import logging
import threading

from luna.config import VoiceConfig
from luna.voice import clean_for_speech
from luna.voice_tags import has_delivery_tags, strip_delivery_tags, strip_tags_for_display

logger = logging.getLogger(__name__)


class ElevenLabsVoice:
    """Native ElevenLabs v3 TTS — tags like [whispers] and [laughs] pass through to the model."""

    audio_format = "mpeg"

    def __init__(self, config: VoiceConfig) -> None:
        if not config.elevenlabs_api_key.strip():
            raise ValueError("voice.elevenlabs_api_key is required when provider is elevenlabs")
        if not config.elevenlabs_voice_id.strip():
            raise ValueError("voice.elevenlabs_voice_id is required when provider is elevenlabs")

        self.config = config
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="elevenlabs",
        )

    def set_mood(self, mood: str) -> None:
        return

    def _prepare_text(self, text: str) -> str:
        cleaned = clean_for_speech(text)
        if not cleaned:
            return ""
        if self.config.use_delivery_tags and has_delivery_tags(cleaned):
            return cleaned
        return cleaned

    def _synthesize_blocking(self, text: str, mood: str | None = None) -> bytes:
        del mood
        prepared = self._prepare_text(text)
        if not prepared:
            return b""

        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=self.config.elevenlabs_api_key.strip())
        audio = client.text_to_speech.convert(
            text=prepared,
            voice_id=self.config.elevenlabs_voice_id.strip(),
            model_id=self.config.elevenlabs_model,
            output_format=self.config.elevenlabs_output_format,
        )
        if isinstance(audio, (bytes, bytearray)):
            return bytes(audio)
        return b"".join(audio)

    def synthesize(self, text: str, mood: str | None = None) -> bytes:
        with self._lock:
            future = self._executor.submit(self._synthesize_blocking, text, mood)
            return future.result(timeout=300)

    def start(self) -> None:
        return

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)