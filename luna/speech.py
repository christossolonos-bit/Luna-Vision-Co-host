from __future__ import annotations

import tempfile
from pathlib import Path

from faster_whisper import WhisperModel

from luna.config import SpeechConfig


class SpeechRecognizer:
    def __init__(self, config: SpeechConfig) -> None:
        self.config = config
        self._model: WhisperModel | None = None

    def _resolve_device(self) -> tuple[str, str]:
        device = self.config.device
        compute_type = self.config.compute_type
        if device != "auto":
            return device, compute_type

        # Prefer CPU so Whisper does not compete with Ollama vision on the GPU.
        return "cpu", compute_type

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            device, compute_type = self._resolve_device()
            self._model = WhisperModel(
                self.config.model,
                device=device,
                compute_type=compute_type,
            )
        return self._model

    def transcribe_file(self, path: Path | str) -> str:
        segments, _info = self.model.transcribe(
            str(path),
            beam_size=1,
            best_of=1,
            vad_filter=True,
            condition_on_previous_text=False,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
                "threshold": 0.45,
            },
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text

    def transcribe_bytes(self, payload: bytes, suffix: str = ".webm") -> str:
        if len(payload) < 4096:
            return ""

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        try:
            text = self.transcribe_file(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

        cleaned = text.strip()
        if len(cleaned) < 2:
            return ""
        return cleaned
