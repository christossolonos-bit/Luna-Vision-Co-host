from __future__ import annotations

import asyncio
import concurrent.futures
import io
import re
import threading

import edge_tts

from luna.config import VoiceConfig

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)
_TEXT_EMOTICON_RE = re.compile(
    r"(?:"
    r"[:;8=B][\-^']?[)\(DPp3oO/\\|>\]]+"
    r"|[)\(][\-^']?[):;DPp3oO/\\|>\]]+"
    r"|</?3"
    r"|<3"
    r"|\^[_\-.]*\^"
    r"|xD+|XD+"
    r")",
    flags=re.IGNORECASE,
)


def clean_for_speech(text: str) -> str:
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = _TEXT_EMOTICON_RE.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


class EdgeVoice:
    def __init__(self, config: VoiceConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="edge-tts",
        )

    async def _synthesize(self, text: str) -> bytes:
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.config.edge_voice,
            rate=self.config.rate,
            pitch=self.config.pitch,
        )
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    def _synthesize_blocking(self, text: str) -> bytes:
        return asyncio.run(self._synthesize(text))

    def synthesize(self, text: str) -> bytes:
        cleaned = clean_for_speech(text)
        if not cleaned:
            return b""

        with self._lock:
            future = self._executor.submit(self._synthesize_blocking, cleaned)
            return future.result()

    def synthesize_to_buffer(self, text: str) -> io.BytesIO:
        payload = self.synthesize(text)
        buffer = io.BytesIO(payload)
        buffer.seek(0)
        return buffer
