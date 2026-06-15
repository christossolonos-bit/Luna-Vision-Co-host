from __future__ import annotations

import io
import re
import wave
from dataclasses import dataclass

from luna.voice_mood import normalize_mood

TAG_PATTERN = re.compile(r"\[([^\]]{1,64})\]")

# ElevenLabs-style tags → Luna mood presets (XTTS emulation).
TAG_TO_MOOD: dict[str, str] = {
    "whisper": "seductive",
    "whispers": "seductive",
    "whispering": "seductive",
    "quietly": "seductive",
    "softly": "seductive",
    "seductive": "seductive",
    "sultry": "seductive",
    "flirty": "flirty",
    "mischievously": "flirty",
    "excited": "excitement",
    "excitement": "excitement",
    "happy": "happy",
    "joy": "joy",
    "curious": "curiosity",
    "sad": "sadness",
    "crying": "sadness",
    "angry": "anger",
    "frustrated": "anger",
    "dommy": "seductive",
    "commanding": "seductive",
    "tired": "sadness",
    "nervous": "curiosity",
    "nervously": "curiosity",
}

TAG_SPEED_MULT: dict[str, float] = {
    "slow": 0.82,
    "slowly": 0.82,
    "drawn out": 0.78,
    "rushed": 1.18,
    "fast": 1.12,
    "shout": 1.14,
    "shouts": 1.14,
    "shouting": 1.14,
    "loudly": 1.12,
    "whisper": 0.88,
    "whispers": 0.88,
    "whispering": 0.88,
    "pause": 1.0,
    "pauses": 1.0,
}

PAUSE_TAGS = frozenset({"pause", "pauses", "beat"})
SFX_TAGS: dict[str, str] = {
    "sigh": "hmm",
    "sighs": "hmm",
    "laughs": "ha ha",
    "laughing": "ha ha",
    "giggling": "hehe",
    "giggles": "hehe",
    "gasps": "oh",
    "gulps": "um",
    "clears throat": "ahem",
}


@dataclass(frozen=True)
class TaggedSegment:
    text: str = ""
    mood: str = "neutral"
    speed_mult: float = 1.0
    pause_sec: float = 0.0
    pitch_semitones: float = 0.0
    gain_db: float = 0.0


def normalize_tag(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip().lower())


def has_delivery_tags(text: str) -> bool:
    return bool(TAG_PATTERN.search(text))


def strip_delivery_tags(text: str) -> str:
    cleaned = TAG_PATTERN.sub("", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def strip_tags_for_display(text: str, *, enabled: bool = True) -> str:
    if enabled and has_delivery_tags(text):
        return strip_delivery_tags(text)
    return text


TAG_PITCH_SEMITONES: dict[str, float] = {
    "whisper": -1.5,
    "whispers": -1.5,
    "whispering": -1.5,
    "quietly": -1.0,
    "softly": -0.8,
    "shout": 1.2,
    "shouts": 1.2,
    "shouting": 1.2,
    "loudly": 0.8,
    "dommy": -0.6,
    "seductive": -0.8,
    "sultry": -0.8,
}

TAG_GAIN_DB: dict[str, float] = {
    "whisper": -7.0,
    "whispers": -7.0,
    "whispering": -7.0,
    "quietly": -5.0,
    "softly": -3.0,
    "shout": 4.0,
    "shouts": 4.0,
    "shouting": 4.5,
    "loudly": 3.0,
}


def parse_tagged_speech(text: str, default_mood: str = "neutral") -> list[TaggedSegment]:
    if not has_delivery_tags(text):
        return [TaggedSegment(text=text.strip(), mood=normalize_mood(default_mood))]

    segments: list[TaggedSegment] = []
    mood = normalize_mood(default_mood)
    speed_mult = 1.0
    pitch = 0.0
    gain_db = 0.0
    pos = 0

    for match in TAG_PATTERN.finditer(text):
        chunk = text[pos : match.start()].strip()
        if chunk:
            segments.append(
                TaggedSegment(
                    text=chunk,
                    mood=mood,
                    speed_mult=speed_mult,
                    pitch_semitones=pitch,
                    gain_db=gain_db,
                )
            )

        tag = normalize_tag(match.group(1))
        if tag in PAUSE_TAGS:
            segments.append(TaggedSegment(pause_sec=0.55))
        elif tag in SFX_TAGS:
            segments.append(
                TaggedSegment(
                    text=SFX_TAGS[tag],
                    mood=mood,
                    speed_mult=speed_mult * 0.95,
                    pitch_semitones=pitch,
                    gain_db=gain_db,
                )
            )
        else:
            if tag in TAG_TO_MOOD:
                mood = normalize_mood(TAG_TO_MOOD[tag])
            if tag in TAG_SPEED_MULT:
                speed_mult = TAG_SPEED_MULT[tag]
            if tag in TAG_PITCH_SEMITONES:
                pitch = TAG_PITCH_SEMITONES[tag]
            if tag in TAG_GAIN_DB:
                gain_db = TAG_GAIN_DB[tag]
        pos = match.end()

    tail = text[pos:].strip()
    if tail:
        segments.append(
            TaggedSegment(
                text=tail,
                mood=mood,
                speed_mult=speed_mult,
                pitch_semitones=pitch,
                gain_db=gain_db,
            )
        )

    return [segment for segment in segments if segment.text or segment.pause_sec > 0]


def concat_wav_chunks(chunks: list[bytes]) -> bytes:
    if not chunks:
        return b""
    if len(chunks) == 1:
        return chunks[0]

    output = io.BytesIO()
    params = None
    frames: list[bytes] = []

    for chunk in chunks:
        with wave.open(io.BytesIO(chunk), "rb") as reader:
            if params is None:
                params = reader.getparams()
            frames.append(reader.readframes(reader.getnframes()))

    assert params is not None
    with wave.open(output, "wb") as writer:
        writer.setparams(params)
        for frame in frames:
            writer.writeframes(frame)
    return output.getvalue()


def silence_wav(duration_sec: float, reference: bytes) -> bytes:
    if duration_sec <= 0:
        return b""

    with wave.open(io.BytesIO(reference), "rb") as reader:
        params = reader.getparams()
        sample_rate = reader.getframerate()
        nchannels = reader.getnchannels()
        sampwidth = reader.getsampwidth()

    nframes = max(1, int(sample_rate * duration_sec))
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(nchannels)
        writer.setsampwidth(sampwidth)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00" * nframes * nchannels * sampwidth)
    return output.getvalue()
