from __future__ import annotations

import io
import wave

import numpy as np


def _read_wav(data: bytes) -> tuple[np.ndarray, int, int]:
    with wave.open(io.BytesIO(data), "rb") as reader:
        sample_rate = reader.getframerate()
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        frames = reader.readframes(reader.getnframes())

    if sample_width != 2:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels)
    return audio, sample_rate, channels


def _write_wav(audio: np.ndarray, sample_rate: int, channels: int) -> bytes:
    if channels > 1 and audio.ndim == 1:
        audio = np.repeat(audio[:, None], channels, axis=1)

    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())
    return output.getvalue()


def _resample_mono(mono: np.ndarray, new_length: int) -> np.ndarray:
    if new_length <= 1 or len(mono) <= 1:
        return mono
    x_old = np.linspace(0.0, 1.0, len(mono))
    x_new = np.linspace(0.0, 1.0, new_length)
    return np.interp(x_new, x_old, mono).astype(np.float32)


def _pitch_shift_mono(mono: np.ndarray, semitones: float) -> np.ndarray:
    if abs(semitones) < 0.05:
        return mono
    factor = 2.0 ** (semitones / 12.0)
    pitched = _resample_mono(mono, max(2, int(len(mono) / factor)))
    return _resample_mono(pitched, len(mono))


def _time_stretch_mono(mono: np.ndarray, speed: float) -> np.ndarray:
    if abs(speed - 1.0) < 0.02:
        return mono
    new_length = max(2, int(len(mono) / speed))
    stretched = _resample_mono(mono, new_length)
    return _resample_mono(stretched, len(mono))


def apply_wav_fx(
    data: bytes,
    *,
    pitch_semitones: float = 0.0,
    gain_db: float = 0.0,
    speed: float = 1.0,
) -> bytes:
    if abs(pitch_semitones) < 0.05 and abs(gain_db) < 0.05 and abs(speed - 1.0) < 0.02:
        return data

    audio, sample_rate, channels = _read_wav(data)
    if channels == 1:
        mono = audio
    else:
        mono = audio.mean(axis=1)

    mono = _time_stretch_mono(mono, speed)
    mono = _pitch_shift_mono(mono, pitch_semitones)

    if abs(gain_db) >= 0.05:
        mono *= 10.0 ** (gain_db / 20.0)

    if channels == 1:
        processed = mono
    else:
        processed = np.repeat(mono[:, None], channels, axis=1)

    return _write_wav(processed, sample_rate, channels)
