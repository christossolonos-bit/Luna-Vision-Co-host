from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen3.5:4b"
    think: bool = False
    temperature: float = 0.85
    max_tokens: int = 180
    num_ctx: int = 4096


@dataclass
class ScreenConfig:
    monitor: int = 1
    max_width: int = 960
    jpeg_quality: int = 68
    capture_interval_sec: float = 8.0


@dataclass
class CohostConfig:
    name: str = "Luna"
    player_name: str = "solonaras"
    style: str = "energetic"
    max_history: int = 12


@dataclass
class UIConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False


@dataclass
class VRMConfig:
    model_path: str = r"D:\Luna Singing\Luna.vrm"
    idle_animation_path: str = r"D:\Luna Singing\standing2.vrma"


@dataclass
class VoiceConfig:
    edge_voice: str = "en-US-AvaMultilingualNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"


@dataclass
class SpeechConfig:
    model: str = "base"
    device: str = "auto"
    compute_type: str = "int8"


@dataclass
class AppConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    cohost: CohostConfig = field(default_factory=CohostConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    vrm: VRMConfig = field(default_factory=VRMConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)


def load_config(path: Path | str | None = None) -> AppConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return AppConfig()

    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    return AppConfig(
        ollama=OllamaConfig(**raw.get("ollama", {})),
        screen=ScreenConfig(**raw.get("screen", {})),
        cohost=CohostConfig(**raw.get("cohost", {})),
        ui=UIConfig(**raw.get("ui", {})),
        vrm=VRMConfig(**raw.get("vrm", {})),
        voice=VoiceConfig(**raw.get("voice", {})),
        speech=SpeechConfig(**raw.get("speech", {})),
    )
