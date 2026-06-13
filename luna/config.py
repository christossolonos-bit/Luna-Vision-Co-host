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
    video_max_segments: int = 5
    video_bytes_per_second: int = 1_000_000
    video_fps: int = 4
    video_segment_seconds: float = 1.0


@dataclass
class CohostConfig:
    name: str = "Luna"
    player_name: str = "solonaras"
    player_champion: str = ""
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
class LeagueConfig:
    enabled: bool = True
    live_port: int = 2999
    timeout_sec: float = 1.5
    lockfile_paths: list[str] = field(default_factory=list)


@dataclass
class TwitchConfig:
    enabled: bool = False
    username: str = ""
    oauth_token: str = ""
    channel: str = ""
    client_id: str = ""
    client_secret: str = ""
    creator_name: str = ""
    owner_login: str = ""
    creator_aliases: str = ""
    system_prompt: str = ""
    persona: str = ""
    voice_rules: str = ""
    send_replies: bool = True
    auto_reply: bool = True
    auto_trigger: str = "all"
    auto_cooldown_sec: float = 6.0
    speak_replies: bool = True
    command_prefix: str = "!luna"
    use_screen: bool = True


@dataclass
class AppConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    cohost: CohostConfig = field(default_factory=CohostConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)
    twitch: TwitchConfig = field(default_factory=TwitchConfig)
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
        league=LeagueConfig(**raw.get("league", {})),
        twitch=_load_twitch_config(raw.get("twitch", {})),
        ui=UIConfig(**raw.get("ui", {})),
        vrm=VRMConfig(**raw.get("vrm", {})),
        voice=VoiceConfig(**raw.get("voice", {})),
        speech=SpeechConfig(**raw.get("speech", {})),
    )


def _load_twitch_config(raw: dict[str, Any]) -> TwitchConfig:
    if not raw:
        return TwitchConfig()

    aliases = {
        "TWITCH_OAUTH_TOKEN": "oauth_token",
        "twitch_client_id": "client_id",
        "twitch_client_secret": "client_secret",
        "twitch_channel": "channel",
        "twitch_username": "username",
        "TWITCH_SYSTEM": "system_prompt",
        "LUNA_PERSONA": "persona",
        "LUNA_VOICE_RULES": "voice_rules",
        "LUNA_CREATOR_NAME": "creator_name",
        "LUNA_OWNER_TWITCH_LOGIN": "owner_login",
        "LUNA_CREATOR_ALIASES": "creator_aliases",
        "TWITCH_SEND_REPLIES": "send_replies",
        "TWITCH_AUTO_REPLY": "auto_reply",
        "TWITCH_AUTO_TRIGGER": "auto_trigger",
        "TWITCH_AUTO_COOLDOWN": "auto_cooldown_sec",
    }
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        target = aliases.get(key, key)
        normalized[target] = value

    bool_keys = ("enabled", "send_replies", "auto_reply", "speak_replies")
    for key in bool_keys:
        if key in normalized and isinstance(normalized[key], str):
            normalized[key] = normalized[key].strip().lower() in {"1", "true", "yes", "on"}

    if "auto_cooldown_sec" in normalized:
        try:
            normalized["auto_cooldown_sec"] = float(normalized["auto_cooldown_sec"])
        except (TypeError, ValueError):
            normalized.pop("auto_cooldown_sec", None)

    field_names = {field.name for field in TwitchConfig.__dataclass_fields__.values()}
    filtered = {key: value for key, value in normalized.items() if key in field_names}
    return TwitchConfig(**filtered)
