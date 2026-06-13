from __future__ import annotations

from luna.config import TwitchConfig

DEFAULT_VOICE_RULES = (
    "Voice rules: plain text for TTS. No markdown. Avoid filler like 'pack', "
    "'stream', or tail-wag spam unless it fits naturally once."
)


def _split_aliases(raw: str) -> list[str]:
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def build_twitch_system_prompt(config: TwitchConfig) -> str:
    parts: list[str] = []

    if config.system_prompt.strip():
        parts.append(config.system_prompt.strip())
    else:
        parts.append(
            "You are Luna, a fun co-host on Twitch chat. Reply naturally in plain text for TTS."
        )

    if config.persona.strip():
        parts.append(config.persona.strip())

    voice_rules = config.voice_rules.strip() or DEFAULT_VOICE_RULES
    parts.append(voice_rules)

    if config.creator_name or config.owner_login:
        creator = config.creator_name or config.owner_login
        aliases = _split_aliases(config.creator_aliases)
        alias_text = ", ".join(aliases) if aliases else config.owner_login
        parts.append(
            f"The stream creator is {creator} (Twitch login: {config.owner_login}). "
            f"Also known as: {alias_text}. Be warm with them; playful with chat."
        )

    parts.append(
        "You are reading live Twitch chat with an optional screen capture. "
        "Answer the person's message directly — do not describe the whole screen. "
        "Only mention on-screen details that support your reply. "
        "Keep replies concise unless someone asks for more."
    )
    return "\n\n".join(parts)


def is_creator_login(username: str, config: TwitchConfig) -> bool:
    login = username.strip().lower()
    if not login:
        return False
    if config.owner_login and login == config.owner_login.strip().lower():
        return True
    return login in _split_aliases(config.creator_aliases)
