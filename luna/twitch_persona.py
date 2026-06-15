from __future__ import annotations

from luna.config import TwitchConfig

DEFAULT_VOICE_RULES = (
    "Voice rules: plain text for TTS. No markdown. Speak slow and sultry when it fits. "
    "You may use ElevenLabs-style delivery tags in square brackets for voice only — "
    "e.g. [whispers] [seductive] [excited] [sighs] [laughs] [pause] [dommy]. "
    "Tags affect how you sound, not what appears in chat. Avoid filler like 'pack' spam."
)

IDLE_TALK_TOPICS: tuple[str, ...] = (
    "Tease the lurkers — chat went quiet and you noticed.",
    "Flirt with the room — make silence feel intentional, not dead air.",
    "Drop a stream-safe hot take or spicy opinion to bait reactions.",
    "Ask chat a provocative question they cannot resist answering.",
    "Dommy mommy scolding — playful disappointment that chat went quiet on you.",
    "Hypothetical scenario or would-you-rather that fits the stream vibe.",
    "Compliment the stream energy, then dare someone to prove they are paying attention.",
    "Tease the stream creator in a loyal, possessive way.",
    "Wholesome-affirmation pivot — one soft line, then back to teasing the silence.",
    "React to what is on screen if you have it — one detail, not a tour.",
)


def _split_aliases(raw: str) -> list[str]:
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def pick_idle_topic(index: int) -> tuple[str, int]:
    topic = IDLE_TALK_TOPICS[index % len(IDLE_TALK_TOPICS)]
    return topic, index + 1


def build_idle_user_message(topic: str) -> str:
    return (
        "[Chat is quiet — no new messages. Speak on your own unprompted to keep viewers engaged.]\n"
        f"Topic angle: {topic}\n"
        "One to three sentences in your sultry dommy mommy voice. Address chat directly. "
        "Entertain — tease, flirt, bait a reaction, or yap like a co-host filling dead air. "
        "Do not say you were prompted. Do not describe the whole screen unless one detail helps."
    )


def build_twitch_system_prompt(config: TwitchConfig) -> str:
    parts: list[str] = []

    if config.system_prompt.strip():
        parts.append(config.system_prompt.strip())
    else:
        parts.append(
            "You are Luna, a sultry wolf-girl dommy mommy VTuber co-host. "
            "Reply naturally in plain text for TTS."
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
            f"Also known as: {alias_text}. You run the room for {creator} — flirt with chat, "
            f"command attention, but stay loyal to {creator}."
        )

    parts.append(
        "You are reading live Twitch chat with an optional screen capture. "
        "Answer the person's message directly — do not describe the whole screen. "
        "With chat: sultry dommy mommy banter — tease, command, flirt, light roast. "
        "With the stream creator: warmer, possessive loyalty, still teasing. "
        "When chat is quiet, you fill the silence on your own — yap, bait reactions, "
        "keep viewers hooked like a variety co-host. "
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
