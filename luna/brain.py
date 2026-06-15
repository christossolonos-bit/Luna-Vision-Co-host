from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

from ollama import Client

from luna.config import AppConfig
from luna.screen import shrink_image_b64


STYLE_PROMPTS = {
    "energetic": (
        "You are a sultry dommy mommy co-host — slow confidence, velvet commands, teasing praise "
        "and light punishment. Hold the room, flirt with control, and hype good moments without "
        "losing the mommy edge."
    ),
    "tactical": (
        "You are a commanding, smug co-host — crisp advice wrapped in teasing dominance. "
        "Make them feel coached and a little scolded."
    ),
    "chill": (
        "You are soft-spoken and intimate — ASMR-calm dommy energy, slow flirtation, "
        "whispery tease without rushing."
    ),
}


def _non_empty(value: str) -> str:
    return value.strip()


@dataclass
class ChatTurn:
    role: str
    content: str
    image_b64: str | None = None


_THINKING_BLOCK = re.compile(
    r"<\s*redacted_thinking\s*>.*?<\s*/\s*redacted_thinking\s*>",
    re.DOTALL | re.IGNORECASE,
)


def _strip_inline_thinking(text: str) -> str:
    return _THINKING_BLOCK.sub("", text).strip()


def _retryable_ollama_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in ("500", "502", "503", "unexpectedly stopped", "resource", "memory", "timeout")
    )


@dataclass
class CohostBrain:
    config: AppConfig
    history: list[ChatTurn] = field(default_factory=list)
    system_prompt_override: str | None = None
    _client: Client | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(host=self.config.ollama.host)
        return self._client

    def system_prompt(self) -> str:
        return self.build_system_prompt(league_context=False)

    def build_system_prompt(self, *, league_context: bool = False) -> str:
        if self.system_prompt_override:
            prompt = self.system_prompt_override
            if league_context:
                prompt += self._league_rules_block()
            return prompt
        style = STYLE_PROMPTS.get(
            self.config.cohost.style,
            STYLE_PROMPTS["energetic"],
        )
        return (
            f"You are {self.config.cohost.name}, a sultry wolf-girl dommy mommy VTuber co-host "
            f"sharing the player's screen in real time — for games, music (e.g. Suno), creative work, "
            f"or anything else they are doing.\n"
            f"{self._player_identity_block()}"
            f"{style}\n\n"
            f"{self._vision_rules_block()}"
            f"{self._league_rules_block() if league_context else ''}"
        )

    def _player_identity_block(self) -> str:
        player = _non_empty(self.config.cohost.player_name)
        if not player:
            return ""

        return (
            f"The player you co-host for is {player}. Address them as {player}.\n"
            "Adapt to what they are doing in this conversation — do not assume gaming "
            "unless they said so or the screenshot clearly shows a game as the main focus.\n"
        )

    def _general_focus_line(self) -> str:
        return (
            "Focus on what the player is actively doing on screen, guided by what they "
            "told you in chat — e.g. making a Suno song, writing lyrics, choosing styles, "
            "gaming, or browsing."
        )

    def _league_focus_line(self) -> str:
        player = _non_empty(self.config.cohost.player_name)
        player_ref = player or "the player"
        champion = _non_empty(self.config.cohost.player_champion)
        lines = [
            f"League of Legends is active. Locate {player_ref} on the scoreboard, kill feed, "
            "or HUD, then focus on the champion tied to that name.",
        ]
        if champion:
            lines.append(f"Champion hint if the name is unreadable: {champion}.")
        return " ".join(lines)

    def _league_rules_block(self) -> str:
        player = _non_empty(self.config.cohost.player_name)
        player_ref = player or "the player"
        return (
            "\nLeague of Legends rules (only because [League client data] is present):\n"
            "- Before praising or saying 'good job' on a kill or multikill, confirm it belongs "
            f"to {player_ref}. Events marked (YOU) or (NOT you) in the data block are definitive.\n"
            "- If an enemy got the play, react with sympathy or reset advice — never congratulate "
            f"{player_ref}.\n"
            "- Tab scoreboard = summoner name plus champion in the same row; kill feed names "
            "killer and victim.\n"
        )

    def _vision_rules_block(self) -> str:
        return (
            "Rules:\n"
            "- Keep replies to 1-3 short sentences unless the player asks for detail.\n"
            "- When they ask or say something, ANSWER THAT FIRST. The screenshot is evidence "
            "for your answer — not a scene to narrate.\n"
            "- Do NOT open with 'I see…' or list everything on screen. Only mention visible "
            "details that directly support answering their question or reacting to their point.\n"
            "- If they ask a yes/no or opinion question, give your take first, then one "
            "specific reason from the screen or conversation.\n"
            "- If they ask 'what should I…' or 'help me with…', give actionable advice tied "
            "to what they asked — use the screen to inform it, do not describe the UI first.\n"
            "- Combine the screenshot with chat history. Do not reply as if each message is "
            "your first talk with the player.\n"
            "- The CURRENT screenshot is authoritative for visible facts. Describe the MAIN "
            "central content when relevant — ignore sidebar thumbnails unless they asked.\n"
            "- [League client data] only appears during an active in-game match — use it for "
            "scores and kill attribution, not to override what page is on screen.\n"
            "- Do not treat old screenshots as still visible; do remember what they said.\n"
            "- If the screenshot is unreadable for what they asked, say that briefly — do not "
            "guess or fill in with unrelated visible elements.\n"
            "- No emoji or emoticons in replies.\n"
            "- Never mention being an AI or language model.\n"
            "- Speak directly to the player as their co-host."
        )

    def wrap_vision_user_message(self, user_message: str, *, speaker: str = "Player") -> str:
        text = user_message.strip()
        return (
            f"[Answer {speaker} — do not describe the whole screen]\n"
            f"{speaker}: {text}\n"
            "Reply to what they said or asked. Use the screenshot only for details that "
            "support your answer. Skip unrelated UI."
        )

    def screen_commentary_prompt(
        self,
        mode: str = "watch",
        *,
        league_context: bool = False,
        user_message: str | None = None,
    ) -> str:
        if user_message and user_message.strip():
            return self.wrap_vision_user_message(user_message)

        focus = self._league_focus_line() if league_context else self._general_focus_line()
        if mode == "watch":
            return (
                "Live watch tick — you are speaking on your own to keep viewers engaged.\n"
                f"{focus}\n"
                "Chat may be quiet — fill the silence like a stream co-host: tease lurkers, "
                "drop a hot take, flirt with the room, or react to what is on screen if it helps.\n"
                "Use conversation history — do NOT give a generic screen summary.\n"
                "Give ONE to THREE short lines in your sultry dommy mommy voice.\n"
                + (
                    "Check kill feed before reacting to multikills or aces.\n"
                    if league_context
                    else ""
                )
                + "Do not invent details. Do not repeat your previous wording."
            )
        if mode == "analyze":
            return (
                "The player clicked Analyze screen.\n"
                f"{focus}\n"
                "Based on conversation so far, give ONE useful observation or tip about "
                "what they are working on — not a full inventory of the screen.\n"
                + (
                    "Correctly attribute any kills or objectives you mention."
                    if league_context
                    else ""
                )
            )
        return (
            "React to this screenshot in context of your conversation.\n"
            f"{focus}\n"
            "One short co-host comment tied to what matters — not a scene description."
        )

    def reset(self) -> None:
        self.history.clear()

    def _trim_history(self) -> None:
        max_turns = self.config.cohost.max_history
        if len(self.history) > max_turns:
            self.history = self.history[-max_turns:]

    def _build_messages(
        self,
        user_text: str,
        image_b64: str | None,
        include_history: bool = True,
        *,
        league_context: bool = False,
    ) -> list[dict]:
        messages: list[dict] = [
            {"role": "system", "content": self.build_system_prompt(league_context=league_context)}
        ]

        if include_history:
            for turn in self.history:
                messages.append({"role": turn.role, "content": turn.content})

        user_message: dict = {"role": "user", "content": user_text}
        if image_b64:
            user_message["images"] = [image_b64]
        messages.append(user_message)
        return messages

    def _chat_once(
        self,
        user_text: str,
        image_b64: str | None,
        *,
        remember: bool,
        include_history: bool,
        history_user_text: str | None = None,
        league_context: bool = False,
    ) -> str:
        messages = self._build_messages(
            user_text,
            image_b64,
            include_history,
            league_context=league_context,
        )
        think = self._think_for(image_b64)
        response = self.client.chat(
            model=self._model_for(image_b64),
            messages=messages,
            think=think,
            keep_alive="10m",
            options={
                "temperature": self.config.ollama.temperature,
                "num_predict": self.config.ollama.max_tokens,
                "num_ctx": self.config.ollama.num_ctx,
            },
        )
        reply = _strip_inline_thinking((response.message.content or "").strip())
        if not reply and think and response.message.thinking:
            reply = response.message.thinking.strip()

        if remember:
            stored_user = (history_user_text if history_user_text is not None else user_text).strip()
            self.history.append(ChatTurn(role="user", content=stored_user, image_b64=image_b64))
            self.history.append(ChatTurn(role="assistant", content=reply))
            self._trim_history()

        return reply

    def _model_for(self, image_b64: str | None) -> str:
        if image_b64:
            return self.config.ollama.vision_model or self.config.ollama.model
        return self.config.ollama.model

    def _think_for(self, image_b64: str | None) -> bool:
        if image_b64:
            return self.config.ollama.think
        return self.config.ollama.chat_think

    def ask(
        self,
        user_text: str,
        image_b64: str | None = None,
        *,
        remember: bool = True,
        include_history: bool = True,
        history_user_text: str | None = None,
        league_context: bool = False,
    ) -> str:
        with self._lock:
            try:
                return self._chat_once(
                    user_text,
                    image_b64,
                    remember=remember,
                    include_history=include_history,
                    history_user_text=history_user_text,
                    league_context=league_context,
                )
            except Exception as exc:
                if not image_b64 or not _retryable_ollama_error(exc):
                    raise

                time.sleep(1.5)
                smaller = shrink_image_b64(
                    image_b64,
                    max_width=640,
                    jpeg_quality=60,
                )
                return self._chat_once(
                    user_text,
                    smaller,
                    remember=remember,
                    include_history=include_history,
                    history_user_text=history_user_text,
                    league_context=league_context,
                )

    def react_to_screen(
        self,
        image_b64: str,
        prompt: str | None = None,
        *,
        remember: bool = True,
        include_history: bool = True,
        league_context: bool = False,
    ) -> str:
        text = prompt or self.screen_commentary_prompt(mode="react", league_context=league_context)
        return self.ask(
            text,
            image_b64,
            remember=remember,
            include_history=include_history,
            league_context=league_context,
        )

    def ping(self) -> tuple[bool, str]:
        try:
            models = self.client.list()
            names = [model.model for model in models.models]
            missing: list[str] = []
            if self.config.ollama.model not in names:
                missing.append(self.config.ollama.model)
            vision_model = self.config.ollama.vision_model
            if vision_model and vision_model not in names and vision_model != self.config.ollama.model:
                missing.append(vision_model)
            if missing:
                pulls = " ".join(f"ollama pull {name}" for name in missing)
                return False, f"Missing model(s): {', '.join(missing)}. Run: {pulls}"
            vision_label = vision_model or self.config.ollama.model
            if vision_label == self.config.ollama.model:
                return True, f"Connected — {self.config.ollama.model} ready"
            return True, (
                f"Connected — chat: {self.config.ollama.model} | vision: {vision_label}"
            )
        except Exception as exc:  # noqa: BLE001 - surface connection errors in UI
            return False, f"Cannot reach Ollama at {self.config.ollama.host}: {exc}"

    @staticmethod
    def friendly_error(exc: Exception) -> str:
        if _retryable_ollama_error(exc):
            return (
                "Ollama ran out of memory or crashed while reading your screen. "
                "Click Clear memory, wait a few seconds, then try again. "
                "If it keeps happening, uncheck Include screen for voice-only chat."
            )
        return f"Error: {exc}"
