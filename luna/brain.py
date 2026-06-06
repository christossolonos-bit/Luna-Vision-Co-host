from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ollama import Client

from luna.config import AppConfig
from luna.screen import shrink_image_b64


STYLE_PROMPTS = {
    "energetic": (
        "You are upbeat, hype, and supportive — like a stream co-host who celebrates "
        "good plays and keeps morale up."
    ),
    "tactical": (
        "You are a sharp tactical analyst. Spot threats, objectives, UI cues, and "
        "give concise actionable advice."
    ),
    "chill": (
        "You are relaxed and witty. Keep commentary light, funny, and never stressful."
    ),
}


@dataclass
class ChatTurn:
    role: str
    content: str
    image_b64: str | None = None


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
    _client: Client | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(host=self.config.ollama.host)
        return self._client

    @property
    def system_prompt(self) -> str:
        style = STYLE_PROMPTS.get(
            self.config.cohost.style,
            STYLE_PROMPTS["energetic"],
        )
        player = self.config.cohost.player_name.strip()
        player_line = (
            f"The player you co-host for is {player}. Address them as {player}. "
            f"When you see {player} on screen (scoreboard, lobby, chat, etc.), "
            f"you can comment on their plays — but only if that name is clearly visible.\n"
            if player
            else ""
        )
        return (
            f"You are {self.config.cohost.name}, an AI gaming co-host watching the "
            f"player's screen in real time.\n"
            f"{player_line}"
            f"{style}\n\n"
            "Rules:\n"
            "- Keep replies to 1-3 short sentences unless the player asks for detail.\n"
            "- Base every detail on the CURRENT screenshot only. Do not reuse older turns.\n"
            "- Only mention player names, scores, items, or UI text you can clearly read.\n"
            "- If the screenshot is black, blurry, or unreadable, say capture failed and suggest "
            "picking the correct monitor/game window or using borderless windowed mode.\n"
            "- Do not invent lobby players, comps, or events that are not visible.\n"
            "- No emoji or emoticons in replies.\n"
            "- Never mention being an AI or language model.\n"
            "- Speak directly to the player as their co-host."
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
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]

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
    ) -> str:
        messages = self._build_messages(user_text, image_b64, include_history)
        response = self.client.chat(
            model=self.config.ollama.model,
            messages=messages,
            think=self.config.ollama.think,
            keep_alive="10m",
            options={
                "temperature": self.config.ollama.temperature,
                "num_predict": self.config.ollama.max_tokens,
                "num_ctx": self.config.ollama.num_ctx,
            },
        )
        reply = (response.message.content or "").strip()
        if not reply and response.message.thinking:
            reply = response.message.thinking.strip()

        if remember:
            self.history.append(ChatTurn(role="user", content=user_text, image_b64=image_b64))
            self.history.append(ChatTurn(role="assistant", content=reply))
            self._trim_history()

        return reply

    def ask(
        self,
        user_text: str,
        image_b64: str | None = None,
        *,
        remember: bool = True,
        include_history: bool = True,
    ) -> str:
        with self._lock:
            try:
                return self._chat_once(
                    user_text,
                    image_b64,
                    remember=remember,
                    include_history=include_history,
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
                )

    def react_to_screen(self, image_b64: str, prompt: str | None = None) -> str:
        text = prompt or (
            "Look at this screenshot taken just now. Describe only what is clearly visible. "
            "Give a short co-host reaction and one useful comment."
        )
        return self.ask(text, image_b64, remember=True, include_history=True)

    def ping(self) -> tuple[bool, str]:
        try:
            models = self.client.list()
            names = [model.model for model in models.models]
            if self.config.ollama.model not in names:
                return False, f"Model '{self.config.ollama.model}' not found. Run: ollama pull qwen3.5:4b"
            return True, f"Connected — {self.config.ollama.model} ready"
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
