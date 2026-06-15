from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING, Callable

from luna.config import TwitchConfig
from luna.voice_tags import strip_tags_for_display

if TYPE_CHECKING:
    from luna.server import LunaService, ObsRelay

logger = logging.getLogger(__name__)

IRC_HOST = "irc.chat.twitch.tv"
IRC_PORT = 6667
PRIVMSG_RE = re.compile(
    r"(?:@(?P<tags>[^ ]+) )?"
    r":(?P<user>[^!]+)![^ ]+ PRIVMSG #(?P<channel>[^ ]+) :(?P<text>.+)"
)


class TwitchChatBot:
    def __init__(
        self,
        config: TwitchConfig,
        service: LunaService,
        obs_relay: ObsRelay,
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.service = service
        self.obs_relay = obs_relay
        self._on_status = on_status
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._last_reply_at = 0.0
        self._last_channel_message_at = 0.0
        self._last_idle_talk_at = 0.0
        self._connected = False
        self._busy = False
        self._idle_task: asyncio.Task | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    def status_line(self) -> str:
        if not self.config.enabled:
            return "Twitch: disabled"
        if not self.config.oauth_token or not self.config.channel:
            return "Twitch: not configured"
        if self._connected:
            return f"Twitch: connected #{self.config.channel.lower()}"
        return "Twitch: connecting…"

    async def start(self) -> None:
        if not self.config.enabled:
            return
        if not self.config.oauth_token or not self.config.username or not self.config.channel:
            logger.warning("Twitch enabled but username, oauth_token, or channel is missing")
            self._set_status("Twitch: missing credentials")
            return
        self._stop.clear()
        self._last_channel_message_at = time.time()
        self._task = asyncio.create_task(self._run_loop(), name="twitch-chat")
        self._idle_task = asyncio.create_task(self._idle_loop(), name="twitch-idle")

    async def stop(self) -> None:
        self._stop.set()
        if self._idle_task is not None:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
            self._idle_task = None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._disconnect()

    def _set_status(self, message: str) -> None:
        if self._on_status:
            self._on_status(message)
        logger.info(message)

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._connect()
                await self._listen()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Twitch IRC error: %s", exc)
                self._connected = False
                self._set_status(f"Twitch: error — retrying ({exc})")
            finally:
                await self._disconnect()

            if self._stop.is_set():
                break
            await asyncio.sleep(5)

    async def _connect(self) -> None:
        self._reader, self._writer = await asyncio.open_connection(IRC_HOST, IRC_PORT)
        token = self.config.oauth_token.strip()
        if not token.startswith("oauth:"):
            token = f"oauth:{token}"

        await self._send(f"PASS {token}")
        await self._send(f"NICK {self.config.username.strip()}")
        await self._send("CAP REQ :twitch.tv/tags twitch.tv/commands")
        channel = self.config.channel.strip().lstrip("#").lower()
        await self._send(f"JOIN #{channel}")
        self._connected = True
        self._set_status(f"Twitch: connected #{channel}")

    async def _disconnect(self) -> None:
        self._connected = False
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass
        self._writer = None
        self._reader = None

    async def _send(self, line: str) -> None:
        if self._writer is None:
            return
        self._writer.write(f"{line}\r\n".encode("utf-8"))
        await self._writer.drain()

    async def _send_chat(self, message: str) -> None:
        channel = self.config.channel.strip().lstrip("#").lower()
        text = message.replace("\r", " ").replace("\n", " ").strip()
        if not text:
            return
        if len(text) > 480:
            text = text[:477] + "..."
        await self._send(f"PRIVMSG #{channel} :{text}")

    async def _listen(self) -> None:
        assert self._reader is not None
        while not self._stop.is_set():
            line = await self._reader.readline()
            if not line:
                raise ConnectionError("Twitch IRC disconnected")
            decoded = line.decode("utf-8", errors="ignore").strip()
            if not decoded:
                continue
            if decoded.startswith("PING"):
                await self._send("PONG :tmi.twitch.tv")
                continue
            await self._handle_line(decoded)

    async def _handle_line(self, line: str) -> None:
        match = PRIVMSG_RE.match(line)
        if not match:
            return

        user = match.group("user")
        text = match.group("text").strip()
        if not text or user.lower() == self.config.username.strip().lower():
            return
        self._last_channel_message_at = time.time()
        if not self._should_reply(user, text):
            return

        now = time.time()
        if now - self._last_reply_at < self.config.auto_cooldown_sec:
            return

        self._last_reply_at = now
        asyncio.create_task(self._handle_message(user, text))

    def _should_reply(self, user: str, text: str) -> bool:
        if not self.config.auto_reply:
            return False
        trigger = self.config.auto_trigger.strip().lower()
        lowered = text.lower()
        bot_name = self.config.username.strip().lower()

        if trigger == "command":
            prefix = self.config.command_prefix.strip().lower()
            return lowered.startswith(prefix)
        if trigger == "mention":
            return f"@{bot_name}" in lowered or bot_name in lowered
        return True

    async def _handle_message(self, user: str, text: str) -> None:
        self._busy = True
        try:
            reply = await asyncio.to_thread(self.service.twitch_chat_reply, user, text)
            if not reply:
                return

            if self.config.send_replies:
                try:
                    await self._send_chat(
                        strip_tags_for_display(
                            reply,
                            enabled=self.service.config.voice.use_delivery_tags,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Twitch send failed: %s", exc)

            await self._speak_reply(reply)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Twitch reply failed: %s", exc)
        finally:
            self._busy = False

    async def _idle_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.sleep(5)
                if not self.config.idle_talk or not self._connected or self._busy:
                    continue

                now = time.time()
                quiet_for = now - self._last_channel_message_at
                if quiet_for < self.config.idle_talk_quiet_sec:
                    continue
                if now - self._last_idle_talk_at < self.config.idle_talk_cooldown_sec:
                    continue

                self._busy = True
                try:
                    reply = await asyncio.to_thread(self.service.twitch_idle_talk)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Twitch idle talk failed: %s", exc)
                    continue
                finally:
                    self._busy = False

                if not reply:
                    continue

                self._last_idle_talk_at = time.time()
                if self.config.idle_talk_send_chat and self.config.send_replies:
                    try:
                        await self._send_chat(
                            strip_tags_for_display(
                                reply,
                                enabled=self.service.config.voice.use_delivery_tags,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Twitch idle send failed: %s", exc)

                await self._speak_reply(reply)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Twitch idle loop error: %s", exc)

    async def _speak_reply(self, reply: str) -> None:
        if not self.config.speak_replies:
            return

        try:
            audio_b64, _ = await asyncio.to_thread(self.service.speak_text_for_obs, reply)
            if audio_b64 and self.obs_relay.is_active():
                self.obs_relay.queue_tts(audio_b64)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Twitch TTS failed: %s", exc)
