from __future__ import annotations

import base64
import json
import os
import re
import ssl
import subprocess
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from luna.config import LeagueConfig


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _http_get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 1.5,
) -> Any:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, context=_ssl_context(), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _matches_player(name: str, player_ref: str) -> bool:
    if not name or not player_ref:
        return False
    left = name.strip().lower()
    right = player_ref.strip().lower()
    if not left or not right:
        return False
    return left == right or left.startswith(right) or right.startswith(left)


def _format_game_time(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def _player_label(player: dict[str, Any]) -> str:
    for key in ("riotIdGameName", "summonerName", "riotId"):
        value = player.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Unknown"


@dataclass
class LeagueSnapshot:
    active: bool = False
    phase: str = "offline"
    summary: str = "League client not detected"
    context_block: str = ""


@dataclass
class LeagueClient:
    config: LeagueConfig
    player_name: str
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _lcu_auth: tuple[int, str] | None = field(default=None, repr=False)
    _champion_names: dict[int, str] = field(default_factory=dict, repr=False)

    def snapshot(self) -> LeagueSnapshot:
        if not self.config.enabled:
            return LeagueSnapshot(
                active=False,
                phase="disabled",
                summary="League integration disabled",
            )

        with self._lock:
            live = self._live_snapshot()
            if live.active:
                return live
            return self._lcu_snapshot()

    def context_block(self) -> str:
        return self.snapshot().context_block

    def context_block_for_vision(self) -> str:
        """Only inject League data into vision prompts during an active match."""
        snap = self.snapshot()
        if snap.phase == "in_game":
            return snap.context_block
        return ""

    def is_in_game(self) -> bool:
        return self.snapshot().phase == "in_game"

    def status_line(self) -> str:
        snap = self.snapshot()
        if snap.phase == "disabled":
            return ""
        return snap.summary

    def _live_get(self, endpoint: str) -> Any | None:
        url = f"https://127.0.0.1:{self.config.live_port}/liveclientdata/{endpoint}"
        try:
            return _http_get_json(url, timeout=self.config.timeout_sec)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None

    def _find_lcu_auth(self) -> tuple[int, str] | None:
        if self._lcu_auth is not None:
            return self._lcu_auth

        for path in self._lockfile_paths():
            if not path.exists():
                continue
            try:
                parts = path.read_text(encoding="utf-8").strip().split(":")
                if len(parts) >= 4 and parts[2].isdigit():
                    self._lcu_auth = (int(parts[2]), parts[3])
                    return self._lcu_auth
            except OSError:
                continue

        auth = self._lcu_auth_from_process()
        if auth is not None:
            self._lcu_auth = auth
        return self._lcu_auth

    def _lockfile_paths(self) -> list[Path]:
        paths: list[Path] = []
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            paths.append(Path(local_app) / "Riot Games" / "League of Legends" / "lockfile")
        for extra in self.config.lockfile_paths:
            paths.append(Path(extra))
        return paths

    def _lcu_auth_from_process(self) -> tuple[int, str] | None:
        try:
            output = subprocess.check_output(
                [
                    "wmic",
                    "process",
                    "where",
                    "name='LeagueClientUx.exe'",
                    "get",
                    "CommandLine",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            return None

        port_match = re.search(r"--app-port=(\d+)", output)
        token_match = re.search(r"--remoting-auth-token=([\w-]+)", output)
        if not port_match or not token_match:
            return None
        return int(port_match.group(1)), token_match.group(1)

    def _lcu_get(self, path: str) -> Any | None:
        auth = self._find_lcu_auth()
        if auth is None:
            return None

        port, password = auth
        credentials = base64.b64encode(f"riot:{password}".encode()).decode()
        url = f"https://127.0.0.1:{port}{path}"
        try:
            return _http_get_json(
                url,
                headers={"Authorization": f"Basic {credentials}"},
                timeout=self.config.timeout_sec,
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None

    def _ensure_champion_names(self) -> None:
        if self._champion_names:
            return
        payload = self._lcu_get("/lol-game-data/assets/v1/champions.json")
        if not isinstance(payload, list):
            return
        for champion in payload:
            champion_id = champion.get("id")
            name = champion.get("name") or champion.get("alias")
            if isinstance(champion_id, int) and isinstance(name, str):
                self._champion_names[champion_id] = name

    def _champion_name(self, champion_id: int | None) -> str:
        if not champion_id or champion_id <= 0:
            return "unpicked"
        self._ensure_champion_names()
        if champion_id in self._champion_names:
            return self._champion_names[champion_id]
        live_players = self._live_get("playerlist")
        if isinstance(live_players, list):
            for player in live_players:
                if player.get("championId") == champion_id:
                    name = player.get("championName")
                    if isinstance(name, str) and name:
                        return name
        return f"Champion#{champion_id}"

    def _find_local_player(self, players: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not self.player_name:
            active = self._live_get("activeplayername")
            if isinstance(active, str):
                for player in players:
                    if _matches_player(_player_label(player), active):
                        return player
            return players[0] if len(players) == 1 else None

        for player in players:
            for key in ("riotIdGameName", "summonerName", "riotId"):
                value = player.get(key)
                if isinstance(value, str) and _matches_player(value, self.player_name):
                    return player
        return None

    def _live_snapshot(self) -> LeagueSnapshot:
        players = self._live_get("playerlist")
        if not isinstance(players, list) or not players:
            return LeagueSnapshot()

        local = self._find_local_player(players)
        game_stats = self._live_get("gamestats") or {}
        events = self._live_get("eventdata") or {}

        game_time = _format_game_time(float(game_stats.get("gameTime") or 0))
        mode = str(game_stats.get("gameMode") or "GAME").replace("_", " ")

        lines = [
            "[League client data — authoritative for champion, K/D/A, CS, and kill attribution]",
            f"In game — {mode} — {game_time}",
        ]

        if local:
            scores = local.get("scores") or {}
            champion = local.get("championName") or self._champion_name(local.get("championId"))
            k = scores.get("kills", 0)
            d = scores.get("deaths", 0)
            a = scores.get("assists", 0)
            cs = scores.get("creepScore", 0)
            level = local.get("level", "?")
            status = "dead" if local.get("isDead") else "alive"
            lines.append(
                f"You ({self.player_name or _player_label(local)}): {champion} — "
                f"{k}/{d}/{a}, {cs} CS, level {level}, {status}"
            )
        else:
            lines.append(
                f"Could not match summoner '{self.player_name}' in player list — "
                "verify player_name in config matches your in-game name."
            )

        event_lines = self._format_recent_events(events, local)
        if event_lines:
            lines.append("Recent events:")
            lines.extend(event_lines)

        team_lines = self._format_teams(players, local)
        if team_lines:
            lines.append("Scoreboard:")
            lines.extend(team_lines)

        lines.append(
            "Prefer this data over the screenshot for scores and who got kills. "
            "Use the screenshot for positioning and teamfight visuals."
        )

        summary = "League: in game"
        if local:
            champion = local.get("championName") or self._champion_name(local.get("championId"))
            summary = f"League: in game as {champion} ({game_time})"

        return LeagueSnapshot(
            active=True,
            phase="in_game",
            summary=summary,
            context_block="\n".join(lines),
        )

    def _format_recent_events(
        self,
        events_payload: dict[str, Any],
        local: dict[str, Any] | None,
    ) -> list[str]:
        raw_events = events_payload.get("Events")
        if not isinstance(raw_events, list):
            return []

        interesting = {
            "ChampKill",
            "Multikill",
            "Ace",
            "DragonKill",
            "BaronKill",
            "HeraldKill",
            "TurretKilled",
            "InhibKilled",
        }
        recent: list[str] = []
        for event in raw_events[-12:]:
            if not isinstance(event, dict):
                continue
            name = event.get("EventName")
            if name not in interesting:
                continue
            recent.append(self._format_event_line(event, local))

        return recent[-6:]

    def _format_event_line(self, event: dict[str, Any], local: dict[str, Any] | None) -> str:
        name = str(event.get("EventName") or "Event")
        killer = str(event.get("KillerName") or "")
        victim = str(event.get("VictimName") or "")
        streak = event.get("KillStreak")

        if name == "Multikill" and killer:
            label = {2: "Double kill", 3: "Triple kill", 4: "Quadra kill", 5: "PENTAKILL"}.get(
                int(streak or 0),
                f"{streak}-kill streak",
            )
            attribution = self._attribution_suffix(killer)
            return f"- {label} by {killer}{attribution}"

        if name == "ChampKill" and killer and victim:
            attribution = self._attribution_suffix(killer)
            return f"- {killer} killed {victim}{attribution}"

        if name == "Ace":
            return "- Team ace"

        if killer:
            return f"- {name}: {killer}{self._attribution_suffix(killer)}"
        return f"- {name}"

    def _attribution_suffix(self, actor_name: str) -> str:
        if self.player_name and _matches_player(actor_name, self.player_name):
            return " (YOU)"
        if self.player_name:
            return " (NOT you)"
        return ""

    def _format_teams(
        self,
        players: list[dict[str, Any]],
        local: dict[str, Any] | None,
    ) -> list[str]:
        if not local:
            team_key = players[0].get("team")
        else:
            team_key = local.get("team")

        allies: list[str] = []
        enemies: list[str] = []
        for player in players:
            scores = player.get("scores") or {}
            label = _player_label(player)
            champion = player.get("championName") or self._champion_name(player.get("championId"))
            k = scores.get("kills", 0)
            d = scores.get("deaths", 0)
            a = scores.get("assists", 0)
            row = f"{label} ({champion} {k}/{d}/{a})"
            if local and _matches_player(label, self.player_name):
                row = f"{row} ← you"
            if player.get("team") == team_key:
                allies.append(row)
            else:
                enemies.append(row)

        lines: list[str] = []
        if allies:
            lines.append("Your team: " + ", ".join(allies[:5]))
        if enemies:
            lines.append("Enemy team: " + ", ".join(enemies[:5]))
        return lines

    def _lcu_snapshot(self) -> LeagueSnapshot:
        gameflow = self._lcu_get("/lol-gameflow/v1/session")
        if not isinstance(gameflow, dict):
            return LeagueSnapshot()

        phase = str(gameflow.get("phase") or "None")
        if phase in {"None", "Terminate", "WaitingForStats", "PreEndOfGame", "EndOfGame"}:
            return LeagueSnapshot()

        if phase == "ChampSelect":
            return self._champ_select_snapshot(gameflow)

        if phase in {"Lobby", "Matchmaking", "ReadyCheck"}:
            queue = self._queue_label(gameflow)
            return LeagueSnapshot(
                active=True,
                phase=phase.lower(),
                summary=f"League: {phase.lower()} ({queue})",
                context_block="\n".join(
                    [
                        "[League client data]",
                        f"Client phase: {phase} — {queue}",
                        f"Summoner: {self.player_name or 'unknown'}",
                    ]
                ),
            )

        if phase == "InProgress":
            loading = self._loading_screen_snapshot(gameflow)
            if loading.active:
                return loading

        return LeagueSnapshot(
            active=True,
            phase=phase.lower(),
            summary=f"League: {phase.lower()}",
            context_block=f"[League client data]\nClient phase: {phase}",
        )

    def _queue_label(self, gameflow: dict[str, Any]) -> str:
        game_data = gameflow.get("gameData") or {}
        queue = game_data.get("queue") or {}
        name = queue.get("name") or queue.get("type") or "queue"
        return str(name).replace("_", " ")

    def _loading_screen_snapshot(self, gameflow: dict[str, Any]) -> LeagueSnapshot:
        game_data = gameflow.get("gameData") or {}
        selections = game_data.get("playerChampionSelections") or []
        yours: dict[str, Any] | None = None
        for selection in selections:
            if not isinstance(selection, dict):
                continue
            internal = str(selection.get("summonerInternalName") or "")
            if self.player_name and _matches_player(internal, self.player_name):
                yours = selection
                break

        if yours is None and selections:
            yours = selections[0]

        champion = self._champion_name(yours.get("championId") if yours else None)
        return LeagueSnapshot(
            active=True,
            phase="loading",
            summary=f"League: loading — {champion}",
            context_block="\n".join(
                [
                    "[League client data]",
                    "Game loading — live match API not ready yet.",
                    f"You ({self.player_name or 'player'}): {champion}",
                ]
            ),
        )

    def _champ_select_snapshot(self, gameflow: dict[str, Any]) -> LeagueSnapshot:
        session = self._lcu_get("/lol-champ-select/v1/session")
        if not isinstance(session, dict):
            queue = self._queue_label(gameflow)
            return LeagueSnapshot(
                active=True,
                phase="champ_select",
                summary=f"League: champion select ({queue})",
                context_block=f"[League client data]\nChampion select — {queue}",
            )

        local_cell = session.get("localPlayerCellId")
        my_team = session.get("myTeam") or []
        their_team = session.get("theirTeam") or []
        queue = self._queue_label(gameflow)

        your_pick = "unpicked"
        ally_lines: list[str] = []
        for member in my_team:
            if not isinstance(member, dict):
                continue
            name = str(member.get("summonerId") or member.get("name") or "ally")
            champion_id = member.get("championId") or member.get("championPickIntent") or 0
            champion = self._champion_name(int(champion_id or 0))
            if member.get("cellId") == local_cell:
                your_pick = champion
                name = self.player_name or name
            ally_lines.append(f"{name}: {champion}")

        enemy_lines: list[str] = []
        for member in their_team:
            if not isinstance(member, dict):
                continue
            champion_id = member.get("championId") or member.get("championPickIntent") or 0
            if int(champion_id or 0) <= 0:
                continue
            enemy_lines.append(self._champion_name(int(champion_id)))

        lines = [
            "[League client data]",
            f"Champion select — {queue}",
            f"You ({self.player_name or 'player'}): {your_pick}",
        ]
        if ally_lines:
            lines.append("Allies: " + ", ".join(ally_lines[:5]))
        if enemy_lines:
            lines.append("Enemy picks visible: " + ", ".join(enemy_lines[:5]))

        return LeagueSnapshot(
            active=True,
            phase="champ_select",
            summary=f"League: champ select — {your_pick}",
            context_block="\n".join(lines),
        )
