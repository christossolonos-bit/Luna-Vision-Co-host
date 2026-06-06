from __future__ import annotations

import os

import win32gui
import win32process

MIN_GAME_WIDTH = 1024
MIN_GAME_HEIGHT = 576

TITLE_BLOCKLIST = (
    "chrome",
    "microsoft edge",
    "edge",
    "firefox",
    "opera",
    "brave",
    "cursor",
    "visual studio",
    "code",
    "windows terminal",
    "powershell",
    "command prompt",
    "task manager",
    "settings",
    "spotify",
    "discord",
    "slack",
    "teams",
    "zoom",
    "obs",
    "streamlabs",
    "nvidia",
    "geforce",
    "amd software",
    "luna gaming",
    "program manager",
    "task switching",
)

EXE_BLOCKLIST = {
    "applicationframehost.exe",
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
    "opera.exe",
    "brave.exe",
    "cursor.exe",
    "code.exe",
    "devenv.exe",
    "windowsterminal.exe",
    "powershell.exe",
    "cmd.exe",
    "explorer.exe",
    "searchhost.exe",
    "shellexperiencehost.exe",
    "textinputhost.exe",
    "systemsettings.exe",
    "taskmgr.exe",
    "spotify.exe",
    "discord.exe",
    "slack.exe",
    "teams.exe",
    "zoom.exe",
    "obs64.exe",
    "obs32.exe",
    "streamlabs.exe",
    "python.exe",
    "pythonw.exe",
    "python3.exe",
    "windowspackager.exe",
}

KNOWN_GAME_EXES = {
    "league of legends.exe",
    "leagueclientux.exe",
    "leagueclient.exe",
    "riotclientux.exe",
    "riot client.exe",
    "valorant-win64-shipping.exe",
    "valorant.exe",
    "cs2.exe",
    "csgo.exe",
    "fortniteclient-win64-shipping.exe",
    "rocketleague.exe",
}


def _process_exe(pid: int) -> str:
    import win32api
    import win32con

    try:
        handle = win32api.OpenProcess(
            win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ,
            False,
            pid,
        )
        exe_path = win32process.GetModuleFileNameEx(handle, 0)
        win32api.CloseHandle(handle)
        return os.path.basename(exe_path).lower()
    except Exception:  # noqa: BLE001
        return ""


def _is_likely_game(title: str, exe_name: str, width: int, height: int) -> bool:
    if exe_name in KNOWN_GAME_EXES:
        return width >= 800 and height >= 600

    if width < MIN_GAME_WIDTH or height < MIN_GAME_HEIGHT:
        return False

    lowered_title = title.lower()
    if any(blocked in lowered_title for blocked in TITLE_BLOCKLIST):
        return False

    if exe_name in EXE_BLOCKLIST:
        return False

    return True


def list_game_windows() -> list[dict[str, str | int]]:
    windows: list[dict[str, str | int]] = []

    def callback(hwnd: int, _: object) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        if win32gui.IsIconic(hwnd):
            return True

        title = win32gui.GetWindowText(hwnd).strip()
        if not title:
            return True

        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        exe_name = _process_exe(pid)
        if not _is_likely_game(title, exe_name, width, height):
            return True

        windows.append(
            {
                "id": f"window:{hwnd}",
                "hwnd": hwnd,
                "title": title,
                "width": width,
                "height": height,
                "pid": pid,
            }
        )
        return True

    win32gui.EnumWindows(callback, None)
    windows.sort(key=lambda item: str(item["title"]).lower())
    return windows
