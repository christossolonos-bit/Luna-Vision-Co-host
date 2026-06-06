# Luna Vision Co-Host

> *"Hey — I'm Luna. I watch your screen, talk with you, and hype your plays. Want a co-host who actually sees the game? You're in the right place."*

**Luna Vision Co-Host** is a local AI gaming companion that runs on your machine. She uses [Ollama](https://ollama.com) with **qwen3.5:4b** for vision and chat, **Edge TTS** for voice, and **Whisper** for speech — all without sending your gameplay to the cloud.

---

## What Luna can do

- **Screen vision** — sees your game via monitor or game-window capture
- **Voice chat** — toggle **Listen ON** and talk naturally; she ignores background noise
- **Text chat** — ask questions with optional screen context
- **VRM avatar** — 3D Luna with idle animation and lip sync
- **OBS overlay** — green-screen route at `/obs` for chroma key streaming
- **Co-host styles** — energetic, tactical, or chill

---

## Requirements

- **Windows 10/11**
- **Python 3.11+**
- **[Ollama](https://ollama.com)** with the vision model:

```bash
ollama pull qwen3.5:4b
```

- A **VRM model** and optional **VRMA idle animation** (paths go in config)
- Microphone for voice listen mode

---

## Quick start

1. **Clone the repo**

```bash
git clone https://github.com/christossolonos-bit/Luna-Vision-Co-host.git
cd Luna-Vision-Co-host
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure Luna**

```bash
copy config.example.yaml config.yaml
```

Edit `config.yaml`:

- Set `cohost.player_name` to your in-game name
- Point `vrm.model_path` and `vrm.idle_animation_path` to your VRM files

4. **Run**

```bash
python main.py
```

Or double-click `run.bat`.

Open **http://127.0.0.1:7860** — I'll be waiting.

---

## OBS streaming

1. Keep the main window for controls and chat.
2. Add a **Browser Source** in OBS:
   - URL: `http://127.0.0.1:7860/obs`
   - Size: e.g. 1920×1080
3. Add a **Chroma Key** filter (green `#00ff00`).
4. When the OBS overlay is open, TTS plays there — not on the control window.

---

## Screen capture tips

- Use **Screen 1 / Screen 2** for the monitor that shows your game.
- Prefer **borderless windowed** in games — fullscreen often captures as black.
- Check the **capture preview** thumbnail in chat to confirm I can see what you see.

---

## Project layout

```
Luna-Vision-Co-host/
├── main.py              # Entry point
├── config.example.yaml  # Copy to config.yaml
├── luna/
│   ├── server.py        # FastAPI backend
│   ├── brain.py         # Ollama vision + chat
│   ├── screen.py        # Screen / window capture
│   ├── voice.py         # Edge TTS
│   ├── speech.py        # Whisper STT
│   └── static/          # Web UI + VRM viewer
└── requirements.txt
```

---

## Tech stack

| Piece | Tool |
|-------|------|
| Vision + chat | Ollama `qwen3.5:4b` |
| Voice | Edge TTS (Ava Multilingual Neural) |
| Speech-to-text | faster-whisper |
| UI | FastAPI + Three.js VRM |
| Capture | mss + Win32 |

---

## License

MIT — see [LICENSE](LICENSE).

---

*Pick your screen, turn Listen on, and let's run it back together.*

— **Luna**
