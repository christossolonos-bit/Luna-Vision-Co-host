from __future__ import annotations

import threading
import time

import gradio as gr
from PIL import Image

from luna.brain import CohostBrain
from luna.config import AppConfig, load_config
from luna.screen import ScreenCapture


def _format_chat(history: list[dict[str, str]], user: str, assistant: str) -> list[dict[str, str]]:
    updated = list(history)
    updated.append({"role": "user", "content": user})
    updated.append({"role": "assistant", "content": assistant})
    return updated


class LunaApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.brain = CohostBrain(config)
        self.screen = ScreenCapture(config.screen)
        self._watch_lock = threading.Lock()
        self._watch_running = False
        self._last_frame_b64: str | None = None

    def monitor_choices(self) -> list[tuple[str, int]]:
        labels = self.screen.list_monitors()
        return [(label, index) for index, label in enumerate(labels)]

    def refresh_preview(self, monitor_index: int) -> Image.Image:
        capture = self.screen.capture(monitor_index)
        self._last_frame_b64 = capture.image_b64
        return capture.preview

    def capture_frame(self, monitor_index: int) -> tuple[Image.Image, str]:
        capture = self.screen.capture(monitor_index)
        self._last_frame_b64 = capture.image_b64
        return capture.preview, capture.source_label

    def status(self) -> str:
        ok, message = self.brain.ping()
        return f"{'OK' if ok else 'ERROR'} — {message}"

    def chat_with_vision(
        self,
        message: str,
        history: list[dict[str, str]],
        monitor_index: int,
        use_screen: bool,
        style: str,
    ) -> tuple[list[dict[str, str]], Image.Image | None, str]:
        if not message.strip():
            return history, None, self.status()

        self.config.cohost.style = style
        image_b64 = None
        preview = None
        if use_screen:
            preview, _ = self.capture_frame(monitor_index)
            image_b64 = self._last_frame_b64

        try:
            reply = self.brain.ask(message.strip(), image_b64)
            updated = _format_chat(history, message.strip(), reply)
            return updated, preview, self.status()
        except Exception as exc:  # noqa: BLE001
            error = f"Could not reach Luna's brain: {exc}"
            updated = _format_chat(history, message.strip(), error)
            return updated, preview, self.status()

    def analyze_screen(
        self,
        history: list[dict[str, str]],
        monitor_index: int,
        style: str,
    ) -> tuple[list[dict[str, str]], Image.Image, str]:
        self.config.cohost.style = style
        preview, label = self.capture_frame(monitor_index)
        try:
            reply = self.brain.react_to_screen(self._last_frame_b64 or "")
            user_line = f"[Screen capture — {label}] What do you see?"
            updated = _format_chat(history, user_line, reply)
            return updated, preview, self.status()
        except Exception as exc:  # noqa: BLE001
            updated = _format_chat(history, "[Screen capture]", f"Error: {exc}")
            return updated, preview, self.status()

    def watch_tick(
        self,
        history: list[dict[str, str]],
        monitor_index: int,
        style: str,
        enabled: bool,
        interval: float,
    ) -> tuple[list[dict[str, str]], Image.Image | None, str, str]:
        if not enabled:
            return history, None, self.status(), "Watch mode off"

        with self._watch_lock:
            if self._watch_running:
                return history, None, self.status(), f"Watch mode on — every {interval:.0f}s"
            self._watch_running = True

        try:
            self.config.cohost.style = style
            preview, label = self.capture_frame(monitor_index)
            reply = self.brain.react_to_screen(
                self._last_frame_b64 or "",
                remember=False,
                include_history=True,
            )
            user_line = f"[Auto watch — {label}]"
            updated = _format_chat(history, user_line, reply)
            return updated, preview, self.status(), f"Watch mode on — last tick {time.strftime('%H:%M:%S')}"
        except Exception as exc:  # noqa: BLE001
            updated = _format_chat(history, "[Auto watch]", f"Error: {exc}")
            return updated, None, self.status(), f"Watch error: {exc}"
        finally:
            with self._watch_lock:
                self._watch_running = False

    def clear_memory(self) -> tuple[list[dict[str, str]], str]:
        self.brain.reset()
        return [], "Conversation cleared."


def build_ui(config: AppConfig) -> gr.Blocks:
    app = LunaApp(config)
    monitors = app.monitor_choices()
    default_monitor = min(config.screen.monitor, len(monitors) - 1)

    with gr.Blocks(title="Luna Gaming Co-Host") as demo:
        gr.Markdown(
            """
            # Luna Gaming Co-Host
            Local AI co-host powered by **Ollama `qwen3.5:4b`** with live screen vision.
            Run your game on one monitor and keep this window on another, or capture your main display.
            """
        )

        with gr.Row():
            status_box = gr.Textbox(label="Ollama status", value=app.status(), interactive=False)
            watch_status = gr.Textbox(label="Watch mode", value="Watch mode off", interactive=False)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Luna", height=420)
                with gr.Row():
                    message = gr.Textbox(
                        label="Message",
                        placeholder="Ask Luna about your game, e.g. 'What should I do here?'",
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    use_screen = gr.Checkbox(label="Include current screen", value=True)
                    analyze_btn = gr.Button("Analyze screen now")
                    clear_btn = gr.Button("Clear memory")

            with gr.Column(scale=2):
                preview = gr.Image(label="Screen preview", type="pil", height=360)
                monitor = gr.Dropdown(
                    label="Capture monitor",
                    choices=monitors,
                    value=default_monitor,
                )
                style = gr.Dropdown(
                    label="Co-host style",
                    choices=list(["energetic", "tactical", "chill"]),
                    value=config.cohost.style,
                )
                watch_enabled = gr.Checkbox(label="Auto watch (live commentary)", value=False)
                interval = gr.Slider(
                    label="Watch interval (seconds)",
                    minimum=5,
                    maximum=60,
                    step=1,
                    value=int(config.screen.capture_interval_sec),
                )
                refresh_btn = gr.Button("Refresh preview")

        demo.load(app.status, outputs=status_box)
        demo.load(lambda idx=default_monitor: app.refresh_preview(idx), outputs=preview)

        refresh_btn.click(app.refresh_preview, inputs=monitor, outputs=preview)

        send_event = send_btn.click(
            app.chat_with_vision,
            inputs=[message, chatbot, monitor, use_screen, style],
            outputs=[chatbot, preview, status_box],
        )
        message.submit(
            app.chat_with_vision,
            inputs=[message, chatbot, monitor, use_screen, style],
            outputs=[chatbot, preview, status_box],
        )
        send_event.then(lambda: "", outputs=message)
        message.submit(lambda: "", outputs=message)

        analyze_btn.click(
            app.analyze_screen,
            inputs=[chatbot, monitor, style],
            outputs=[chatbot, preview, status_box],
        )

        clear_btn.click(app.clear_memory, outputs=[chatbot, watch_status])

        timer = gr.Timer(value=config.screen.capture_interval_sec, active=False)

        def _watch_label(enabled: bool, seconds: float) -> tuple[dict, str]:
            return (
                gr.Timer(active=enabled, value=max(5.0, seconds)),
                ("Watch mode on" if enabled else "Watch mode off"),
            )

        watch_enabled.change(
            _watch_label,
            inputs=[watch_enabled, interval],
            outputs=[timer, watch_status],
        )
        interval.release(
            _watch_label,
            inputs=[watch_enabled, interval],
            outputs=[timer, watch_status],
        )

        timer.tick(
            app.watch_tick,
            inputs=[chatbot, monitor, style, watch_enabled, interval],
            outputs=[chatbot, preview, status_box, watch_status],
        )

    return demo


def _custom_css() -> str:
    return """
    .gradio-container { max-width: 1100px !important; }
    footer { display: none !important; }
    """


def launch(config_path: str | None = None) -> None:
    config = load_config(config_path)
    demo = build_ui(config)
    theme = gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="purple",
        neutral_hue="slate",
    ).set(
        body_background_fill="*neutral_950",
        block_background_fill="*neutral_900",
        block_border_color="*neutral_700",
        body_text_color="*neutral_100",
    )
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=config.ui.host,
        server_port=config.ui.port,
        share=config.ui.share,
        inbrowser=True,
        theme=theme,
        css=_custom_css(),
    )
