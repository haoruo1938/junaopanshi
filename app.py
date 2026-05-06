"""Gradio demo: language-driven embodied navigation."""

from __future__ import annotations

import atexit
from typing import Generator

import gradio as gr

from src.mock_nav import MockNavigator
from src.targets import normalize_target
from src.thor_nav import ThorNavigator


def _build_navigator() -> tuple[object, str]:
    try:
        nav = ThorNavigator(
            scene="FloorPlan201",
            vlm_detector=None,
            allow_privileged_fallback=True,
        )
        return nav, "AI2-THOR | SimulatorDetection"
    except Exception:
        # Windows often has no downloadable AI2-THOR build; fallback keeps demo runnable.
        return MockNavigator(), "MockSim | heuristic"


navigator, backend_name = _build_navigator()
atexit.register(getattr(navigator, "close", lambda: None))


def _current_frame():
    if hasattr(navigator, "last_event"):
        from PIL import Image

        return Image.fromarray(navigator.last_event.frame)
    return navigator._render()  # type: ignore[attr-defined]


def run_command(
    user_text: str, history: list[dict]
) -> Generator[tuple[list[dict], object, str], None, None]:
    history = history or []
    target = normalize_target(user_text)

    if target is None:
        history.append({"role": "user", "content": user_text})
        history.append(
            {
                "role": "assistant",
                "content": "我目前支持导航到 sofa/bed/table/apple。请重新下达目标地点或目标物体。",
            }
        )
        yield history, _current_frame(), f"idle | backend={backend_name}"
        return

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": f"收到，我正在前往 {target}。"})
    yield history, _current_frame(), f"navigating to {target} | backend={backend_name}"

    for obs in navigator.navigate(target=target):
        yield history, obs.frame, obs.status
        if obs.arrived:
            history.append({"role": "assistant", "content": f"已到达 {target} 附近。还需要什么？"})
            yield history, obs.frame, "arrived"
            return

    history.append(
        {
            "role": "assistant",
            "content": f"这次没能稳定到达 {target}。你可以让我重试，或换一个目标。",
        }
    )
    yield history, _current_frame(), f"timeout | backend={backend_name}"


with gr.Blocks(title="Embodied Navigation MVP") as demo:
    gr.Markdown("# Embodied Navigation MVP")
    gr.Markdown(
        "输入示例：`请到沙发旁边`、`go to the bed`、`navigate to table`。"
    )
    gr.Markdown(f"当前后端: `{backend_name}`")

    with gr.Row():
        chatbot = gr.Chatbot(label="Agent Conversation", scale=2)
        image = gr.Image(type="pil", label="Robot First-person View", scale=3)

    status = gr.Textbox(
        label="Runtime Status",
        value=f"ready | backend={backend_name}",
        interactive=False,
    )
    user_input = gr.Textbox(label="Command", placeholder="请到沙发旁边")
    send_btn = gr.Button("Send")

    send_btn.click(
        fn=run_command,
        inputs=[user_input, chatbot],
        outputs=[chatbot, image, status],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
