"""CLI demo for embodied navigation without web UI."""

from __future__ import annotations

import argparse
import atexit
import os
from typing import Iterable

from src.mock_nav import MockNavigator
from src.targets import normalize_target
from src.thor_nav import StepObservation, ThorNavigator
from src.vlm_detector import VLMDetector


def build_navigator(force_mock: bool = False):
    """Build navigator with AI2-THOR first, then fallback to mock."""
    if force_mock:
        return MockNavigator(), "MockSim | forced"

    detector = VLMDetector.from_env()
    allow_privileged_fallback = os.getenv("ALLOW_PRIVILEGED_FALLBACK", "1").strip() != "0"
    try:
        nav = ThorNavigator(
            scene="FloorPlan201",
            vlm_detector=detector,
            allow_privileged_fallback=allow_privileged_fallback,
        )
        perception = "VLM+THOR" if detector is not None else "THOR-only"
        return nav, f"AI2-THOR | {perception}"
    except Exception:
        return MockNavigator(), "MockSim | auto-fallback"


def run_once(command: str, nav, backend_name: str) -> None:
    """Run one command and print streaming navigation logs."""
    target = normalize_target(command)
    print(f"[User] {command}")
    if target is None:
        print("[Agent] 当前仅支持 sofa/bed/table/apple，请换个目标。")
        return

    print(f"[Agent] 收到，正在前往 {target}。")
    observations: Iterable[StepObservation] = nav.navigate(target=target)

    final_status = "未完成"
    for idx, obs in enumerate(observations, start=1):
        if idx <= 5 or idx % 10 == 0 or obs.arrived:
            print(f"[Nav] {obs.status}")
        final_status = obs.status
        if obs.arrived:
            print(f"[Agent] 已到达 {target} 附近。还需要什么？")
            return

    print(f"[Agent] 本次未稳定到达 {target}。最终状态: {final_status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embodied Navigation CLI Demo")
    parser.add_argument("--command", type=str, default="", help="Single command mode, e.g. 'go to sofa'")
    parser.add_argument("--mock", action="store_true", help="Force mock backend")
    args = parser.parse_args()

    navigator, backend = build_navigator(force_mock=args.mock)
    atexit.register(getattr(navigator, "close", lambda: None))
    print(f"[System] backend={backend}")

    if args.command.strip():
        run_once(args.command.strip(), navigator, backend)
        return

    print("[System] 进入交互模式，输入 quit 退出。")
    while True:
        text = input("You> ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            print("[System] 已退出。")
            return
        if not text:
            continue
        run_once(text, navigator, backend)


if __name__ == "__main__":
    main()
