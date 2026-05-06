"""Lightweight fallback simulator for Windows-unavailable AI2-THOR builds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class StepObservation:
    frame: Image.Image
    status: str
    arrived: bool


class MockNavigator:
    """Simple 2D room simulator with first-person-like top-down rendering."""

    def __init__(self) -> None:
        self.room_size = 10.0
        self.agent_xy = np.array([1.5, 1.5], dtype=np.float32)
        self.agent_theta = 0.0
        self.targets = {
            "sofa": np.array([8.0, 2.0], dtype=np.float32),
            "bed": np.array([8.0, 8.0], dtype=np.float32),
            "table": np.array([4.5, 5.0], dtype=np.float32),
            "apple": np.array([3.0, 7.5], dtype=np.float32),
        }

    def close(self) -> None:
        return

    def _render(self, active_target: str | None = None) -> Image.Image:
        w, h = 640, 480
        img = Image.new("RGB", (w, h), (18, 20, 24))
        draw = ImageDraw.Draw(img)

        margin = 40
        draw.rectangle([margin, margin, w - margin, h - margin], outline=(120, 130, 145), width=2)

        def to_px(xy: np.ndarray) -> tuple[int, int]:
            x = int(margin + (xy[0] / self.room_size) * (w - 2 * margin))
            y = int(margin + (xy[1] / self.room_size) * (h - 2 * margin))
            return x, y

        # Targets
        for name, xy in self.targets.items():
            x, y = to_px(xy)
            color = (230, 90, 90) if name == active_target else (150, 170, 210)
            draw.ellipse([x - 10, y - 10, x + 10, y + 10], fill=color)
            draw.text((x + 12, y - 8), name, fill=(220, 220, 230))

        # Agent
        ax, ay = to_px(self.agent_xy)
        draw.ellipse([ax - 9, ay - 9, ax + 9, ay + 9], fill=(90, 230, 120))
        dx = int(16 * np.cos(self.agent_theta))
        dy = int(16 * np.sin(self.agent_theta))
        draw.line([ax, ay, ax + dx, ay + dy], fill=(90, 230, 120), width=3)

        return img

    def navigate(
        self,
        target: str,
        max_steps: int = 120,
        reach_threshold_m: float = 0.7,
    ) -> Iterable[StepObservation]:
        target_xy = self.targets.get(target)
        if target_xy is None:
            yield StepObservation(self._render(), f"unknown target: {target}", False)
            return

        for step in range(1, max_steps + 1):
            vec = target_xy - self.agent_xy
            dist = float(np.linalg.norm(vec))
            if dist <= reach_threshold_m:
                yield StepObservation(
                    self._render(active_target=target),
                    f"[{step}] reached {target} at approx {dist:.2f}m",
                    True,
                )
                return

            desired = float(np.arctan2(vec[1], vec[0]))
            # Normalize angle to [-pi, pi]
            delta = (desired - self.agent_theta + np.pi) % (2 * np.pi) - np.pi

            if delta < -0.20:
                self.agent_theta -= 0.22
                action = "RotateLeft"
            elif delta > 0.20:
                self.agent_theta += 0.22
                action = "RotateRight"
            else:
                forward = np.array([np.cos(self.agent_theta), np.sin(self.agent_theta)], dtype=np.float32)
                self.agent_xy += 0.22 * forward
                self.agent_xy = np.clip(self.agent_xy, 0.6, self.room_size - 0.6)
                action = "MoveAhead"

            yield StepObservation(
                self._render(active_target=target),
                f"[{step}] {action}, dist={dist:.2f}m",
                False,
            )

        yield StepObservation(
            self._render(active_target=target),
            f"navigation timeout after {max_steps} steps",
            False,
        )
