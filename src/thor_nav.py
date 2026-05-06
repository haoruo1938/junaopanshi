"""AI2-THOR navigation primitives and visual servo controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from ai2thor.controller import Controller
from PIL import Image

from src.vlm_detector import Detection, VLMDetector

TARGET_OBJECT_TYPES: dict[str, set[str]] = {
    "sofa": {"Sofa", "ArmChair"},
    "bed": {"Bed"},
    "table": {"CoffeeTable", "DiningTable", "Desk", "SideTable", "TVStand", "CounterTop"},
    "apple": {"Apple"},
}

TARGET_REACH_THRESHOLD_M: dict[str, float] = {
    # Large objects are often not physically reachable to <0.7m due to collision hull.
    "sofa": 1.15,
    "bed": 1.05,
    "table": 0.95,
    "apple": 0.70,
}


@dataclass
class StepObservation:
    frame: Image.Image
    status: str
    arrived: bool


class ThorNavigator:
    """Wrapper around AI2-THOR for RGB-D based target seeking."""

    def __init__(
        self,
        scene: str = "FloorPlan201",
        vlm_detector: VLMDetector | None = None,
        allow_privileged_fallback: bool = False,
    ) -> None:
        # Do not force CloudRendering: Windows often has no cloud build.
        # Let AI2-THOR auto-select a platform compatible with local OS.
        self.controller = Controller(
            scene=scene,
            width=640,
            height=480,
            renderDepthImage=True,
            renderInstanceSegmentation=True,
            visibilityDistance=1.5,
        )
        self.last_event = self.controller.last_event
        self.vlm_detector = vlm_detector
        self.allow_privileged_fallback = allow_privileged_fallback

    def reset(self, scene: str | None = None) -> None:
        if scene:
            self.last_event = self.controller.reset(scene=scene)
        else:
            current = self.controller.last_event.metadata.get("sceneName", "FloorPlan201")
            self.last_event = self.controller.reset(scene=current)

    def close(self) -> None:
        self.controller.stop()

    def _visible_target_bbox(self, canonical_target: str) -> tuple[int, int, int, int] | None:
        if self.vlm_detector is not None:
            det = self._detect_with_vlm(canonical_target)
            if det is not None:
                return det
            if not self.allow_privileged_fallback:
                return None

        if not self.allow_privileged_fallback:
            return None

        # First choice: use AI2-THOR's 2D instance detections when available.
        detections_2d = getattr(self.last_event, "instance_detections2D", None) or {}
        objects = {obj["objectId"]: obj for obj in self.last_event.metadata.get("objects", [])}
        wanted = TARGET_OBJECT_TYPES.get(canonical_target, set())
        if not wanted:
            return None

        best_bbox = None
        best_area = -1
        for object_id, bbox in detections_2d.items():
            obj = objects.get(object_id)
            if not obj:
                continue
            if obj.get("objectType") not in wanted:
                continue
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > best_area:
                best_area = area
                best_bbox = (x1, y1, x2, y2)
        if best_bbox is not None:
            return best_bbox

        # Fallback: search visible masks by matching segmentation color with object type.
        seg_frame = self.last_event.instance_segmentation_frame
        if seg_frame is None:
            return None
        color_to_object_id = self.last_event.color_to_object_id
        for color, object_id in color_to_object_id.items():
            obj = objects.get(object_id)
            if not obj:
                continue
            if obj.get("objectType") not in wanted:
                continue
            color_arr = np.array(color, dtype=np.uint8)
            mask = np.all(seg_frame == color_arr, axis=2)
            if not np.any(mask):
                continue
            ys, xs = np.where(mask)
            return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        return None

    def _detect_with_vlm(self, canonical_target: str) -> tuple[int, int, int, int] | None:
        rgb = Image.fromarray(self.last_event.frame)
        detection: Detection = self.vlm_detector.detect(rgb, canonical_target)  # type: ignore[union-attr]
        if detection.found and detection.bbox is not None and detection.confidence >= 0.35:
            return detection.bbox
        return None

    def _distance_from_depth(self, x: int, y: int) -> float:
        depth = self.last_event.depth_frame
        if depth is None:
            return 999.0
        h, w = depth.shape
        x0, x1 = max(0, x - 4), min(w - 1, x + 4)
        y0, y1 = max(0, y - 4), min(h - 1, y + 4)
        patch = depth[y0 : y1 + 1, x0 : x1 + 1]
        valid = patch[np.isfinite(patch)]
        if valid.size == 0:
            return 999.0
        return float(np.median(valid))

    def _as_image(self) -> Image.Image:
        return Image.fromarray(self.last_event.frame)

    def _step(self, action: str, **kwargs) -> tuple[bool, str]:
        self.last_event = self.controller.step(action=action, **kwargs)
        meta = self.last_event.metadata
        return bool(meta.get("lastActionSuccess", True)), str(meta.get("errorMessage", ""))

    def _depth_sector_clearance(self) -> tuple[float, float, float]:
        """Estimate free space from depth image for left/center/right sectors."""
        depth = self.last_event.depth_frame
        if depth is None:
            return 0.0, 0.0, 0.0
        h, w = depth.shape
        top = int(h * 0.35)
        bottom = int(h * 0.85)
        left_slice = depth[top:bottom, : int(w * 0.33)]
        center_slice = depth[top:bottom, int(w * 0.33) : int(w * 0.66)]
        right_slice = depth[top:bottom, int(w * 0.66) :]

        def robust_median(arr: np.ndarray) -> float:
            valid = arr[np.isfinite(arr)]
            if valid.size == 0:
                return 0.0
            return float(np.median(valid))

        return robust_median(left_slice), robust_median(center_slice), robust_median(right_slice)

    def _smart_recover(self, blocked_counter: int) -> str:
        """Depth-guided obstacle bypass to avoid spinning in place."""
        left_clear, center_clear, right_clear = self._depth_sector_clearance()

        # Prefer the side with more free space, then keep advancing on that heading.
        prefer_left = left_clear >= right_clear
        turn = "RotateLeft" if prefer_left else "RotateRight"
        self._step(action=turn, degrees=35)

        ok_ahead_1, err_ahead_1 = self._step(action="MoveAhead", moveMagnitude=0.25)
        ok_ahead_2, err_ahead_2 = self._step(action="MoveAhead", moveMagnitude=0.25)
        if ok_ahead_1 or ok_ahead_2:
            return f"smart-detour:{turn}+MoveAheadx2"

        # Escalate: bigger sweep turn every few failures to break local loops.
        big_turn_deg = 35 if blocked_counter < 5 else 60
        sweep_turn = "RotateRight" if blocked_counter % 2 == 0 else "RotateLeft"
        self._step(action=sweep_turn, degrees=big_turn_deg)
        ok_sweep_ahead, err_sweep = self._step(action="MoveAhead", moveMagnitude=0.25)
        return (
            f"recover-sweep:{sweep_turn}({big_turn_deg}) "
            f"depth(L/C/R)=({left_clear:.2f}/{center_clear:.2f}/{right_clear:.2f}) "
            f"ahead_err={err_ahead_1 or err_ahead_2 or err_sweep or 'n/a'} "
            f"sweep_ahead={'ok' if ok_sweep_ahead else 'blocked'}"
        )

    def _try_detour(self) -> tuple[bool, str]:
        """Try short left/right bypass sequence when front movement is blocked."""
        # Left bypass
        ok1, err1 = self._step(action="RotateLeft", degrees=30)
        ok2, err2 = self._step(action="MoveAhead", moveMagnitude=0.25)
        if ok1 and ok2:
            return True, "detour-left"
        self._step(action="RotateRight", degrees=15)

        # Right bypass
        ok3, err3 = self._step(action="RotateRight", degrees=45)
        ok4, err4 = self._step(action="MoveAhead", moveMagnitude=0.25)
        if ok3 and ok4:
            return True, "detour-right"
        self._step(action="RotateLeft", degrees=15)

        return False, f"detour-failed: {err2 or err4 or err1 or err3}"

    def _emergency_recover(self) -> str:
        """Hard recovery sequence when agent is repeatedly stuck."""
        self._step(action="RotateRight", degrees=90)
        ok1, _ = self._step(action="MoveAhead", moveMagnitude=0.30)
        self._step(action="RotateLeft", degrees=45)
        ok2, _ = self._step(action="MoveAhead", moveMagnitude=0.30)
        self._step(action="RotateLeft", degrees=45)
        return f"emergency-recover:move1={'ok' if ok1 else 'blocked'},move2={'ok' if ok2 else 'blocked'}"

    def _break_spin(self) -> str:
        """Break pure-rotation loops with a forced displacement burst."""
        left_clear, _, right_clear = self._depth_sector_clearance()
        first_turn = "RotateLeft" if left_clear >= right_clear else "RotateRight"
        second_turn = "RotateRight" if first_turn == "RotateLeft" else "RotateLeft"

        self._step(action=first_turn, degrees=45)
        ok_moves = 0
        for _ in range(3):
            ok, _ = self._step(action="MoveAhead", moveMagnitude=0.25)
            if ok:
                ok_moves += 1
            else:
                self._step(action=second_turn, degrees=30)
        return f"break-spin:{first_turn},move_ok={ok_moves}/3"

    def navigate(
        self,
        target: str,
        max_steps: int = 120,
        reach_threshold_m: float = 0.7,
    ) -> Iterable[StepObservation]:
        """Yield frame-by-frame progress using visual servoing heuristics."""
        effective_reach_m = max(reach_threshold_m, TARGET_REACH_THRESHOLD_M.get(target, reach_threshold_m))
        lost_counter = 0
        blocked_counter = 0
        center_band = 0.18
        recent_distances: list[float] = []
        bypass_steps_remaining = 0
        bypass_turn = "RotateLeft"
        hard_stuck_counter = 0
        no_progress_counter = 0
        rotate_streak = 0

        for step in range(1, max_steps + 1):
            if bypass_steps_remaining > 0:
                if bypass_steps_remaining == 4:
                    self._step(action=bypass_turn, degrees=35)
                    action_note = f"bypass-enter:{bypass_turn}"
                    rotate_streak = 0
                else:
                    ok_bypass, _ = self._step(action="MoveAhead", moveMagnitude=0.25)
                    action_note = f"bypass-move:{'ok' if ok_bypass else 'blocked'}"
                    if not ok_bypass:
                        # Flip bypass direction when still blocked.
                        bypass_turn = "RotateRight" if bypass_turn == "RotateLeft" else "RotateLeft"
                        self._step(action=bypass_turn, degrees=40)
                    else:
                        rotate_streak = 0
                bypass_steps_remaining -= 1
                yield StepObservation(
                    frame=self._as_image(),
                    status=f"[{step}] {action_note}",
                    arrived=False,
                )
                continue

            bbox = self._visible_target_bbox(target)
            if bbox is None:
                lost_counter += 1
                if lost_counter % 2 == 0:
                    ok, _ = self._step(action="MoveAhead", moveMagnitude=0.25)
                    action = "MoveAhead(explore)" if ok else "MoveAheadBlocked(explore)"
                    if ok:
                        rotate_streak = 0
                else:
                    left_clear, _, right_clear = self._depth_sector_clearance()
                    if left_clear >= right_clear:
                        action_name = "RotateLeft"
                    else:
                        action_name = "RotateRight"
                    self._step(action=action_name, degrees=20)
                    action = action_name
                    rotate_streak += 1

                if rotate_streak >= 6:
                    burst_note = self._break_spin()
                    action = f"{action} -> {burst_note}"
                    rotate_streak = 0
                    lost_counter = 0
                yield StepObservation(
                    frame=self._as_image(),
                    status=f"[{step}] target not visible, scanning by {action}",
                    arrived=False,
                )
                continue

            lost_counter = 0
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            frame_h, frame_w, _ = self.last_event.frame.shape
            norm_x = (cx / frame_w) * 2 - 1
            distance_m = self._distance_from_depth(cx, cy)
            recent_distances.append(distance_m)
            if len(recent_distances) > 6:
                recent_distances.pop(0)
            if len(recent_distances) >= 2 and (recent_distances[-2] - recent_distances[-1]) < 0.02:
                no_progress_counter += 1
            else:
                no_progress_counter = max(0, no_progress_counter - 1)

            if distance_m <= effective_reach_m:
                yield StepObservation(
                    frame=self._as_image(),
                    status=f"[{step}] reached {target} at approx {distance_m:.2f}m (threshold {effective_reach_m:.2f})",
                    arrived=True,
                )
                return

            if norm_x < -center_band:
                ok, err = self._step(action="RotateLeft", degrees=10)
                action_note = "RotateLeft"
                if not ok:
                    action_note = f"RotateLeftBlocked:{err or 'unknown'}"
                rotate_streak += 1
            elif norm_x > center_band:
                ok, err = self._step(action="RotateRight", degrees=10)
                action_note = "RotateRight"
                if not ok:
                    action_note = f"RotateRightBlocked:{err or 'unknown'}"
                rotate_streak += 1
            else:
                ok, err = self._step(action="MoveAhead", moveMagnitude=0.25)
                if ok:
                    blocked_counter = 0
                    hard_stuck_counter = max(0, hard_stuck_counter - 1)
                    action_note = "MoveAhead"
                    rotate_streak = 0
                else:
                    blocked_counter += 1
                    hard_stuck_counter += 1
                    detour_ok, detour_note = self._try_detour()
                    if detour_ok:
                        action_note = f"MoveAheadBlocked:{err or 'collision'} -> {detour_note}"
                    else:
                        smart_note = self._smart_recover(blocked_counter=blocked_counter)
                        action_note = f"MoveAheadBlocked:{err or 'collision'} -> {detour_note} -> {smart_note}"
                    if blocked_counter >= 2:
                        left_clear, _, right_clear = self._depth_sector_clearance()
                        bypass_turn = "RotateLeft" if left_clear >= right_clear else "RotateRight"
                        bypass_steps_remaining = 4
                    if hard_stuck_counter >= 5 or no_progress_counter >= 8:
                        emergency_note = self._emergency_recover()
                        action_note += f" | {emergency_note}"
                        hard_stuck_counter = 0
                        no_progress_counter = 0
                        rotate_streak = 0

                if rotate_streak >= 8:
                    burst_note = self._break_spin()
                    action_note += f" | {burst_note}"
                    rotate_streak = 0

                # If repeatedly blocked but already near a bulky target, accept arrival.
                if blocked_counter >= 3 and distance_m <= (effective_reach_m + 0.2):
                    yield StepObservation(
                        frame=self._as_image(),
                        status=f"[{step}] near-target stop accepted at {distance_m:.2f}m after repeated block",
                        arrived=True,
                    )
                    return

                # Stagnation guard: if distance does not improve, force exploratory sweep.
                if len(recent_distances) >= 5:
                    progress = recent_distances[0] - recent_distances[-1]
                    if progress < 0.08 and blocked_counter >= 2:
                        sweep_turn = "RotateLeft" if (step % 2 == 0) else "RotateRight"
                        self._step(action=sweep_turn, degrees=55)
                        ok_explore, _ = self._step(action="MoveAhead", moveMagnitude=0.25)
                        action_note += f" | stagnation-> {sweep_turn}+MoveAhead({'ok' if ok_explore else 'blocked'})"

            yield StepObservation(
                frame=self._as_image(),
                status=f"[{step}] {action_note}, dist={distance_m:.2f}m, bbox=({x1},{y1})-({x2},{y2})",
                arrived=False,
            )

        yield StepObservation(
            frame=self._as_image(),
            status=f"navigation timeout after {max_steps} steps",
            arrived=False,
        )
