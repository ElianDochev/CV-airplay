from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class ControlOutput:
    steer: float
    accel: bool
    brake: bool


class ControllerState:
    def __init__(self, smooth_n: int, action_smooth_n: int):
        self.neutral_by_mode: dict[str, Optional[float]] = {"2-hand": None, "1-hand": None}
        self.steer_hist = deque(maxlen=smooth_n)
        self.accel_hist = deque(maxlen=action_smooth_n)
        self.brake_hist = deque(maxlen=action_smooth_n)
        self.last = ControlOutput(steer=0.0, accel=False, brake=False)

    def set_neutral(self, angle_deg: float, mode: str) -> None:
        self.neutral_by_mode[mode] = angle_deg
        self.steer_hist.clear()

    def smooth_steer(self, steer: float) -> float:
        self.steer_hist.append(steer)
        return sum(self.steer_hist) / len(self.steer_hist)

    def smooth_action(self, value: bool, hist: deque) -> bool:
        if not value:
            hist.clear()
            return False
        hist.append(1)
        threshold = (hist.maxlen + 1) // 2 if hist.maxlen else 1
        return sum(hist) >= threshold
