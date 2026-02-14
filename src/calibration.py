from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import yaml

from .config_utils import get_cfg
from .hand_utils import (
    compute_neutral_and_max,
    hand_finger_pattern,
    steering_angle_one_hand,
    steering_angle_two_hands,
)


@dataclass
class CalibrationData:
    version: int
    created_at: str
    two_hand_left_deg: float
    two_hand_right_deg: float
    two_hand_neutral_deg: float
    two_hand_max_steer_deg: float
    one_hand_left_deg: float
    one_hand_right_deg: float
    one_hand_neutral_deg: float
    one_hand_max_steer_deg: float
    brake_left: Optional[dict]
    brake_right: Optional[dict]
    accel_left: Optional[dict]
    accel_right: Optional[dict]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "two_hand_left_deg": self.two_hand_left_deg,
            "two_hand_right_deg": self.two_hand_right_deg,
            "two_hand_neutral_deg": self.two_hand_neutral_deg,
            "two_hand_max_steer_deg": self.two_hand_max_steer_deg,
            "one_hand_left_deg": self.one_hand_left_deg,
            "one_hand_right_deg": self.one_hand_right_deg,
            "one_hand_neutral_deg": self.one_hand_neutral_deg,
            "one_hand_max_steer_deg": self.one_hand_max_steer_deg,
            "brake_left": self.brake_left,
            "brake_right": self.brake_right,
            "accel_left": self.accel_left,
            "accel_right": self.accel_right,
        }


def calibration_root() -> Path:
    return Path(__file__).resolve().parents[1] / "calibration"


def calibration_path() -> Path:
    return calibration_root() / "calibration.yml"


def load_calibration(path: Path) -> Optional[CalibrationData]:
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        return None
    with path.open("r", encoding="utf-8") as handle:
        try:
            data = yaml.safe_load(handle)
        except (yaml.YAMLError, TypeError, ValueError):
            return None
    if not isinstance(data, dict):
        return None

    try:
        return CalibrationData(
            version=int(data.get("version", 1)),
            created_at=str(data.get("created_at", "")),
            two_hand_left_deg=float(data["two_hand_left_deg"]),
            two_hand_right_deg=float(data["two_hand_right_deg"]),
            two_hand_neutral_deg=float(data["two_hand_neutral_deg"]),
            two_hand_max_steer_deg=float(data["two_hand_max_steer_deg"]),
            one_hand_left_deg=float(data["one_hand_left_deg"]),
            one_hand_right_deg=float(data["one_hand_right_deg"]),
            one_hand_neutral_deg=float(data["one_hand_neutral_deg"]),
            one_hand_max_steer_deg=float(data["one_hand_max_steer_deg"]),
            brake_left=data.get("brake_left") if isinstance(data.get("brake_left"), dict) else None,
            brake_right=data.get("brake_right") if isinstance(data.get("brake_right"), dict) else None,
            accel_left=data.get("accel_left") if isinstance(data.get("accel_left"), dict) else None,
            accel_right=data.get("accel_right") if isinstance(data.get("accel_right"), dict) else None,
        )
    except (KeyError, TypeError, ValueError):
        return None


def save_calibration(path: Path, calibration: CalibrationData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(calibration.to_dict(), handle, sort_keys=False)


def _single_hand(left, right) -> Tuple[Optional[object], Optional[str]]:
    if left is not None and right is None:
        return left, "Left"
    if right is not None and left is None:
        return right, "Right"
    return None, None


class CalibrationSession:
    def __init__(self, show_ui: bool, countdown_seconds: int = 3, stage_duration: float = 3.0) -> None:
        self.show_ui = show_ui
        self.countdown_seconds = countdown_seconds
        self.stage_delay_seconds = 4.0
        self.stage_duration = stage_duration
        self.start_time: Optional[float] = None
        self.countdown_end: Optional[float] = None
        self.go_end: Optional[float] = None
        self.waiting_for_start = show_ui
        self.step_index = 0
        self.samples: list[float] = []
        self.data: dict[str, float | str] = {}
        self.brake_left: Optional[dict] = None
        self.brake_right: Optional[dict] = None
        self.accel_left: Optional[dict] = None
        self.accel_right: Optional[dict] = None
        self.stage_start_time: Optional[float] = None
        self.stage_delay_end: Optional[float] = None
        self.pattern_samples: list[Tuple[bool, bool, bool, bool, bool]] = []
        self.completed = False
        self.stages = [
            {"key": "two_hand_left", "prompt": "Two hands: steer LEFT", "kind": "two_hand"},
            {"key": "one_hand_left", "prompt": "One hand: steer LEFT", "kind": "one_hand"},
            {"key": "two_hand_right", "prompt": "Two hands: steer RIGHT", "kind": "two_hand"},
            {"key": "one_hand_right", "prompt": "One hand: steer RIGHT", "kind": "one_hand"},
            {"key": "brake_left", "prompt": "Left hand: BRAKE (ex. closed fist)", "kind": "gesture_brake", "hand": "Left"},
            {"key": "brake_right", "prompt": "Right hand: BRAKE (ex. closed fist)", "kind": "gesture_brake", "hand": "Right"},
            {"key": "accel_left", "prompt": "Left hand: ACCEL (ex. open palm )", "kind": "gesture_accel", "hand": "Left"},
            {"key": "accel_right", "prompt": "Right hand: ACCEL (ex. open palm)", "kind": "gesture_accel", "hand": "Right"},
        ]
        if not self.waiting_for_start:
            self.start()

    def start(self) -> None:
        if self.start_time is not None:
            return
        self.waiting_for_start = False
        self.start_time = time.time()
        self.countdown_end = self.start_time + self.countdown_seconds
        self.go_end = self.countdown_end + 0.5
        self.stage_delay_end = self.start_time

    def reset_samples(self, now: Optional[float] = None) -> None:
        self.samples = []
        self.pattern_samples = []
        self.stage_start_time = None
        now_time = now if now is not None else time.time()
        self.stage_delay_end = now_time + self.stage_delay_seconds

    def _step_prompt(self) -> str:
        if self.step_index >= len(self.stages):
            return "Calibration complete"
        return self.stages[self.step_index]["prompt"]

    def _maybe_start_stage(self, now: float) -> None:
        if self.stage_start_time is None:
            self.stage_start_time = now

    def _stage_elapsed(self, now: float) -> bool:
        return self.stage_start_time is not None and (now - self.stage_start_time) >= self.stage_duration

    def _delay_countdown_text(self, now: float) -> Optional[str]:
        if self.stage_delay_end is None:
            return None
        remaining = self.stage_delay_end - now
        if remaining <= 0:
            return None
        if remaining <= 1.0:
            return "GO"
        display = int(math.ceil(remaining)) - 1
        display = max(1, min(3, display))
        return str(display)

    def update(self, left, right, cfg: dict) -> Tuple[str, Optional[str]]:
        now = time.time()
        if self.waiting_for_start:
            return self._step_prompt(), "Press SPACE"
        if self.countdown_end is not None and now < self.countdown_end:
            remaining = max(1, int(self.countdown_end - now) + 1)
            return self._step_prompt(), str(remaining)
        if self.go_end is not None and now < self.go_end:
            return self._step_prompt(), "GO"

        delay_text = self._delay_countdown_text(now)
        if delay_text is not None:
            return self._step_prompt(), delay_text

        margin = cfg["finger_extended_margin"]
        radial_margin = get_cfg(cfg, "finger_extended_radial_margin", 0.03)
        action_margin = get_cfg(cfg, "action_finger_extended_margin", margin)
        action_radial_margin = get_cfg(cfg, "action_finger_extended_radial_margin", radial_margin)

        if self.step_index >= len(self.stages):
            self.completed = True
            return "Calibration complete", None

        stage = self.stages[self.step_index]
        kind = stage["kind"]

        if kind == "two_hand":
            if left and right:
                self._maybe_start_stage(now)
                self.samples.append(steering_angle_two_hands(left, right))
                if self._stage_elapsed(now):
                    self.data[f"{stage['key']}_deg"] = sum(self.samples) / len(self.samples)
                    self.reset_samples(now)
                    self.step_index += 1
                    if self.step_index >= len(self.stages):
                        self.completed = True
            return self._step_prompt(), None

        if kind == "one_hand":
            hand, _label = _single_hand(left, right)
            if hand is not None:
                self._maybe_start_stage(now)
                self.samples.append(steering_angle_one_hand(hand))
                if self._stage_elapsed(now):
                    self.data[f"{stage['key']}_deg"] = sum(self.samples) / len(self.samples)
                    self.reset_samples(now)
                    self.step_index += 1
                    if self.step_index >= len(self.stages):
                        self.completed = True
            return self._step_prompt(), None

        required_hand = stage.get("hand")
        if required_hand == "Left":
            hand = left
            label = "Left" if left is not None else None
        elif required_hand == "Right":
            hand = right
            label = "Right" if right is not None else None
        else:
            hand, label = _single_hand(left, right)

        if hand is None or label is None:
            return self._step_prompt(), None

        self._maybe_start_stage(now)
        if kind.startswith("gesture_"):
            pattern = hand_finger_pattern(hand, action_margin, action_radial_margin)
        else:
            pattern = hand_finger_pattern(hand, margin, radial_margin)
        self.pattern_samples.append(
            (pattern["thumb"], pattern["index"], pattern["middle"], pattern["ring"], pattern["pinky"])
        )
        if self._stage_elapsed(now):
            if self.pattern_samples:
                counts: dict[Tuple[bool, bool, bool, bool, bool], int] = {}
                for sample in self.pattern_samples:
                    counts[sample] = counts.get(sample, 0) + 1
                chosen = max(counts.items(), key=lambda item: item[1])[0]
                chosen_pattern = {
                    "thumb": chosen[0],
                    "index": chosen[1],
                    "middle": chosen[2],
                    "ring": chosen[3],
                    "pinky": chosen[4],
                }
                if kind == "gesture_brake":
                    if label == "Left":
                        self.brake_left = chosen_pattern
                    else:
                        self.brake_right = chosen_pattern
                else:
                    if label == "Left":
                        self.accel_left = chosen_pattern
                    else:
                        self.accel_right = chosen_pattern
            self.reset_samples(now)
            self.step_index += 1
            if self.step_index >= len(self.stages):
                self.completed = True
        return self._step_prompt(), None

    def build_calibration(self) -> CalibrationData:
        two_left = float(self.data["two_hand_left_deg"])
        two_right = float(self.data["two_hand_right_deg"])
        one_left = float(self.data["one_hand_left_deg"])
        one_right = float(self.data["one_hand_right_deg"])

        two_neutral, two_max = compute_neutral_and_max(two_left, two_right)
        one_neutral, one_max = compute_neutral_and_max(one_left, one_right)

        return CalibrationData(
            version=1,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            two_hand_left_deg=two_left,
            two_hand_right_deg=two_right,
            two_hand_neutral_deg=two_neutral,
            two_hand_max_steer_deg=two_max,
            one_hand_left_deg=one_left,
            one_hand_right_deg=one_right,
            one_hand_neutral_deg=one_neutral,
            one_hand_max_steer_deg=one_max,
            brake_left=self.brake_left,
            brake_right=self.brake_right,
            accel_left=self.accel_left,
            accel_right=self.accel_right,
        )
