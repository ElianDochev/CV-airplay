from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import yaml

from .config_utils import get_cfg
from .hand_utils import hand_finger_pattern, is_thumb_up, normalize_angle_deg, steering_angle_two_hands


@dataclass
class CalibrationData:
    version: int
    created_at: str
    two_hand_left_deg: float
    two_hand_right_deg: float
    two_hand_neutral_deg: float
    two_hand_max_steer_deg: float
    brake_neutral_left: Optional[dict]
    brake_neutral_right: Optional[dict]
    brake_left_left: Optional[dict]
    brake_left_right: Optional[dict]
    brake_right_left: Optional[dict]
    brake_right_right: Optional[dict]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "two_hand_left_deg": self.two_hand_left_deg,
            "two_hand_right_deg": self.two_hand_right_deg,
            "two_hand_neutral_deg": self.two_hand_neutral_deg,
            "two_hand_max_steer_deg": self.two_hand_max_steer_deg,
            "brake_neutral_left": self.brake_neutral_left,
            "brake_neutral_right": self.brake_neutral_right,
            "brake_left_left": self.brake_left_left,
            "brake_left_right": self.brake_left_right,
            "brake_right_left": self.brake_right_left,
            "brake_right_right": self.brake_right_right,
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
        version = int(data.get("version", 1))
    except (TypeError, ValueError):
        return None

    if version < 2:
        return None

    try:
        return CalibrationData(
            version=version,
            created_at=str(data.get("created_at", "")),
            two_hand_left_deg=float(data["two_hand_left_deg"]),
            two_hand_right_deg=float(data["two_hand_right_deg"]),
            two_hand_neutral_deg=float(data["two_hand_neutral_deg"]),
            two_hand_max_steer_deg=float(data["two_hand_max_steer_deg"]),
            brake_neutral_left=data.get("brake_neutral_left") if isinstance(data.get("brake_neutral_left"), dict) else None,
            brake_neutral_right=data.get("brake_neutral_right") if isinstance(data.get("brake_neutral_right"), dict) else None,
            brake_left_left=data.get("brake_left_left") if isinstance(data.get("brake_left_left"), dict) else None,
            brake_left_right=data.get("brake_left_right") if isinstance(data.get("brake_left_right"), dict) else None,
            brake_right_left=data.get("brake_right_left") if isinstance(data.get("brake_right_left"), dict) else None,
            brake_right_right=data.get("brake_right_right") if isinstance(data.get("brake_right_right"), dict) else None,
        )
    except (KeyError, TypeError, ValueError):
        return None


def save_calibration(path: Path, calibration: CalibrationData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(calibration.to_dict(), handle, sort_keys=False)


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
        self.brake_neutral_left: Optional[dict] = None
        self.brake_neutral_right: Optional[dict] = None
        self.brake_left_left: Optional[dict] = None
        self.brake_left_right: Optional[dict] = None
        self.brake_right_left: Optional[dict] = None
        self.brake_right_right: Optional[dict] = None
        self.stage_start_time: Optional[float] = None
        self.stage_delay_end: Optional[float] = None
        self.pattern_samples: dict[str, list[Tuple[bool, bool, bool, bool, bool]]] = {"Left": [], "Right": []}
        self.completed = False
        self.stages = [
            {"key": "two_hand_neutral", "prompt": "Two hands: NEUTRAL (thumbs up)", "kind": "two_hand_neutral"},
            {"key": "two_hand_left", "prompt": "Two hands: steer LEFT (rotate wrists)", "kind": "two_hand"},
            {"key": "two_hand_right", "prompt": "Two hands: steer RIGHT (rotate wrists)", "kind": "two_hand"},
            {"key": "brake_neutral", "prompt": "Two hands: BRAKE in NEUTRAL (thumb up + fingers sideways)", "kind": "brake"},
            {"key": "brake_left", "prompt": "Two hands: BRAKE while LEFT (thumb up + fingers sideways)", "kind": "brake"},
            {"key": "brake_right", "prompt": "Two hands: BRAKE while RIGHT (thumb up + fingers sideways)", "kind": "brake"},
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
        self.pattern_samples = {"Left": [], "Right": []}
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

        if kind == "two_hand_neutral":
            if left and right:
                thumbs_up = is_thumb_up(left, margin, radial_margin) and is_thumb_up(right, margin, radial_margin)
                warning = None if thumbs_up else "WARNING: Show two thumbs up"
                self._maybe_start_stage(now)
                if warning is None:
                    self.samples.append(steering_angle_two_hands(left, right))
                if self._stage_elapsed(now):
                    if self.samples:
                        self.data[f"{stage['key']}_deg"] = sum(self.samples) / len(self.samples)
                        self.reset_samples(now)
                        self.step_index += 1
                        if self.step_index >= len(self.stages):
                            self.completed = True
                    else:
                        self.reset_samples(now)
                return self._step_prompt(), warning
            return self._step_prompt(), None

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

        if kind == "brake":
            if left and right:
                self._maybe_start_stage(now)
                left_pattern = hand_finger_pattern(left, action_margin, action_radial_margin)
                right_pattern = hand_finger_pattern(right, action_margin, action_radial_margin)
                self.pattern_samples["Left"].append(
                    (
                        left_pattern["thumb"],
                        left_pattern["index"],
                        left_pattern["middle"],
                        left_pattern["ring"],
                        left_pattern["pinky"],
                    )
                )
                self.pattern_samples["Right"].append(
                    (
                        right_pattern["thumb"],
                        right_pattern["index"],
                        right_pattern["middle"],
                        right_pattern["ring"],
                        right_pattern["pinky"],
                    )
                )
                if self._stage_elapsed(now):
                    if self.pattern_samples["Left"] and self.pattern_samples["Right"]:
                        for label in ("Left", "Right"):
                            counts: dict[Tuple[bool, bool, bool, bool, bool], int] = {}
                            for sample in self.pattern_samples[label]:
                                counts[sample] = counts.get(sample, 0) + 1
                            chosen = max(counts.items(), key=lambda item: item[1])[0]
                            chosen_pattern = {
                                "thumb": chosen[0],
                                "index": chosen[1],
                                "middle": chosen[2],
                                "ring": chosen[3],
                                "pinky": chosen[4],
                            }
                            if stage["key"] == "brake_neutral":
                                if label == "Left":
                                    self.brake_neutral_left = chosen_pattern
                                else:
                                    self.brake_neutral_right = chosen_pattern
                            elif stage["key"] == "brake_left":
                                if label == "Left":
                                    self.brake_left_left = chosen_pattern
                                else:
                                    self.brake_left_right = chosen_pattern
                            else:
                                if label == "Left":
                                    self.brake_right_left = chosen_pattern
                                else:
                                    self.brake_right_right = chosen_pattern
                    self.reset_samples(now)
                    self.step_index += 1
                    if self.step_index >= len(self.stages):
                        self.completed = True
        return self._step_prompt(), None

    def build_calibration(self) -> CalibrationData:
        two_neutral = float(self.data["two_hand_neutral_deg"])
        two_left = float(self.data["two_hand_left_deg"])
        two_right = float(self.data["two_hand_right_deg"])

        two_max = max(
            abs(normalize_angle_deg(two_left - two_neutral)),
            abs(normalize_angle_deg(two_right - two_neutral)),
        )

        return CalibrationData(
            version=2,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            two_hand_left_deg=two_left,
            two_hand_right_deg=two_right,
            two_hand_neutral_deg=two_neutral,
            two_hand_max_steer_deg=two_max,
            brake_neutral_left=self.brake_neutral_left,
            brake_neutral_right=self.brake_neutral_right,
            brake_left_left=self.brake_left_left,
            brake_left_right=self.brake_left_right,
            brake_right_left=self.brake_right_left,
            brake_right_right=self.brake_right_right,
        )
