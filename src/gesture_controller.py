from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp

from ulits import load_config


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


@dataclass
class ControlOutput:
    steer: float
    accel: bool
    brake: bool


class ControllerState:
    def __init__(self, smooth_n: int):
        self.neutral_deg: Optional[float] = None
        self.steer_hist = deque(maxlen=smooth_n)
        self.last = ControlOutput(steer=0.0, accel=False, brake=False)
        self.last_calib_time = 0.0

    def set_neutral(self, angle_deg: float) -> None:
        self.neutral_deg = angle_deg
        self.steer_hist.clear()

    def smooth_steer(self, steer: float) -> float:
        self.steer_hist.append(steer)
        return sum(self.steer_hist) / len(self.steer_hist)


# -------------------------
# Utility / math
# -------------------------

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def rad2deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def angle_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Angle from p1->p2 in radians in image coordinate space."""
    return math.atan2((p2[1] - p1[1]), (p2[0] - p1[0]))


def normalize_angle_deg(angle_deg: float) -> float:
    """Normalize to [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def lm_xy(hand_lms, idx: int) -> Tuple[float, float]:
    lm = hand_lms.landmark[idx]
    return (lm.x, lm.y)


# -------------------------
# Finger state (extended / folded)
# -------------------------

def is_finger_extended(hand_lms, tip: int, pip: int, margin: float = 0.02) -> bool:
    tip_y = hand_lms.landmark[tip].y
    pip_y = hand_lms.landmark[pip].y
    return (pip_y - tip_y) > margin


def get_finger_states(hand_lms, margin: float) -> dict:
    return {
        "index": is_finger_extended(hand_lms, 8, 6, margin),
        "middle": is_finger_extended(hand_lms, 12, 10, margin),
        "ring": is_finger_extended(hand_lms, 16, 14, margin),
        "pinky": is_finger_extended(hand_lms, 20, 18, margin),
    }


def is_open_palm(hand_lms, margin: float) -> bool:
    fs = get_finger_states(hand_lms, margin)
    return fs["index"] and fs["middle"] and fs["ring"] and fs["pinky"]


def is_pointing_index(hand_lms, margin: float) -> bool:
    fs = get_finger_states(hand_lms, margin)
    return fs["index"] and (not fs["middle"]) and (not fs["ring"]) and (not fs["pinky"])


# -------------------------
# Steering angles
# -------------------------

def steering_angle_two_hands(left_lms, right_lms) -> float:
    L = lm_xy(left_lms, 0)
    R = lm_xy(right_lms, 0)
    return rad2deg(angle_2d(L, R))


def steering_angle_one_hand(hand_lms) -> float:
    wrist = lm_xy(hand_lms, 0)
    index_mcp = lm_xy(hand_lms, 5)
    pinky_mcp = lm_xy(hand_lms, 17)
    mid = ((index_mcp[0] + pinky_mcp[0]) / 2.0, (index_mcp[1] + pinky_mcp[1]) / 2.0)
    return rad2deg(angle_2d(wrist, mid))


def _select_hand(left, right, preferred: str):
    if preferred.lower() == "left":
        return left if left is not None else right
    return right if right is not None else left


def detect_brake(left, right, margin: float, preferred: str) -> bool:
    hand = _select_hand(left, right, preferred)
    return bool(hand and is_open_palm(hand, margin))


def detect_accel(left, right, margin: float, preferred: str) -> bool:
    hand = _select_hand(left, right, preferred)
    return bool(hand and is_pointing_index(hand, margin))


def compute_controls(left, right, cfg: dict, state: ControllerState) -> Tuple[ControlOutput, str, Optional[float]]:
    raw_angle = None
    using = "none"

    if cfg["use_two_hand_when_possible"] and left and right:
        raw_angle = steering_angle_two_hands(left, right)
        using = "2-hand"
    elif left:
        raw_angle = steering_angle_one_hand(left)
        using = "L-1hand"
    elif right:
        raw_angle = steering_angle_one_hand(right)
        using = "R-1hand"

    margin = cfg["finger_extended_margin"]
    brake = detect_brake(left, right, margin, cfg["brake_preferred_hand"])
    accel = detect_accel(left, right, margin, cfg["accel_preferred_hand"])

    if cfg["brake_priority_over_accel"] and brake:
        accel = False
    elif accel:
        brake = False

    steer = 0.0
    if raw_angle is not None:
        if state.neutral_deg is None:
            state.set_neutral(raw_angle)

        delta = normalize_angle_deg(raw_angle - state.neutral_deg)
        if abs(delta) < cfg["steer_deadzone_deg"]:
            delta = 0.0

        delta = clamp(delta, -cfg["max_steer_deg"], cfg["max_steer_deg"])
        steer = delta / cfg["max_steer_deg"]
        steer = state.smooth_steer(steer)
    else:
        steer = state.smooth_steer(0.0)

    output = ControlOutput(steer=steer, accel=accel, brake=brake)
    state.last = output

    return output, using, raw_angle


def run_camera_loop(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    state = ControllerState(cfg["steer_smoothing_frames"])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    with mp_hands.Hands(
        max_num_hands=cfg["max_num_hands"],
        model_complexity=cfg["model_complexity"],
        min_detection_confidence=cfg["min_detection_confidence"],
        min_tracking_confidence=cfg["min_tracking_confidence"],
    ) as hands:
        print("Controls:")
        print("  c = calibrate neutral steering (center)")
        print("  q = quit")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            left = None
            right = None

            if res.multi_hand_landmarks and res.multi_handedness:
                for hand_lms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = handed.classification[0].label
                    if label == "Left":
                        left = hand_lms
                    else:
                        right = hand_lms
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            output, using, raw_angle = compute_controls(left, right, cfg, state)

            cv2.putText(
                frame,
                f"Steer: {output.steer:+.2f}  (src: {using})",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Accel: {output.accel}  Brake: {output.brake}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Neutral deg: {state.neutral_deg:.1f}" if state.neutral_deg is not None else "Neutral: None",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("MediaPipe Gesture Controller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and raw_angle is not None:
                state.set_neutral(raw_angle)
                state.last_calib_time = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_loop()
