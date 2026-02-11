from __future__ import annotations

import json
import math
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .control.backend import create_control_backend
from .ulits import load_config, load_controls_config


DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)


@dataclass
class ControlOutput:
    steer: float
    accel: bool
    brake: bool


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
    brake_hand: Optional[str]
    accel_hand: Optional[str]

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
            "brake_hand": self.brake_hand,
            "accel_hand": self.accel_hand,
        }


class ControllerState:
    def __init__(self, smooth_n: int, action_smooth_n: int):
        self.neutral_by_mode: dict[str, Optional[float]] = {"2-hand": None, "1-hand": None}
        self.steer_hist = deque(maxlen=smooth_n)
        self.accel_hist = deque(maxlen=action_smooth_n)
        self.brake_hist = deque(maxlen=action_smooth_n)
        self.last = ControlOutput(steer=0.0, accel=False, brake=False)
        self.last_calib_time = 0.0

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


def get_cfg(cfg: dict, key: str, default):
    return cfg[key] if key in cfg else default


def ensure_model(model_path: Path, model_url: str) -> None:
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(model_url, model_path)


def build_hand_landmarker(cfg: dict) -> vision.HandLandmarker:
    model_path = Path(get_cfg(cfg, "hand_landmarker_model_path", "models/hand_landmarker.task"))
    model_url = get_cfg(cfg, "hand_landmarker_model_url", DEFAULT_MODEL_URL)
    ensure_model(model_path, model_url)

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=cfg["max_num_hands"],
        min_hand_detection_confidence=cfg["min_detection_confidence"],
        min_hand_presence_confidence=cfg["min_detection_confidence"],
        min_tracking_confidence=cfg["min_tracking_confidence"],
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


def hand_label(handedness_entry) -> str:
    if not handedness_entry:
        return "Unknown"
    classification = handedness_entry[0]
    return getattr(classification, "category_name", getattr(classification, "label", "Unknown"))


def render_landmarks(frame, hand_lms, color=(0, 255, 0)) -> None:
    landmarks = hand_lms.landmark if hasattr(hand_lms, "landmark") else hand_lms
    for lm in landmarks:
        x = int(lm.x * frame.shape[1])
        y = int(lm.y * frame.shape[0])
        cv2.circle(frame, (x, y), 3, color, -1)


def calibration_root() -> Path:
    return Path(__file__).resolve().parents[1] / "calibration"


def calibration_path() -> Path:
    return calibration_root() / "calibration.json"


def load_calibration(path: Path) -> Optional[CalibrationData]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

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
            brake_hand=data.get("brake_hand"),
            accel_hand=data.get("accel_hand"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def save_calibration(path: Path, calibration: CalibrationData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(calibration.to_dict(), handle, indent=2)


def _single_hand(left, right) -> Tuple[Optional[object], Optional[str]]:
    if left is not None and right is None:
        return left, "Left"
    if right is not None and left is None:
        return right, "Right"
    return None, None


class CalibrationSession:
    def __init__(self, show_ui: bool, countdown_seconds: int = 3, sample_frames: int = 18) -> None:
        self.show_ui = show_ui
        self.countdown_seconds = countdown_seconds
        self.sample_frames = sample_frames
        self.start_time = time.time()
        self.countdown_end = self.start_time + (countdown_seconds if show_ui else 0)
        self.step_index = 0
        self.samples: list[float] = []
        self.data: dict[str, float | str] = {}
        self.one_hand_label: Optional[str] = None
        self.brake_hand: Optional[str] = None
        self.accel_hand: Optional[str] = None
        self.completed = False

    def reset_samples(self) -> None:
        self.samples = []

    def _step_prompt(self) -> str:
        prompts = [
            "Two hands: steer LEFT",
            "Two hands: steer RIGHT",
            "One hand: steer LEFT",
            "One hand: steer RIGHT",
            "One hand: BRAKE (open palm)",
            "One hand: ACCEL (point index)",
        ]
        return prompts[self.step_index] if self.step_index < len(prompts) else "Calibration complete"

    def update(self, left, right, cfg: dict) -> str:
        now = time.time()
        if now < self.countdown_end:
            remaining = int(math.ceil(self.countdown_end - now))
            return f"Calibration starts in {remaining}..."

        margin = cfg["finger_extended_margin"]
        radial_margin = get_cfg(cfg, "finger_extended_radial_margin", 0.03)

        if self.step_index == 0:
            if left and right:
                self.samples.append(steering_angle_two_hands(left, right))
                if len(self.samples) >= self.sample_frames:
                    self.data["two_hand_left_deg"] = sum(self.samples) / len(self.samples)
                    self.reset_samples()
                    self.step_index += 1
            return self._step_prompt()

        if self.step_index == 1:
            if left and right:
                self.samples.append(steering_angle_two_hands(left, right))
                if len(self.samples) >= self.sample_frames:
                    self.data["two_hand_right_deg"] = sum(self.samples) / len(self.samples)
                    self.reset_samples()
                    self.step_index += 1
            return self._step_prompt()

        if self.step_index == 2:
            hand, label = _single_hand(left, right)
            if hand is not None:
                if self.one_hand_label is None:
                    self.one_hand_label = label
                if label == self.one_hand_label:
                    self.samples.append(steering_angle_one_hand(hand))
                    if len(self.samples) >= self.sample_frames:
                        self.data["one_hand_left_deg"] = sum(self.samples) / len(self.samples)
                        self.reset_samples()
                        self.step_index += 1
            return self._step_prompt()

        if self.step_index == 3:
            hand, label = _single_hand(left, right)
            if hand is not None and label == self.one_hand_label:
                self.samples.append(steering_angle_one_hand(hand))
                if len(self.samples) >= self.sample_frames:
                    self.data["one_hand_right_deg"] = sum(self.samples) / len(self.samples)
                    self.reset_samples()
                    self.step_index += 1
            return self._step_prompt()

        if self.step_index == 4:
            hand, label = _single_hand(left, right)
            if hand is not None and is_open_palm(hand, margin, radial_margin):
                self.brake_hand = label
                self.step_index += 1
            return self._step_prompt()

        if self.step_index == 5:
            hand, label = _single_hand(left, right)
            if hand is not None and is_pointing_index(hand, margin, radial_margin):
                self.accel_hand = label
                self.step_index += 1
                if self.step_index >= 6:
                    self.completed = True
            return self._step_prompt()

        self.completed = True
        return "Calibration complete"

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
            brake_hand=self.brake_hand,
            accel_hand=self.accel_hand,
        )


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


def average_angle_deg(a_deg: float, b_deg: float) -> float:
    a_rad = math.radians(a_deg)
    b_rad = math.radians(b_deg)
    x = math.cos(a_rad) + math.cos(b_rad)
    y = math.sin(a_rad) + math.sin(b_rad)
    if x == 0.0 and y == 0.0:
        return normalize_angle_deg(a_deg)
    return normalize_angle_deg(rad2deg(math.atan2(y, x)))


def compute_neutral_and_max(left_deg: float, right_deg: float) -> Tuple[float, float]:
    neutral = average_angle_deg(left_deg, right_deg)
    max_delta = max(
        abs(normalize_angle_deg(left_deg - neutral)),
        abs(normalize_angle_deg(right_deg - neutral)),
    )
    return neutral, max_delta


def _get_landmark(hand_lms, idx: int):
    if hasattr(hand_lms, "landmark"):
        return hand_lms.landmark[idx]
    return hand_lms[idx]


def lm_xyz(hand_lms, idx: int) -> Tuple[float, float, float]:
    lm = _get_landmark(hand_lms, idx)
    return (lm.x, lm.y, lm.z)


def distance_3d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def lm_xy(hand_lms, idx: int) -> Tuple[float, float]:
    lm = _get_landmark(hand_lms, idx)
    return (lm.x, lm.y)


# -------------------------
# Finger state (extended / folded)
# -------------------------

def is_finger_extended(
    hand_lms,
    tip: int,
    pip: int,
    margin: float = 0.02,
    radial_margin: float = 0.03,
) -> bool:
    tip_y = _get_landmark(hand_lms, tip).y
    pip_y = _get_landmark(hand_lms, pip).y
    y_extended = (pip_y - tip_y) > margin

    wrist = lm_xyz(hand_lms, 0)
    tip_dist = distance_3d(lm_xyz(hand_lms, tip), wrist)
    pip_dist = distance_3d(lm_xyz(hand_lms, pip), wrist)
    radial_extended = (tip_dist - pip_dist) > radial_margin

    return y_extended or radial_extended


def get_finger_states(hand_lms, margin: float, radial_margin: float) -> dict:
    return {
        "index": is_finger_extended(hand_lms, 8, 6, margin, radial_margin),
        "middle": is_finger_extended(hand_lms, 12, 10, margin, radial_margin),
        "ring": is_finger_extended(hand_lms, 16, 14, margin, radial_margin),
        "pinky": is_finger_extended(hand_lms, 20, 18, margin, radial_margin),
    }


def is_thumb_up(hand_lms, margin: float, radial_margin: float) -> bool:
    thumb_extended = is_finger_extended(hand_lms, 4, 3, margin, radial_margin)
    fs = get_finger_states(hand_lms, margin, radial_margin)
    return thumb_extended and (not fs["index"]) and (not fs["middle"]) and (not fs["ring"]) and (not fs["pinky"])


def is_open_palm(hand_lms, margin: float, radial_margin: float) -> bool:
    fs = get_finger_states(hand_lms, margin, radial_margin)
    return fs["index"] and fs["middle"] and fs["ring"] and fs["pinky"]


def is_pointing_index(hand_lms, margin: float, radial_margin: float) -> bool:
    fs = get_finger_states(hand_lms, margin, radial_margin)
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


def detect_brake(left, right, margin: float, radial_margin: float, preferred: str) -> bool:
    hand = _select_hand(left, right, preferred)
    return bool(hand and is_open_palm(hand, margin, radial_margin))


def detect_accel(left, right, margin: float, radial_margin: float, preferred: str) -> bool:
    hand = _select_hand(left, right, preferred)
    return bool(hand and is_pointing_index(hand, margin, radial_margin))


def _calibration_mode(using: str) -> str:
    return "2-hand" if using == "2-hand" else "1-hand"


def _get_calibration_steer(calibration: Optional[CalibrationData], mode: str) -> Tuple[Optional[float], Optional[float]]:
    if calibration is None:
        return None, None
    if mode == "2-hand":
        return calibration.two_hand_neutral_deg, calibration.two_hand_max_steer_deg
    return calibration.one_hand_neutral_deg, calibration.one_hand_max_steer_deg


def compute_controls(
    left,
    right,
    cfg: dict,
    state: ControllerState,
    calibration: Optional[CalibrationData],
) -> Tuple[ControlOutput, str, Optional[float]]:
    raw_angle = None
    using = "none"

    if left and right:
        raw_angle = steering_angle_two_hands(left, right)
        using = "2-hand"
    elif left:
        raw_angle = steering_angle_one_hand(left)
        using = "L-1hand"
    elif right:
        raw_angle = steering_angle_one_hand(right)
        using = "R-1hand"

    margin = cfg["finger_extended_margin"]
    radial_margin = get_cfg(cfg, "finger_extended_radial_margin", 0.03)
    brake_preferred = calibration.brake_hand if calibration else None
    accel_preferred = calibration.accel_hand if calibration else None
    brake = False
    accel = False
    if brake_preferred:
        brake = detect_brake(left, right, margin, radial_margin, brake_preferred)
    if accel_preferred:
        accel = detect_accel(left, right, margin, radial_margin, accel_preferred)

    if brake and accel:
        accel = False

    brake = state.smooth_action(brake, state.brake_hist)
    accel = state.smooth_action(accel, state.accel_hist)

    if brake and accel:
        accel = False

    steer = 0.0
    if raw_angle is not None:
        mode = _calibration_mode(using)
        calib_neutral, calib_max = _get_calibration_steer(calibration, mode)
        neutral = calib_neutral
        max_steer = calib_max
        if neutral is None:
            if state.neutral_by_mode[mode] is None:
                state.set_neutral(raw_angle, mode)
            neutral = state.neutral_by_mode[mode]
        if max_steer is None or neutral is None:
            steer = state.smooth_steer(0.0)
            output = ControlOutput(steer=steer, accel=False, brake=False)
            state.last = output
            return output, using, raw_angle

        delta = normalize_angle_deg(raw_angle - float(neutral))
        delta = clamp(delta, -max_steer, max_steer)
        steer = delta / max_steer if max_steer else 0.0
        steer = state.smooth_steer(steer)
    else:
        steer = state.smooth_steer(0.0)

    output = ControlOutput(steer=steer, accel=accel, brake=brake)
    state.last = output

    return output, using, raw_angle


def run_camera_loop(
    config_path: str | None = None,
    controls_config_path: str | None = None,
    camera_index: int | None = None,
    show_ui: bool | None = None,
    draw_landmarks: bool | None = None,
    mirror_input: bool | None = None,
    backend: str | None = None,
    show_fps: bool | None = None,
) -> None:
    cfg = load_config(config_path)
    controls_cfg = load_controls_config(controls_config_path)
    state = ControllerState(1, 1)

    camera_index = camera_index if camera_index is not None else get_cfg(cfg, "camera_index", 0)
    show_ui = show_ui if show_ui is not None else get_cfg(cfg, "show_ui", True)
    draw_landmarks = draw_landmarks if draw_landmarks is not None else get_cfg(cfg, "draw_landmarks", True)
    mirror_input = mirror_input if mirror_input is not None else get_cfg(cfg, "mirror_input", True)
    show_fps = show_fps if show_fps is not None else get_cfg(cfg, "show_fps", True)
    window_name = get_cfg(cfg, "window_name", "MediaPipe Gesture Controller")
    backend_name = backend if backend is not None else get_cfg(cfg, "controller_backend", None)
    controller = create_control_backend(controls_cfg, backend_name)

    calibration_file = calibration_path()
    calibration = load_calibration(calibration_file)
    calibration_session: Optional[CalibrationSession] = None
    if calibration is None:
        calibration_session = CalibrationSession(show_ui=show_ui)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    detector = build_hand_landmarker(cfg)
    try:
        print("Controls:")
        print("  c = recalibrate (clears calibration and restarts)")
        print("  q = quit")

        prev_time = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mirror_input:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            res = detector.detect_for_video(mp_image, timestamp_ms)

            left = None
            right = None

            if res.hand_landmarks and res.handedness:
                for hand_lms, handed in zip(res.hand_landmarks, res.handedness):
                    label = hand_label(handed)
                    if label == "Left":
                        left = hand_lms
                    else:
                        right = hand_lms
                    if show_ui and draw_landmarks:
                        render_landmarks(frame, hand_lms)

            output = ControlOutput(steer=0.0, accel=False, brake=False)
            using = "none"
            raw_angle = None
            if calibration_session is None or calibration_session.completed:
                output, using, raw_angle = compute_controls(left, right, cfg, state, calibration)
                controller.update(output.steer, output.accel, output.brake)

            if raw_angle is not None and left and right:
                margin = cfg["finger_extended_margin"]
                radial_margin = get_cfg(cfg, "finger_extended_radial_margin", 0.03)
                if is_thumb_up(left, margin, radial_margin) and is_thumb_up(right, margin, radial_margin):
                    state.set_neutral(raw_angle, "2-hand")
                    state.last_calib_time = time.time()

            if show_ui:
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
                if calibration is not None:
                    cv2.putText(
                        frame,
                        f"Calib 2H max: {calibration.two_hand_max_steer_deg:.1f}  1H max: {calibration.one_hand_max_steer_deg:.1f}",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                else:
                    neutral = state.neutral_by_mode["2-hand"]
                    cv2.putText(
                        frame,
                        f"Neutral deg: {neutral:.1f}" if neutral is not None else "Neutral: None",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                if calibration_session is not None and not calibration_session.completed:
                    prompt = calibration_session.update(left, right, cfg)
                    cv2.putText(
                        frame,
                        "Calibration: controller disabled",
                        (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        prompt,
                        (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 255),
                        2,
                    )
                    if calibration_session.completed:
                        calibration = calibration_session.build_calibration()
                        save_calibration(calibration_file, calibration)
                        calibration_session = None

                if show_fps:
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time) if prev_time else 0
                    prev_time = curr_time
                    cv2.putText(
                        frame,
                        f"FPS: {int(fps)}",
                        (20, 205) if (calibration_session is not None and not calibration_session.completed) else (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    if calibration_file.exists():
                        calibration_file.unlink()
                    calibration = None
                    calibration_session = CalibrationSession(show_ui=show_ui)
                    state.neutral_by_mode = {"2-hand": None, "1-hand": None}
                    state.steer_hist.clear()
                    state.accel_hist.clear()
                    state.brake_hist.clear()
            else:
                if calibration_session is not None and not calibration_session.completed:
                    calibration_session.update(left, right, cfg)
                    if calibration_session.completed:
                        calibration = calibration_session.build_calibration()
                        save_calibration(calibration_file, calibration)
                        calibration_session = None
                elif raw_angle is not None and state.neutral_by_mode["2-hand"] is None:
                    state.set_neutral(raw_angle, "2-hand")
    finally:
        detector.close()
        cap.release()
        controller.close()
        if show_ui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_loop()
