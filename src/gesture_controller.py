from __future__ import annotations

import math
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
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


class ControllerState:
    def __init__(self, smooth_n: int, action_smooth_n: int):
        self.neutral_deg: Optional[float] = None
        self.steer_hist = deque(maxlen=smooth_n)
        self.accel_hist = deque(maxlen=action_smooth_n)
        self.brake_hist = deque(maxlen=action_smooth_n)
        self.last = ControlOutput(steer=0.0, accel=False, brake=False)
        self.last_calib_time = 0.0

    def set_neutral(self, angle_deg: float) -> None:
        self.neutral_deg = angle_deg
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


class PygameOverlay:
    def __init__(self, window_name: str):
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.font.init()
        self.window_name = window_name
        self.surface = None
        self.font = pygame.font.SysFont("Arial", 20)

    def ensure_surface(self, width: int, height: int):
        if self.surface is None:
            self.surface = self.pygame.display.set_mode((width, height))
            self.pygame.display.set_caption(self.window_name)

    def process_events(self):
        quit_requested = False
        calibrate = False
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                quit_requested = True
            if event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_q:
                    quit_requested = True
                if event.key == self.pygame.K_c:
                    calibrate = True
        return quit_requested, calibrate

    def update(self, frame_bgr, lines):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_surface = self.pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
        self.ensure_surface(frame_surface.get_width(), frame_surface.get_height())
        self.surface.blit(frame_surface, (0, 0))
        y = 10
        for line in lines:
            text_surface = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_surface, (10, y))
            y += 24
        self.pygame.display.flip()

    def close(self):
        self.pygame.display.quit()
        self.pygame.font.quit()
        self.pygame.quit()


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
    radial_margin = get_cfg(cfg, "finger_extended_radial_margin", 0.03)
    brake = detect_brake(left, right, margin, radial_margin, cfg["brake_preferred_hand"])
    accel = detect_accel(left, right, margin, radial_margin, cfg["accel_preferred_hand"])

    if cfg["brake_priority_over_accel"] and brake:
        accel = False
    elif accel:
        brake = False

    brake = state.smooth_action(brake, state.brake_hist)
    accel = state.smooth_action(accel, state.accel_hist)

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


def run_camera_loop(
    config_path: str | None = None,
    controls_config_path: str | None = None,
    camera_index: int | None = None,
    show_ui: bool | None = None,
    use_pygame_ui: bool | None = None,
    draw_landmarks: bool | None = None,
    mirror_input: bool | None = None,
    backend: str | None = None,
    show_fps: bool | None = None,
) -> None:
    cfg = load_config(config_path)
    controls_cfg = load_controls_config(controls_config_path)
    state = ControllerState(
        cfg["steer_smoothing_frames"],
        get_cfg(cfg, "action_smoothing_frames", 1),
    )

    camera_index = camera_index if camera_index is not None else get_cfg(cfg, "camera_index", 0)
    show_ui = show_ui if show_ui is not None else get_cfg(cfg, "show_ui", True)
    use_pygame_ui = use_pygame_ui if use_pygame_ui is not None else get_cfg(cfg, "use_pygame_ui", False)
    draw_landmarks = draw_landmarks if draw_landmarks is not None else get_cfg(cfg, "draw_landmarks", True)
    mirror_input = mirror_input if mirror_input is not None else get_cfg(cfg, "mirror_input", True)
    show_fps = show_fps if show_fps is not None else get_cfg(cfg, "show_fps", True)
    window_name = get_cfg(cfg, "window_name", "MediaPipe Gesture Controller")
    backend_name = backend if backend is not None else get_cfg(cfg, "controller_backend", None)
    controller = create_control_backend(controls_cfg, backend_name)
    overlay = PygameOverlay(window_name) if show_ui and use_pygame_ui else None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    detector = build_hand_landmarker(cfg)
    try:
        print("Controls:")
        print("  c = calibrate neutral steering (center)")
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

            output, using, raw_angle = compute_controls(left, right, cfg, state)
            controller.update(output.steer, output.accel, output.brake)

            if show_ui and overlay is None:
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

                if show_fps:
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time) if prev_time else 0
                    prev_time = curr_time
                    cv2.putText(
                        frame,
                        f"FPS: {int(fps)}",
                        (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c") and raw_angle is not None:
                    state.set_neutral(raw_angle)
                    state.last_calib_time = time.time()
            elif show_ui and overlay is not None:
                text_lines = [
                    f"Steer: {output.steer:+.2f}  (src: {using})",
                    f"Accel: {output.accel}  Brake: {output.brake}",
                    f"Neutral deg: {state.neutral_deg:.1f}" if state.neutral_deg is not None else "Neutral: None",
                ]
                if show_fps:
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time) if prev_time else 0
                    prev_time = curr_time
                    text_lines.append(f"FPS: {int(fps)}")
                overlay.update(frame, text_lines)
                quit_requested, calibrate = overlay.process_events()
                if calibrate and raw_angle is not None:
                    state.set_neutral(raw_angle)
                    state.last_calib_time = time.time()
                if quit_requested:
                    show_ui = False
                    overlay.close()
                    overlay = None
            else:
                if raw_angle is not None and state.neutral_deg is None:
                    state.set_neutral(raw_angle)
    finally:
        detector.close()
        cap.release()
        controller.close()
        if overlay is not None:
            overlay.close()
        if show_ui and overlay is None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_loop()
