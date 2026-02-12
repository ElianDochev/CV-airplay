from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import mediapipe as mp

from .calibration import CalibrationSession, calibration_path, calibration_root, load_calibration, save_calibration
from .config_utils import get_cfg
from .control.backend import create_control_backend
from .control_logic import compute_controls
from .control_types import ControlOutput, ControllerState
from .hand_detection import build_hand_landmarker, hand_label, render_landmarks, resolve_hand_label
from .hand_utils import is_thumb_up
from .ulits import load_config, load_controls_config


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
    steer_smooth_n = get_cfg(cfg, "steer_smooth_n", 1)
    action_smooth_n = get_cfg(cfg, "action_smooth_n", 3)
    state = ControllerState(steer_smooth_n, action_smooth_n)

    camera_index = camera_index if camera_index is not None else get_cfg(cfg, "camera_index", 0)
    show_ui = show_ui if show_ui is not None else get_cfg(cfg, "show_ui", True)
    draw_landmarks = draw_landmarks if draw_landmarks is not None else get_cfg(cfg, "draw_landmarks", True)
    mirror_input = mirror_input if mirror_input is not None else get_cfg(cfg, "mirror_input", True)
    show_fps = show_fps if show_fps is not None else get_cfg(cfg, "show_fps", True)
    window_name = get_cfg(cfg, "window_name", "MediaPipe Gesture Controller")
    backend_name = backend if backend is not None else get_cfg(cfg, "controller_backend", None)
    controller = create_control_backend(controls_cfg, backend_name)
    controller_enabled = backend_name is not None and backend_name != "none"
    controller_active = not controller_enabled
    enable_countdown_end: Optional[float] = None
    enable_countdown_seconds = float(get_cfg(cfg, "controller_enable_countdown_seconds", 3))

    calib_dir = calibration_root()
    calib_dir.mkdir(parents=True, exist_ok=True)
    calibration_file = calibration_path()
    calibration = None
    if any(calib_dir.iterdir()):
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

            if controller_enabled and not controller_active and not show_ui and enable_countdown_end is None:
                enable_countdown_end = time.time() + enable_countdown_seconds

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
                    label = resolve_hand_label(hand_label(handed), mirror_input)
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
                if controller_active:
                    controller.update(output.steer, output.accel, output.brake)

            if raw_angle is not None and left and right:
                margin = cfg["finger_extended_margin"]
                radial_margin = get_cfg(cfg, "finger_extended_radial_margin", 0.03)
                if is_thumb_up(left, margin, radial_margin) and is_thumb_up(right, margin, radial_margin):
                    state.set_neutral(raw_angle, "2-hand")
                    state.last_calib_time = time.time()

            if calibration_session is not None and calibration_session.completed and calibration is None:
                calibration = calibration_session.build_calibration()
                save_calibration(calibration_file, calibration)
                calibration_session = None

            if show_ui:
                if calibration_session is not None and not calibration_session.completed:
                    prompt, countdown = calibration_session.update(left, right, cfg)
                    cv2.putText(
                        frame,
                        "Calibration",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 200, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        prompt,
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 200, 255),
                        2,
                    )
                    if countdown:
                        countdown_text = countdown
                        if countdown == "Press SPACE":
                            countdown_text = "Press SPACE to start"
                        cv2.putText(
                            frame,
                            f"Starts in: {countdown_text}",
                            (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 200, 255),
                            2,
                        )
                    if calibration_session.completed and calibration is None:
                        calibration = calibration_session.build_calibration()
                        save_calibration(calibration_file, calibration)
                        calibration_session = None
                else:
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

                    if controller_enabled and not controller_active:
                        countdown_text = "Press SPACE to enable"
                        if enable_countdown_end is not None:
                            remaining = max(0, int(math.ceil(enable_countdown_end - time.time())))
                            countdown_text = f"Enabling in: {remaining}"
                        cv2.putText(
                            frame,
                            f"Controller: DISABLED ({countdown_text})",
                            (20, 170),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 200, 255),
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
                if key == ord(" "):
                    if calibration_session is not None and calibration_session.waiting_for_start:
                        calibration_session.start()
                    elif controller_enabled and not controller_active and enable_countdown_end is None:
                        enable_countdown_end = time.time() + enable_countdown_seconds
                if key == ord("c"):
                    for existing in calib_dir.glob("calibration.*"):
                        existing.unlink()
                    calibration = None
                    calibration_session = CalibrationSession(show_ui=show_ui)
                    state.neutral_by_mode = {"2-hand": None, "1-hand": None}
                    state.steer_hist.clear()
                    state.accel_hist.clear()
                    state.brake_hist.clear()
            else:
                if calibration_session is not None and not calibration_session.completed:
                    calibration_session.update(left, right, cfg)
                    if calibration_session.completed and calibration is None:
                        calibration = calibration_session.build_calibration()
                        save_calibration(calibration_file, calibration)
                        calibration_session = None
                elif raw_angle is not None and state.neutral_by_mode["2-hand"] is None:
                    state.set_neutral(raw_angle, "2-hand")

            if enable_countdown_end is not None and time.time() >= enable_countdown_end:
                controller_active = True
                enable_countdown_end = None
    finally:
        detector.close()
        cap.release()
        controller.close()
        if show_ui:
            cv2.destroyAllWindows()
