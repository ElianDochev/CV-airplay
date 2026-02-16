from __future__ import annotations

from typing import Optional, Tuple

from .calibration import CalibrationData
from .config_utils import get_cfg
from .control_types import ControlOutput, ControllerState
from .hand_utils import (
    clamp,
    fingers_point_toward_center,
    hand_finger_pattern,
    is_thumb_up,
    normalize_angle_deg,
    palm_sideways,
    pattern_matches,
    steering_angle_two_hands,
    thumb_extended,
)


def compute_controls(
    left,
    right,
    cfg: dict,
    state: ControllerState,
    calibration: Optional[CalibrationData],
) -> Tuple[ControlOutput, str, Optional[float]]:
    margin = cfg["finger_extended_margin"]
    radial_margin = get_cfg(cfg, "finger_extended_radial_margin", 0.03)
    action_margin = get_cfg(cfg, "action_finger_extended_margin", margin)
    action_radial_margin = get_cfg(cfg, "action_finger_extended_radial_margin", radial_margin)
    palm_sideways_max_abs_z = float(get_cfg(cfg, "brake_palm_sideways_max_abs_z", 0.4))
    fingers_toward_min_abs_x = float(get_cfg(cfg, "brake_fingers_toward_min_abs_x", 0.35))
    fingers_toward_min_extended_ratio = float(get_cfg(cfg, "brake_fingers_extended_ratio", 0.69))
    brake = False
    if calibration:
        if left:
            left_pattern = hand_finger_pattern(left, action_margin, action_radial_margin)
            left_brake = any(
                pattern_matches(left_pattern, pattern, max_mismatches=1)
                for pattern in (
                    calibration.brake_neutral_left,
                    calibration.brake_left_left,
                    calibration.brake_right_left,
                )
                if pattern is not None
            )
            brake = brake or (
                left_brake
                and thumb_extended(left, action_margin, action_radial_margin)
                and palm_sideways(left, palm_sideways_max_abs_z)
                and fingers_point_toward_center(
                    left,
                    "right",
                    fingers_toward_min_abs_x,
                    action_margin,
                    action_radial_margin,
                    fingers_toward_min_extended_ratio,
                )
            )
        if right:
            right_pattern = hand_finger_pattern(right, action_margin, action_radial_margin)
            right_brake = any(
                pattern_matches(right_pattern, pattern, max_mismatches=1)
                for pattern in (
                    calibration.brake_neutral_right,
                    calibration.brake_left_right,
                    calibration.brake_right_right,
                )
                if pattern is not None
            )
            brake = brake or (
                right_brake
                and thumb_extended(right, action_margin, action_radial_margin)
                and palm_sideways(right, palm_sideways_max_abs_z)
                and fingers_point_toward_center(
                    right,
                    "left",
                    fingers_toward_min_abs_x,
                    action_margin,
                    action_radial_margin,
                    fingers_toward_min_extended_ratio,
                )
            )

    raw_angle = None
    using = "none"
    if left and right:
        raw_angle = steering_angle_two_hands(left, right)
        using = "2-hand"

    brake = state.smooth_action(brake, state.brake_hist)
    accel = bool(left and right) and not brake

    steer = 0.0
    if raw_angle is not None:
        neutral = calibration.two_hand_neutral_deg if calibration else None
        max_steer = calibration.two_hand_max_steer_deg if calibration else None
        if neutral is None:
            if state.neutral_by_mode["2-hand"] is None:
                state.set_neutral(raw_angle, "2-hand")
            neutral = state.neutral_by_mode["2-hand"]
        if max_steer is None or neutral is None:
            steer = state.smooth_steer(0.0)
            output = ControlOutput(steer=steer, accel=accel, brake=brake)
            state.last = output
            return output, using, raw_angle

        if not max_steer:
            steer = state.smooth_steer(0.0)
        else:
            delta = normalize_angle_deg(raw_angle - float(neutral))
            delta = clamp(delta, -max_steer, max_steer)
            steer = state.smooth_steer(delta / max_steer)
    else:
        steer = state.smooth_steer(0.0)

    output = ControlOutput(steer=steer, accel=accel, brake=brake)
    state.last = output

    return output, using, raw_angle
