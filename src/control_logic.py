from __future__ import annotations

from typing import Optional, Tuple

from .calibration import CalibrationData
from .config_utils import get_cfg
from .control_types import ControlOutput, ControllerState
from .hand_utils import (
    clamp,
    hand_finger_pattern,
    normalize_angle_deg,
    pattern_matches,
    steering_angle_one_hand,
    steering_angle_two_hands,
)


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
    action_margin = get_cfg(cfg, "action_finger_extended_margin", margin)
    action_radial_margin = get_cfg(cfg, "action_finger_extended_radial_margin", radial_margin)
    brake = False
    accel = False
    if calibration:
        if calibration.brake_left and left:
            brake = brake or pattern_matches(
                hand_finger_pattern(left, action_margin, action_radial_margin),
                calibration.brake_left,
                max_mismatches=0,
            )
        if calibration.brake_right and right:
            brake = brake or pattern_matches(
                hand_finger_pattern(right, action_margin, action_radial_margin),
                calibration.brake_right,
                max_mismatches=0,
            )
        if calibration.accel_left and left:
            accel = accel or pattern_matches(
                hand_finger_pattern(left, action_margin, action_radial_margin),
                calibration.accel_left,
                max_mismatches=0,
            )
        if calibration.accel_right and right:
            accel = accel or pattern_matches(
                hand_finger_pattern(right, action_margin, action_radial_margin),
                calibration.accel_right,
                max_mismatches=0,
            )

    if brake and accel:
        brake = False
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
