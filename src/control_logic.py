from __future__ import annotations

from typing import Optional, Tuple

from .calibration import CalibrationData
from .config_utils import get_cfg
from .control_types import ControlOutput, ControllerState
from .hand_utils import (
    hand_finger_pattern,
    normalize_angle_deg,
    pattern_matches,
    steering_angle_one_hand,
)


def _angle_matches(angle_deg: float, target_deg: float, tolerance_deg: float) -> bool:
    return abs(normalize_angle_deg(angle_deg - target_deg)) <= tolerance_deg


def compute_controls(
    left,
    right,
    cfg: dict,
    state: ControllerState,
    calibration: Optional[CalibrationData],
) -> Tuple[ControlOutput, str]:
    margin = cfg["finger_extended_margin"]
    radial_margin = get_cfg(cfg, "finger_extended_radial_margin", 0.03)
    action_margin = get_cfg(cfg, "action_finger_extended_margin", margin)
    action_radial_margin = get_cfg(cfg, "action_finger_extended_radial_margin", radial_margin)
    brake = False
    accel = False
    left_action = False
    right_action = False
    if calibration:
        if calibration.brake_left and left:
            left_brake = pattern_matches(
                hand_finger_pattern(left, action_margin, action_radial_margin),
                calibration.brake_left,
                max_mismatches=0,
            )
            brake = brake or left_brake
            left_action = left_action or left_brake
        if calibration.brake_right and right:
            right_brake = pattern_matches(
                hand_finger_pattern(right, action_margin, action_radial_margin),
                calibration.brake_right,
                max_mismatches=0,
            )
            brake = brake or right_brake
            right_action = right_action or right_brake
        if calibration.accel_left and left:
            left_accel = pattern_matches(
                hand_finger_pattern(left, action_margin, action_radial_margin),
                calibration.accel_left,
                max_mismatches=0,
            )
            accel = accel or left_accel
            left_action = left_action or left_accel
        if calibration.accel_right and right:
            right_accel = pattern_matches(
                hand_finger_pattern(right, action_margin, action_radial_margin),
                calibration.accel_right,
                max_mismatches=0,
            )
            accel = accel or right_accel
            right_action = right_action or right_accel

    steer_left = False
    steer_right = False
    using = "neutral"
    tolerance = float(get_cfg(cfg, "steer_angle_tolerance_deg", 20.0))

    if calibration is not None:
        if left and not left_action:
            left_angle = steering_angle_one_hand(left)
            if _angle_matches(left_angle, calibration.steer_right_left_deg, tolerance):
                steer_right = True
        if right and not right_action:
            right_angle = steering_angle_one_hand(right)
            if _angle_matches(right_angle, calibration.steer_left_right_deg, tolerance):
                steer_left = True

    brake = state.smooth_action(brake, state.brake_hist)
    accel = state.smooth_action(accel, state.accel_hist)

    if brake and accel:
        accel = False

    steer = 0.0
    if steer_left != steer_right:
        steer = -1.0 if steer_left else 1.0
        using = "steer-left" if steer_left else "steer-right"
    steer = state.smooth_steer(steer)

    output = ControlOutput(steer=steer, accel=accel, brake=brake)
    state.last = output

    return output, using
