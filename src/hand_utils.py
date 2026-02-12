from __future__ import annotations

import math
from typing import Optional, Tuple


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


def pattern_matches(current: dict, target: Optional[dict], max_mismatches: int = 1) -> bool:
    if target is None:
        return False
    mismatches = 0
    for key in ("thumb", "index", "middle", "ring", "pinky"):
        if current.get(key) != target.get(key):
            mismatches += 1
            if mismatches > max_mismatches:
                return False
    return True


def hand_finger_pattern(hand_lms, margin: float, radial_margin: float) -> dict:
    fs = get_finger_states(hand_lms, margin, radial_margin)
    thumb = is_finger_extended(hand_lms, 4, 3, margin, radial_margin)
    return {
        "thumb": thumb,
        "index": fs["index"],
        "middle": fs["middle"],
        "ring": fs["ring"],
        "pinky": fs["pinky"],
    }


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
