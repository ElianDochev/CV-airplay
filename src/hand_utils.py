from __future__ import annotations

import math
from typing import Optional, Tuple

from .control.mapping import clamp
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


def thumb_extended(hand_lms, margin: float, radial_margin: float) -> bool:
    return is_finger_extended(hand_lms, 4, 3, margin, radial_margin)


def palm_sideways(hand_lms, max_abs_z: float = 0.4) -> bool:
    wrist = lm_xyz(hand_lms, 0)
    index_mcp = lm_xyz(hand_lms, 5)
    pinky_mcp = lm_xyz(hand_lms, 17)
    v1 = (index_mcp[0] - wrist[0], index_mcp[1] - wrist[1], index_mcp[2] - wrist[2])
    v2 = (pinky_mcp[0] - wrist[0], pinky_mcp[1] - wrist[1], pinky_mcp[2] - wrist[2])
    nx = (v1[1] * v2[2]) - (v1[2] * v2[1])
    ny = (v1[2] * v2[0]) - (v1[0] * v2[2])
    nz = (v1[0] * v2[1]) - (v1[1] * v2[0])
    norm = math.sqrt((nx * nx) + (ny * ny) + (nz * nz))
    if norm == 0.0:
        return False
    return abs(nz) / norm < max_abs_z


def fingers_point_toward_center(
    hand_lms,
    toward: str,
    min_abs_x: float = 0.35,
    margin: float = 0.02,
    radial_margin: float = 0.03,
    min_extended_ratio: float = 0.69,
) -> bool:
    wrist = lm_xyz(hand_lms, 0)
    tips = [lm_xyz(hand_lms, idx) for idx in (8, 12, 16, 20)]
    avg_tip = (
        sum(t[0] for t in tips) / len(tips),
        sum(t[1] for t in tips) / len(tips),
        sum(t[2] for t in tips) / len(tips),
    )
    vx = avg_tip[0] - wrist[0]
    vy = avg_tip[1] - wrist[1]
    vz = avg_tip[2] - wrist[2]
    norm = math.sqrt((vx * vx) + (vy * vy) + (vz * vz))
    if norm == 0.0:
        return False
    fs = get_finger_states(hand_lms, margin, radial_margin)
    extended_ratio = (sum(1 for v in fs.values() if v) / len(fs)) if fs else 0.0
    if extended_ratio < min_extended_ratio:
        return False
    if abs(vx) / norm < min_abs_x:
        return False
    if toward == "right":
        return vx > 0.0
    if toward == "left":
        return vx < 0.0
    return False


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
    left_angle = steering_angle_one_hand(left_lms)
    right_angle = steering_angle_one_hand(right_lms)
    return average_angle_deg(left_angle, right_angle)


def steering_angle_one_hand(hand_lms) -> float:
    wrist = lm_xy(hand_lms, 0)
    index_mcp = lm_xy(hand_lms, 5)
    pinky_mcp = lm_xy(hand_lms, 17)
    mid = ((index_mcp[0] + pinky_mcp[0]) / 2.0, (index_mcp[1] + pinky_mcp[1]) / 2.0)
    return rad2deg(angle_2d(wrist, mid))
