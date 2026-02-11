from __future__ import annotations


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def steer_to_axis_value(steer: float, axis_min: int = -32768, axis_max: int = 32767) -> int:
    steer = clamp(steer, -1.0, 1.0)
    span = axis_max - axis_min
    return int(axis_min + ((steer + 1.0) / 2.0) * span)


def trigger_value(active: bool, max_value: int = 255) -> int:
    return max_value if active else 0
