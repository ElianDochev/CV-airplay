from __future__ import annotations

from typing import Any, Dict, Optional

from .gamepad import GamepadBackend, NullGamepad, create_gamepad_backend
from .keyboard import NullKeyboard, create_keyboard_backend


class ControlBackend:
    def update(self, steer: float, accel: bool, brake: bool) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class GamepadControl(ControlBackend):
    def __init__(self, backend: GamepadBackend) -> None:
        self.backend = backend

    def update(self, steer: float, accel: bool, brake: bool) -> None:
        self.backend.update(steer, accel, brake)

    def close(self) -> None:
        self.backend.close()


class KeyboardControl(ControlBackend):
    def __init__(self, backend: KeyboardBackend) -> None:
        self.backend = backend

    def update(self, steer: float, accel: bool, brake: bool) -> None:
        self.backend.update(steer, accel, brake)

    def close(self) -> None:
        self.backend.close()


class NullControl(ControlBackend):
    def __init__(self) -> None:
        self.backend = NullGamepad()

    def update(self, steer: float, accel: bool, brake: bool) -> None:
        return None


def _get(cfg: Dict[str, Any], key: str, default):
    return cfg[key] if key in cfg else default


def create_control_backend(controls_cfg: Dict[str, Any], backend_override: Optional[str] = None) -> ControlBackend:
    control_type = _get(controls_cfg, "type", "none")

    if backend_override:
        if backend_override == "none":
            return NullControl()
        control_type = "gamepad"
        controls_cfg = dict(controls_cfg)
        controls_cfg.setdefault("gamepad", {})
        controls_cfg["gamepad"]["backend"] = backend_override

    if control_type == "gamepad":
        backend = _get(controls_cfg, "gamepad", {})
        return GamepadControl(create_gamepad_backend(_get(backend, "backend", None)))

    if control_type == "keyboard":
        keyboard_cfg = _get(controls_cfg, "keyboard", {})
        return KeyboardControl(
            create_keyboard_backend(
                name=_get(keyboard_cfg, "backend", None),
                left_key=_get(keyboard_cfg, "left_key", "left"),
                right_key=_get(keyboard_cfg, "right_key", "right"),
                accel_key=_get(keyboard_cfg, "accel_key", "w"),
                brake_key=_get(keyboard_cfg, "brake_key", "space"),
                steer_threshold=float(_get(keyboard_cfg, "steer_threshold", 0.2)),
                steer_hold_threshold=float(_get(keyboard_cfg, "steer_hold_threshold", 0.5)),
                brake_steer_threshold=float(
                    _get(keyboard_cfg, "brake_steer_threshold", _get(keyboard_cfg, "steer_threshold", 0.2))
                ),
                brake_steer_hold_threshold=float(
                    _get(keyboard_cfg, "brake_steer_hold_threshold", _get(keyboard_cfg, "steer_hold_threshold", 0.5))
                ),
            )
        )

    return NullControl()
