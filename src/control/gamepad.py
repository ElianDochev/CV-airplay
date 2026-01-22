"""Virtual gamepad abstraction with a Linux uinput backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class AxisSpec:
    code: str
    min_value: int
    max_value: int


class VirtualGamepadBase:
    """Base interface for virtual gamepad backends.

    Logical controls are used to keep parity across platforms.
    """

    def press(self, button: str) -> None:
        raise NotImplementedError

    def release(self, button: str) -> None:
        raise NotImplementedError

    def set_axis(self, axis: str, value: float) -> None:
        raise NotImplementedError

    def sync(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


LOGICAL_BUTTONS = {
    "A",
    "B",
    "X",
    "Y",
    "LB",
    "RB",
    "BACK",
    "START",
    "LS",
    "RS",
    "GUIDE",
}

LOGICAL_AXES = {
    "LS_X",
    "LS_Y",
    "RS_X",
    "RS_Y",
    "LT",
    "RT",
    "DPAD_X",
    "DPAD_Y",
}


class LinuxUInputGamepad(VirtualGamepadBase):
    """Linux virtual gamepad using python-uinput.

    Logical controls:
      Buttons: A, B, X, Y, LB, RB, BACK, START, LS, RS, GUIDE
      Axes: LS_X, LS_Y, RS_X, RS_Y, LT, RT, DPAD_X, DPAD_Y
    """

    def __init__(self, name: str = "GestureFusion Gamepad") -> None:
        try:
            import uinput  # type: ignore
        except Exception as exc:  # pragma: no cover - handled by caller
            raise ImportError(
                "python-uinput is required on Linux. Install with 'pip install python-uinput'."
            ) from exc

        self._uinput = uinput

        self._button_map = {
            "A": uinput.BTN_A,
            "B": uinput.BTN_B,
            "X": uinput.BTN_X,
            "Y": uinput.BTN_Y,
            "LB": uinput.BTN_TL,
            "RB": uinput.BTN_TR,
            "BACK": uinput.BTN_SELECT,
            "START": uinput.BTN_START,
            "LS": uinput.BTN_THUMBL,
            "RS": uinput.BTN_THUMBR,
            "GUIDE": uinput.BTN_MODE,
        }

        self._axis_map: Dict[str, AxisSpec] = {
            "LS_X": AxisSpec("ABS_X", -32768, 32767),
            "LS_Y": AxisSpec("ABS_Y", -32768, 32767),
            "RS_X": AxisSpec("ABS_RX", -32768, 32767),
            "RS_Y": AxisSpec("ABS_RY", -32768, 32767),
            "LT": AxisSpec("ABS_Z", 0, 255),
            "RT": AxisSpec("ABS_RZ", 0, 255),
            "DPAD_X": AxisSpec("ABS_HAT0X", -1, 1),
            "DPAD_Y": AxisSpec("ABS_HAT0Y", -1, 1),
        }

        self._device = uinput.Device(self._build_events(), name=name)

    def _build_events(self) -> Iterable:
        events = []
        for button in self._button_map.values():
            events.append(button)

        for axis_name, spec in self._axis_map.items():
            code = getattr(self._uinput, spec.code)
            events.append(code + (spec.min_value, spec.max_value, 0, 0))

        return events

    def press(self, button: str) -> None:
        code = self._button_code(button)
        self._device.emit(code, 1, syn=False)

    def release(self, button: str) -> None:
        code = self._button_code(button)
        self._device.emit(code, 0, syn=False)

    def set_axis(self, axis: str, value: float) -> None:
        spec = self._axis_spec(axis)
        code = getattr(self._uinput, spec.code)
        converted = self._convert_axis_value(axis, value, spec)
        self._device.emit(code, converted, syn=False)

    def sync(self) -> None:
        self._device.syn()

    def close(self) -> None:
        self._device.destroy()

    def _button_code(self, button: str) -> int:
        if button not in self._button_map:
            raise ValueError(f"Unknown button '{button}'.")
        return self._button_map[button]

    def _axis_spec(self, axis: str) -> AxisSpec:
        if axis not in self._axis_map:
            raise ValueError(f"Unknown axis '{axis}'.")
        return self._axis_map[axis]

    def _convert_axis_value(self, axis: str, value: float, spec: AxisSpec) -> int:
        if axis in {"LT", "RT"}:
            clamped = max(0.0, min(1.0, value))
        elif axis in {"DPAD_X", "DPAD_Y"}:
            clamped = max(-1.0, min(1.0, value))
        else:
            clamped = max(-1.0, min(1.0, value))

        if spec.min_value == -1 and spec.max_value == 1:
            return int(round(clamped))

        scaled = int(round(((clamped + 1.0) / 2.0) * (spec.max_value - spec.min_value) + spec.min_value))
        if axis in {"LT", "RT"}:
            scaled = int(round(clamped * (spec.max_value - spec.min_value) + spec.min_value))
        return scaled
