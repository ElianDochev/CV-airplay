from __future__ import annotations

from typing import Optional

from .mapping import steer_to_axis_value, trigger_value

try:
    import uinput
except ImportError:  # pragma: no cover - optional dependency
    uinput = None

try:
    import vgamepad as vg
except ImportError:  # pragma: no cover - optional dependency
    vg = None


class GamepadBackend:
    def update(self, steer: float, accel: bool, brake: bool) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class NullGamepad(GamepadBackend):
    def update(self, steer: float, accel: bool, brake: bool) -> None:
        return None


class UInputGamepad(GamepadBackend):
    def __init__(self) -> None:
        if uinput is None:
            raise RuntimeError("python-uinput is not installed")

        self.device = uinput.Device(
            [
                uinput.ABS_X + (-32768, 32767, 0, 0),
                uinput.ABS_Z + (0, 255, 0, 0),
                uinput.ABS_RZ + (0, 255, 0, 0),
            ]
        )

    def update(self, steer: float, accel: bool, brake: bool) -> None:
        self.device.emit(uinput.ABS_X, steer_to_axis_value(steer), syn=False)
        self.device.emit(uinput.ABS_RZ, trigger_value(accel), syn=False)
        self.device.emit(uinput.ABS_Z, trigger_value(brake), syn=True)


class VGamepad(GamepadBackend):
    def __init__(self) -> None:
        if vg is None:
            raise RuntimeError("vgamepad is not installed")

        self.gamepad = vg.VX360Gamepad()

    def update(self, steer: float, accel: bool, brake: bool) -> None:
        x_value = steer_to_axis_value(steer)
        self.gamepad.left_joystick(x_value, 0)
        self.gamepad.right_trigger(trigger_value(accel))
        self.gamepad.left_trigger(trigger_value(brake))
        self.gamepad.update()


def create_gamepad_backend(name: Optional[str]) -> GamepadBackend:
    if not name or name == "none":
        return NullGamepad()
    if name == "uinput":
        return UInputGamepad()
    if name == "vgamepad":
        return VGamepad()

    raise ValueError(f"Unknown backend: {name}")
