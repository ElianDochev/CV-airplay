from __future__ import annotations

import os
from typing import Optional

from pynput import keyboard

try:
    import uinput
except ImportError:  # pragma: no cover - optional dependency
    uinput = None


class KeyboardBackend:
    def __init__(
        self,
        left_key: str,
        right_key: str,
        accel_key: str,
        brake_key: str,
        steer_threshold: float,
        steer_hold_threshold: float,
        brake_steer_threshold: float,
        brake_steer_hold_threshold: float,
    ):
        self.controller = keyboard.Controller()
        self.left_key = self._parse_key(left_key)
        self.right_key = self._parse_key(right_key)
        self.accel_key = self._parse_key(accel_key)
        self.brake_key = self._parse_key(brake_key)
        self.steer_threshold = steer_threshold
        self.steer_hold_threshold = steer_hold_threshold
        self.brake_steer_threshold = brake_steer_threshold
        self.brake_steer_hold_threshold = brake_steer_hold_threshold
        self._pressed = set()

    def _parse_key(self, name: str):
        name = name.lower().strip()
        if hasattr(keyboard.Key, name):
            return getattr(keyboard.Key, name)
        if len(name) == 1:
            return keyboard.KeyCode.from_char(name)
        raise ValueError(f"Unsupported key: {name}")

    def _set_key(self, key, pressed: bool) -> None:
        if pressed and key not in self._pressed:
            self.controller.press(key)
            self._pressed.add(key)
        elif not pressed and key in self._pressed:
            self.controller.release(key)
            self._pressed.remove(key)

    def update(self, steer: float, accel: bool, brake: bool) -> None:
        steer_threshold = self.brake_steer_threshold if brake else self.steer_threshold
        steer_hold_threshold = self.brake_steer_hold_threshold if brake else self.steer_hold_threshold
        if steer >= steer_hold_threshold:
            left = False
            right = True
        elif steer <= -steer_hold_threshold:
            left = True
            right = False
        else:
            left = steer < -steer_threshold
            right = steer > steer_threshold
        self._set_key(self.left_key, left)
        self._set_key(self.right_key, right)
        self._set_key(self.accel_key, accel)
        self._set_key(self.brake_key, brake)

    def close(self) -> None:
        for key in list(self._pressed):
            self.controller.release(key)
        self._pressed.clear()


class NullKeyboard(KeyboardBackend):
    def __init__(self):
        self._pressed = set()

    def update(self, steer: float, accel: bool, brake: bool) -> None:
        return None

    def close(self) -> None:
        return None


class UInputKeyboard:
    def __init__(
        self,
        left_key: str,
        right_key: str,
        accel_key: str,
        brake_key: str,
        steer_threshold: float,
        steer_hold_threshold: float,
        brake_steer_threshold: float,
        brake_steer_hold_threshold: float,
    ):
        if uinput is None:
            raise RuntimeError("python-uinput is not installed")

        self.left_key = self._parse_key(left_key)
        self.right_key = self._parse_key(right_key)
        self.accel_key = self._parse_key(accel_key)
        self.brake_key = self._parse_key(brake_key)
        self.steer_threshold = steer_threshold
        self.steer_hold_threshold = steer_hold_threshold
        self.brake_steer_threshold = brake_steer_threshold
        self.brake_steer_hold_threshold = brake_steer_hold_threshold
        self._pressed = set()

        self.device = uinput.Device(
            [self.left_key, self.right_key, self.accel_key, self.brake_key]
        )

    def _parse_key(self, name: str):
        name = name.lower().strip()
        special = {
            "left": uinput.KEY_LEFT,
            "right": uinput.KEY_RIGHT,
            "up": uinput.KEY_UP,
            "down": uinput.KEY_DOWN,
            "space": uinput.KEY_SPACE,
            "enter": uinput.KEY_ENTER,
            "esc": uinput.KEY_ESC,
            "tab": uinput.KEY_TAB,
            "backspace": uinput.KEY_BACKSPACE,
        }
        if name in special:
            return special[name]
        if len(name) == 1 and name.isalpha():
            key_name = f"KEY_{name.upper()}"
            if hasattr(uinput, key_name):
                return getattr(uinput, key_name)
        if len(name) == 1 and name.isdigit():
            key_name = f"KEY_{name}"
            if hasattr(uinput, key_name):
                return getattr(uinput, key_name)
        raise ValueError(f"Unsupported key: {name}")

    def _set_key(self, key, pressed: bool, syn: bool) -> None:
        if pressed and key not in self._pressed:
            self.device.emit(key, 1, syn=syn)
            self._pressed.add(key)
        elif not pressed and key in self._pressed:
            self.device.emit(key, 0, syn=syn)
            self._pressed.remove(key)

    def update(self, steer: float, accel: bool, brake: bool) -> None:
        steer_threshold = self.brake_steer_threshold if brake else self.steer_threshold
        steer_hold_threshold = self.brake_steer_hold_threshold if brake else self.steer_hold_threshold
        if steer >= steer_hold_threshold:
            left = False
            right = True
        elif steer <= -steer_hold_threshold:
            left = True
            right = False
        else:
            left = steer < -steer_threshold
            right = steer > steer_threshold
        states = [
            (self.left_key, left),
            (self.right_key, right),
            (self.accel_key, accel),
            (self.brake_key, brake),
        ]
        for idx, (key, pressed) in enumerate(states):
            syn = idx == (len(states) - 1)
            self._set_key(key, pressed, syn=syn)

    def close(self) -> None:
        for idx, key in enumerate(list(self._pressed)):
            syn = idx == (len(self._pressed) - 1)
            self.device.emit(key, 0, syn=syn)
        self._pressed.clear()


def create_keyboard_backend(
    name: Optional[str],
    left_key: str,
    right_key: str,
    accel_key: str,
    brake_key: str,
    steer_threshold: float,
    steer_hold_threshold: float,
    brake_steer_threshold: float,
    brake_steer_hold_threshold: float,
):
    backend_name = (name or "auto").lower().strip()
    if backend_name == "uinput":
        backend = UInputKeyboard(
            left_key,
            right_key,
            accel_key,
            brake_key,
            steer_threshold,
            steer_hold_threshold,
            brake_steer_threshold,
            brake_steer_hold_threshold,
        )
        print(f"[keyboard] backend=uinput keys={left_key},{right_key},{accel_key},{brake_key}")
        return backend
    if backend_name == "pynput":
        backend = KeyboardBackend(
            left_key,
            right_key,
            accel_key,
            brake_key,
            steer_threshold,
            steer_hold_threshold,
            brake_steer_threshold,
            brake_steer_hold_threshold,
        )
        print(f"[keyboard] backend=pynput keys={left_key},{right_key},{accel_key},{brake_key}")
        return backend
    if backend_name == "auto":
        if uinput is not None and os.path.exists("/dev/uinput"):
            try:
                backend = UInputKeyboard(
                    left_key,
                    right_key,
                    accel_key,
                    brake_key,
                    steer_threshold,
                    steer_hold_threshold,
                    brake_steer_threshold,
                    brake_steer_hold_threshold,
                )
                print(f"[keyboard] backend=uinput keys={left_key},{right_key},{accel_key},{brake_key}")
                return backend
            except Exception:
                pass
        backend = KeyboardBackend(
            left_key,
            right_key,
            accel_key,
            brake_key,
            steer_threshold,
            steer_hold_threshold,
            brake_steer_threshold,
            brake_steer_hold_threshold,
        )
        print(f"[keyboard] backend=pynput keys={left_key},{right_key},{accel_key},{brake_key}")
        return backend

    raise ValueError(f"Unknown keyboard backend: {name}")
