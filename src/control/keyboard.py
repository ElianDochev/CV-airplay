from __future__ import annotations

from typing import Optional

from pynput import keyboard


class KeyboardBackend:
    def __init__(self, left_key: str, right_key: str, accel_key: str, brake_key: str, steer_threshold: float):
        self.controller = keyboard.Controller()
        self.left_key = self._parse_key(left_key)
        self.right_key = self._parse_key(right_key)
        self.accel_key = self._parse_key(accel_key)
        self.brake_key = self._parse_key(brake_key)
        self.steer_threshold = steer_threshold
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
        left = steer < -self.steer_threshold
        right = steer > self.steer_threshold
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
