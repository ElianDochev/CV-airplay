"""Gesture to gamepad action mappings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .gamepad import LOGICAL_AXES, LOGICAL_BUTTONS, VirtualGamepadBase


@dataclass(frozen=True)
class Action:
    """A single logical gamepad action."""

    kind: str  # "button" or "axis"
    control: str
    value: Optional[float] = None
    pressed: Optional[bool] = None


DEFAULT_GAMEPAD_MAPPING: Dict[str, Dict[str, List[Action]]] = {
    "left": {
        "POINT": [Action(kind="axis", control="LS_Y", value=-1.0)],
        "OPEN": [Action(kind="axis", control="LS_Y", value=0.0)],
        "THUMB_UP": [Action(kind="button", control="LB", pressed=True)],
    },
    "right": {
        "FIST": [Action(kind="axis", control="RT", value=1.0)],
        "PEACE": [Action(kind="button", control="A", pressed=True)],
        "OPEN": [Action(kind="button", control="X", pressed=True)],
    },
}


def actions_from_mapping(hand: str, gesture: str, mapping: Dict[str, Dict[str, List[Action]]] = DEFAULT_GAMEPAD_MAPPING) -> List[Action]:
    """Return actions for a given hand/gesture pair."""
    hand_key = hand.lower()
    if hand_key not in mapping:
        return []
    return mapping[hand_key].get(gesture, [])


def apply_actions(gamepad: VirtualGamepadBase, actions: Iterable[Action]) -> None:
    """Apply actions to a gamepad. Call `sync()` after applying a batch."""
    for action in actions:
        _validate_action(action)
        if action.kind == "button":
            if action.control in {"LT", "RT"} and action.value is not None:
                gamepad.set_axis(action.control, action.value)
            elif action.pressed:
                gamepad.press(action.control)
            else:
                gamepad.release(action.control)
        elif action.kind == "axis":
            if action.value is None:
                raise ValueError("Axis action requires a value.")
            gamepad.set_axis(action.control, action.value)
        else:
            raise ValueError(f"Unknown action kind '{action.kind}'.")


def _validate_action(action: Action) -> None:
    if action.control in LOGICAL_BUTTONS:
        return
    if action.control in LOGICAL_AXES:
        return
    raise ValueError(f"Unknown control '{action.control}'.")


def load_yaml_mapping(path: str) -> Dict[str, Dict[str, List[Action]]]:
    """Load a mapping file in the configs/gestures.yaml format."""
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyYAML is required to load YAML mappings.") from exc

    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    gamepad_mapping = payload.get("gamepad", {})
    converted: Dict[str, Dict[str, List[Action]]] = {}

    for hand, gestures in gamepad_mapping.items():
        converted[hand] = {}
        for gesture, actions in (gestures or {}).items():
            converted[hand][gesture] = [_action_from_dict(item) for item in actions]

    return converted


def _action_from_dict(data: Dict[str, object]) -> Action:
    kind = str(data.get("type", ""))
    control = str(data.get("control", ""))
    value = data.get("value")
    pressed = data.get("pressed")

    if kind == "axis":
        if value is None:
            raise ValueError("Axis mapping requires a value.")
        return Action(kind="axis", control=control, value=float(value))

    if kind == "button":
        pressed_value = True if pressed is None else bool(pressed)
        return Action(kind="button", control=control, pressed=pressed_value)

    raise ValueError(f"Unknown action type '{kind}'.")
