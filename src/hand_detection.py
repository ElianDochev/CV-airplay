from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config_utils import get_cfg


DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)


def ensure_model(model_path: Path, model_url: str) -> None:
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(model_url, model_path)


def build_hand_landmarker(cfg: dict) -> vision.HandLandmarker:
    model_path = Path(get_cfg(cfg, "hand_landmarker_model_path", "models/hand_landmarker.task"))
    model_url = get_cfg(cfg, "hand_landmarker_model_url", DEFAULT_MODEL_URL)
    ensure_model(model_path, model_url)

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=cfg["max_num_hands"],
        min_hand_detection_confidence=cfg["min_detection_confidence"],
        min_hand_presence_confidence=cfg["min_detection_confidence"],
        min_tracking_confidence=cfg["min_tracking_confidence"],
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


def hand_label(handedness_entry) -> str:
    if not handedness_entry:
        return "Unknown"
    classification = handedness_entry[0]
    return getattr(classification, "category_name", getattr(classification, "label", "Unknown"))


def resolve_hand_label(label: str, mirror_input: bool) -> str:
    if not mirror_input:
        return label
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def render_landmarks(frame, hand_lms, color=(0, 255, 0)) -> None:
    landmarks = hand_lms.landmark if hasattr(hand_lms, "landmark") else hand_lms
    for lm in landmarks:
        x = int(lm.x * frame.shape[1])
        y = int(lm.y * frame.shape[0])
        cv2.circle(frame, (x, y), 3, color, -1)
