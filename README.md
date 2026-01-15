# GestureFusion — Dual-Hand Vision Game Controller (OpenCV + MediaPipe + Optional YOLO)

Turn **two-hand gestures** into a **virtual game controller** (keyboard/mouse or gamepad) in real time.
Runs on **Windows + Linux**, supports **live demo while the script is running**, and can be extended with **custom gesture datasets + finetuning**.

## Features (Target)

* 🎥 Real-time webcam hand tracking (2 hands)
* ✋ Static gestures (open, fist, point, peace, thumbs-up, etc.)
* 🌀 Optional dynamic gestures (swipe, circle, push/pull)
* 🎮 Gesture → **controller actions**:

  * Keyboard (WASD, space, etc.)
  * Mouse (aim/click) (optional)
  * Virtual gamepad (recommended for broad game support)
* 🧠 Two recognition modes:

  1. **Rule-based** (fastest, no training) using MediaPipe landmarks
  2. **ML-based** (more scalable) finetune/classify gestures from landmarks and/or YOLO

---

## Project Structure (Proposed)

```
gesturefusion/
  README.md
  requirements.txt
  src/
    app.py                      # main realtime loop (camera -> gesture -> controller)
    vision/
      mediapipe_hands.py         # hand detection + landmarks
      yolo_hands.py              # optional YOLO hand detector
      preprocessing.py
    gestures/
      rules.py                   # rule-based gesture definitions
      features.py                # landmark -> feature vectors
      classifier.py              # ML classifier inference (optional)
      temporal.py                # swipe/circle detection (optional)
    control/
      keyboard_mouse.py          # OS input backend (pynput)
      gamepad.py                 # virtual gamepad backend (vgamepad / uinput)
      mapping.py                 # gesture->action mapping + combos
      debounce.py                # smoothing / cooldown / state machine
    data/
      collect.py                 # data collection tool (record landmarks, labels)
      viewer.py                  # quick dataset inspection
    train/
      train_classifier.py        # train gesture classifier from landmarks
      export.py                  # export model (onnx etc.)
  assets/
    demo.gif
  configs/
    gestures.yaml                # mapping config + thresholds
```

---

## Roadmap / Implementation Checklist

### 1) Baseline Real-Time Hand Tracking ✅

* [ ] OpenCV webcam loop (FPS overlay)
* [ ] MediaPipe Hands: detect up to 2 hands + 21 landmarks
* [ ] Identify handedness (Left / Right)
* [ ] Draw landmarks + show current recognized gesture per hand

**Deliverable:** `src/app.py` runs and prints `Left: OPEN, Right: FIST` at 20–60 FPS.

---

### 2) Gesture Recognition v1 (Rule-Based) ✅

Implement simple gesture classification using landmarks:

* [ ] Finger state detection (up/down for each finger)
* [ ] Basic gesture set:

  * `OPEN`, `FIST`, `POINT`, `PEACE`, `THUMB_UP`, `OK` (optional)
* [ ] Two-hand combos:

  * e.g. `Left=OPEN + Right=FIST => SPECIAL`
* [ ] Temporal smoothing:

  * majority vote over last N frames
* [ ] Debounce + cooldown:

  * avoid spamming actions every frame

**Deliverable:** stable gesture prediction + combo logic.

---

### 3) Controller Agent Tools (Gesture → Input) ✅

#### Option A: Keyboard/Mouse Output (fastest to demo)

* [ ] Use `pynput` to press/release keys
* [ ] Support “hold” actions (move forward while POINT is active)
* [ ] Support “tap” actions (jump once on PEACE trigger)
* [ ] Optional mouse control: move cursor / click (aim & shoot)

**Deliverable:** gestures control a game that supports keyboard/mouse.

#### Option B: Virtual Gamepad (best compatibility)

* **Windows:** `vgamepad` (Xbox controller emulation)

* **Linux:** `uinput` / `python-uinput` (virtual controller device)

* [ ] Implement `control/gamepad.py` backend

* [ ] Map gestures to gamepad buttons/axes (A/B/X/Y, triggers, sticks)

* [ ] Add a “demo tester” mode: show current stick values + pressed buttons

**Deliverable:** games detect it as a real controller.

---

### 4) Data Collection (for ML training / finetuning) ✅

Even if you start rule-based, collecting data makes your project *CV-ready*.

Collect **landmark-based samples**:

* [ ] Build `src/data/collect.py`
* [ ] Hotkey labeling (press `1..9` to label current gesture)
* [ ] Save per-frame:

  * 21 landmarks * (x,y,z) per hand (and handedness)
  * optional bounding box, timestamp, FPS, lighting
* [ ] Save format: CSV or JSONL (recommended: JSONL)
* [ ] Dataset split: train/val/test
* [ ] Record multiple sessions:

  * different lighting, backgrounds, distances, users

**Deliverable:** reproducible dataset with clear labels.

---

### 5) AI Training / Finetuning ✅

You have two good ML approaches:

#### Approach 1: Landmark Classifier (Recommended)

Train a classifier on MediaPipe landmarks (fast + robust):

* [ ] Convert landmarks → features:

  * normalized coordinates (relative to wrist)
  * pairwise distances/angles (optional)
* [ ] Train model:

  * baseline: Logistic Regression / SVM (very fast)
  * stronger: small MLP (PyTorch)
* [ ] Evaluate:

  * accuracy, confusion matrix
  * latency & FPS impact
* [ ] Export model:

  * pickle (sklearn) or ONNX (portable)

**Deliverable:** `train/train_classifier.py` produces `models/gesture.onnx` or `models/gesture.pkl`.

#### Approach 2: YOLO Finetune (Optional)

Use YOLO for direct gesture classification from images:

* [ ] Collect image dataset (hands cropped)
* [ ] Annotate gesture classes
* [ ] Train YOLOv8n for speed
* [ ] Use YOLO inference live

**Deliverable:** works but usually heavier than landmarks for gestures.

---

### 6) Cross-Platform Demo (Windows + Linux) ✅

* [ ] Ensure consistent dependencies
* [ ] Provide separate setup instructions
* [ ] Provide “Demo Mode”:

  * show predicted gestures
  * show which actions are being triggered
  * show controller backend status (keyboard/gamepad)

**Deliverable:** You can run the script and control a game on both OS.

---

## Gesture Set (Example)

**Left hand (movement):**

* `POINT` = move forward (W)
* `OPEN` = stop (release W)
* `THUMB_UP` = sprint (Shift)

**Right hand (actions):**

* `FIST` = shoot (mouse click / gamepad RT)
* `PEACE` = jump (Space / gamepad A)
* `OPEN` = reload (R / gamepad X)

**Combo examples:**

* Left `OPEN` + Right `FIST` => Special ability (E)
* Both `FIST` => Block (hold RMB)

---

## Installation

### Common

```bash
pip install -r requirements.txt
```

`requirements.txt` (starter):

* opencv-python
* mediapipe
* numpy
* pynput

Optional:

* torch (if training MLP)
* scikit-learn (if SVM/logreg)
* onnxruntime (if ONNX inference)
* ultralytics (if YOLO mode)

---

## Run (Demo)

```bash
python -m src.app --backend keyboard --camera 0
```

Example flags (planned):

* `--backend keyboard|gamepad`
* `--mode rules|ml|yolo`
* `--config configs/gestures.yaml`
* `--show-ui 1`
* `--record-data 0`

---

## Data Collection

```bash
python -m src.data.collect --out data/sessions/session_01.jsonl
```

Planned controls:

* Press `1..9` to set label
* Press `r` to start/stop recording
* Press `q` to quit

---

## Training

Landmark classifier:

```bash
python -m src.train.train_classifier \
  --data data/sessions \
  --out models/gesture.onnx
```

---

## Metrics to Report (For CV)

* FPS on CPU and GPU (if applicable)
* Gesture accuracy / confusion matrix
* Latency from camera frame → action trigger
* Robustness: lighting + background variation
* Cross-platform support (Windows + Linux)

---

## Demo Ideas

* Platformer game controlled by gestures (jump/shoot/move)
* A “controller tester” page/app showing button presses
* OBS overlay showing gestures + actions live

---

## Safety Notes

* This project is intended for **local demos** and accessibility/HCI research.
* Do not use it for surveillance or identifying individuals.

---
