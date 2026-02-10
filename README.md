# AirDrive — Vision Racing Controller (OpenCV + MediaPipe)

Turn **two-hand gestures** into a **racing game controller** (steering wheel + throttle + brake) in real time.
Runs on **Windows + Linux**, supports **live demo while the script is running**, and can be extended with **custom gesture datasets + finetuning**.

## Features (Target)

* 🎥 Real-time webcam hand tracking (2 hands)
* 🧭 Steering wheel pose from hand positions
* ⛽ Throttle + brake from hand gestures/positions
* 🎮 Virtual racing controller output (wheel + pedals)
* 🧠 Two recognition modes:

  1. **Rule-based** (fastest, no training) using MediaPipe landmarks
  2. **ML-based** (more scalable) finetune/classify gestures from landmarks

---

## Project Structure (Proposed)

```

  README.md
  requirements.txt
  src/
    app.py                      # main realtime loop (camera -> hands -> race controls)
    vision/
      mediapipe_hands.py         # hand detection + landmarks
      preprocessing.py
    gestures/
      rules.py                   # rule-based gesture definitions
      features.py                # landmark -> feature vectors
      classifier.py              # ML classifier inference (optional)
      temporal.py                # smoothing/temporal filters
    control/
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

  * `OPEN`, `FIST`, `POINT`, `THUMB_UP`, `OK` (optional)
* [ ] Two-hand driving poses:

  * both hands visible => steering wheel
  * one hand visible => fallback steering
* [ ] Temporal smoothing:

  * majority vote over last N frames
* [ ] Debounce + cooldown:

  * avoid spamming actions every frame

**Deliverable:** stable gesture prediction + driving pose logic.

---

### 3) Controller Output (Racing Wheel + Pedals) ✅

#### Virtual Racing Gamepad (best compatibility)

* **Windows:** `vgamepad` (Xbox controller emulation)

* **Linux:** `uinput` / `python-uinput` (virtual controller device)

* [ ] Implement `control/gamepad.py` backend

* [ ] Map steering to left stick X or wheel axis
* [ ] Map throttle to right trigger (RT)
* [ ] Map brake to left trigger (LT)

* [ ] Add a “demo tester” mode: show current steering angle + pedal values

**Deliverable:** games detect it as a racing controller.

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
  * show steering + throttle + brake values
  * show controller backend status (gamepad)

**Deliverable:** You can run the script and control a racing game on both OS.

---

## Gesture Set (Example)

**Steering:**

* Both hands visible: steering angle from the line between hands
* One hand visible: steering angle from wrist x-position

**Throttle / Brake:**

* `RIGHT FIST` = throttle (RT)
* `LEFT FIST` = brake (LT)
* `OPEN` = release pedal (0)

**Assist gestures (optional):**

* `THUMB_UP` = handbrake / boost

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

---

## Run (Demo)

```bash
python -m src.app --backend gamepad --camera 0
```

Example flags (planned):

* `--backend gamepad`
* `--mode rules|ml`
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

* Racing game controlled by gestures (steer/accelerate/brake)
* A “controller tester” page/app showing steering + pedals

---

## Safety Notes

* This project is intended for **local demos** and accessibility/HCI research.
* Do not use it for surveillance or identifying individuals.

---
---
