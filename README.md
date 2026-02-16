# AirDrive — Vision Racing Controller (OpenCV + MediaPipe)

Turn two-hand gestures into a racing controller (steering + throttle + brake) in real time. The current implementation is a rule-based MediaPipe hand landmark pipeline with optional virtual controller backends.

## What It Does

- Tracks up to two hands from a webcam.
- Computes steering angle from two hands (wrist rotation).
- Uses two thumbs up for accel and thumb-up + fingers sideways for brake.
- Sends output to a virtual controller backend (uinput on Linux, vgamepad on Windows).

## Project Structure

```
README.md
requirements.txt
docker-compose.yml
Dockerfile.cpu
Dockerfile.gpu
config/
  main.yml
  controls.yml
notebooks/
  00_env_check.ipynb
  01_mediapipe_hands.ipynb
  02_landmarks_analysis.ipynb
  03_rule_based_gestures.ipynb
src/
  app.py
  gesture_controller.py
  ulits.py
  control/
    backend.py
    gamepad.py
    keyboard.py
    mapping.py
```

## Local Run

```bash
pip install -r requirements.txt
python -m src.app --controller on --camera 0 --show-ui
```

Common flags:

- `--config config/main.yml`: main runtime config (detection thresholds, UI defaults).
- `--controls-config config/controls.yml`: controller mapping and backend settings.
- `--controller on|off`: start controller active after calibration (use SPACE to toggle).
- `--show-ui/--no-show-ui`: enable or disable the on-screen preview window.
- `--draw-landmarks/--no-draw-landmarks`: render MediaPipe hand landmarks on the preview.
- `--mirror-input/--no-mirror-input`: mirror the webcam feed for natural left/right control.
- `--show-fps/--no-show-fps`: display FPS on the preview.

Controls while running:

- `c` clears calibration and starts the guided calibration flow (controller output disabled while calibrating)
- `space` toggles controller output (3 second countdown when enabling)
- two thumbs up (both hands) auto-calibrates neutral steering (only when no calibration file is loaded)
- `q` quits

Calibration UI:

- Shows a 3-2-1-GO countdown before each capture stage
- Shows only the current calibration action
- Stages: two-hand neutral (thumbs up), two-hand left/right (rotate wrists), brake in neutral/left/right (thumb up + fingers sideways toward center)

## Docker (CPU and GPU Profiles)

CPU (Python 3.12):

```bash
export CONTROLLER=on
docker compose --profile cpu up --build
```

GPU (CUDA 11.8, Python 3.11):

```bash
export CONTROLLER=on
docker compose --profile gpu up --build
```

Windows (PowerShell):

```powershell
$env:CONTROLLER = "on"
```

Windows (CMD):

```bat
set CONTROLLER=on
```

Notes:

- On Linux, allow local Docker access to X11 before running:

  ```bash
  xhost +local:
  ```

- Linux requires `/dev/video0` access. Joystick mode also needs `/dev/uinput` access.
- For joystick mode on Linux, you may need `sudo modprobe uinput` on the host.
- The container runs headless by default with `--no-show-ui`.

Windows note: WSL2 + Docker Desktop typically uses WSLg for GUI apps, so `xhost` is not required. If you are running an X server (VcXsrv/Xming), allow local connections in that server's settings.

## How It Works

- **Neutral + accel**: both hands in closed-fist thumbs up, held centered in front of the chest.
- **Steering**: rotate both wrists together left/right while keeping thumbs up.
- **Brake**: keep thumb up and turn the palm sideways; fingers point toward the center (left hand fingers point right, right hand fingers point left). Brake can be done while steering.

## Calibration

1. Press `c` to start calibration.
2. Follow the on-screen prompt for each stage.
3. Keep both hands visible and centered in the frame.

Stages (in order):

- Two hands: NEUTRAL (thumbs up)
- Two hands: steer LEFT (rotate wrists)
- Two hands: steer RIGHT (rotate wrists)
- Two hands: BRAKE in NEUTRAL (thumb up + fingers sideways)
- Two hands: BRAKE while LEFT (thumb up + fingers sideways)
- Two hands: BRAKE while RIGHT (thumb up + fingers sideways)

Calibration data is saved to [calibration/calibration.yml](calibration/calibration.yml). To use a preset, copy [example-calibration/calibration.yml](example-calibration/calibration.yml) into [calibration/calibration.yml](calibration/calibration.yml).

## Config

Main settings live in [config/main.yml](config/main.yml) and control detection thresholds, UI flags, and camera defaults. Control mappings are in [config/controls.yml](config/controls.yml).

### config/main.yml

- `max_num_hands`: Max hands to track (2 recommended).
- `model_complexity`: MediaPipe model size (higher = slower, more accurate).
- `min_detection_confidence`: Hand detection confidence threshold.
- `min_tracking_confidence`: Hand tracking confidence threshold.
- `hand_landmarker_model_path`: Local path to the MediaPipe hand model file.
- `hand_landmarker_model_url`: Download URL for the MediaPipe hand model.
- `finger_extended_margin`: Vertical margin to mark a finger as extended (steering/neutral).
- `finger_extended_radial_margin`: Radial margin to mark a finger as extended (steering/neutral).
- `action_finger_extended_margin`: Vertical margin for action gestures (brake/accel checks).
- `action_finger_extended_radial_margin`: Radial margin for action gestures (brake/accel checks).
- `brake_palm_sideways_max_abs_z`: Palm sideways threshold (lower = stricter).
- `brake_fingers_toward_min_abs_x`: How strongly fingers must point toward center.
- `brake_fingers_extended_ratio`: Fraction of fingers that must look extended for brake.
- `camera_index`: Webcam index.
- `mirror_input`: Mirror the webcam feed for natural left/right control.
- `show_ui`: Show the preview window.
- `draw_landmarks`: Render hand landmarks in the preview.
- `show_fps`: Display FPS in the preview.
- `window_name`: Window title for the preview.
- `controller_backend`: Override control backend (`null` uses controls config).

### config/controls.yml

- `type`: Output type (`keyboard` or `gamepad`).
- `gamepad.backend`: Gamepad backend (`uinput` on Linux).
- `keyboard.backend`: Keyboard backend (`auto`, `uinput`, `pynput`).
- `keyboard.left_key`: Key for steering left.
- `keyboard.right_key`: Key for steering right.
- `keyboard.accel_key`: Key for accel.
- `keyboard.brake_key`: Key for brake.
- `keyboard.steer_threshold`: Steering threshold to press left/right.
- `keyboard.steer_hold_threshold`: Higher threshold to force a held left/right press.

## Notebooks

The `notebooks/` directory contains exploration and rule-based gesture analysis. These are optional and not required for running the app.
