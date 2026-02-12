# AirDrive — Vision Racing Controller (OpenCV + MediaPipe)

Turn two-hand gestures into a racing controller (steering + throttle + brake) in real time. The current implementation is a rule-based MediaPipe hand landmark pipeline with optional virtual controller backends.

## What It Does

- Tracks up to two hands from a webcam.
- Computes steering angle from one or two hands.
- Uses simple gesture rules for throttle and brake.
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
- `--controller on|off`: enable or disable virtual controller output (auto-selects uinput on Linux or vgamepad on Windows).
- `--show-ui/--no-show-ui`: enable or disable the on-screen preview window.
- `--draw-landmarks/--no-draw-landmarks`: render MediaPipe hand landmarks on the preview.
- `--mirror-input/--no-mirror-input`: mirror the webcam feed for natural left/right control.
- `--show-fps/--no-show-fps`: display FPS on the preview.

Controls while running:

- `c` clears calibration and starts the guided calibration flow (controller output disabled while calibrating)
- two thumbs up (both hands) auto-calibrates neutral steering (only when no calibration file is loaded)
- `q` quits

Calibration UI:

- Shows a 3-2-1-GO countdown before each capture stage
- Shows only the current calibration action (steer left/right, brake, accel)
- Stages: two-hand left, one-hand left, two-hand right, one-hand right, left brake, right brake, left accel, right accel

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

- Linux requires `/dev/video0` and `/dev/uinput` access.
- You may need `sudo modprobe uinput` on the host.
- The container runs headless by default with `--no-show-ui`.

Windows note: WSL2 + Docker Desktop typically uses WSLg for GUI apps, so `xhost` is not required. If you are running an X server (VcXsrv/Xming), allow local connections in that server's settings.

## Config

The main settings live in `config/main.yml` and control detection thresholds, UI flags, and camera defaults. Control mappings are in `config/controls.yml`. Steering range and preferred hands now come from the calibration file in `calibration/calibration.json`.

## Notebooks

The `notebooks/` directory contains exploration and rule-based gesture analysis. These are optional and not required for running the app.
