# AirDrive — Vision Racing Controller (OpenCV + MediaPipe)

Turn two-hand gestures into a racing controller (steering + throttle + brake) in real time. The current implementation is a rule-based MediaPipe hand landmark pipeline with optional virtual controller backends.

## What It Does

- Tracks up to two hands from a webcam.
- Computes steering angle from one or two hands.
- Uses thumb-up wrist rotation for steering and index-pointing for braking.
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
- Stages: two-hand neutral (thumbs up), two-hand left, two-hand right, brake in neutral, brake while left, brake while right

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

## Config

The main settings live in `config/main.yml` and control detection thresholds, UI flags, and camera defaults. Control mappings are in `config/controls.yml`. Steering range and preferred hands now come from the calibration file in `calibration/calibration.yml`.

To switch between keyboard and joystick output, edit `config/controls.yml`:

- `type: keyboard` for keyboard output
- `type: gamepad` for joystick output (uses the `gamepad.backend` setting)

If you do not want to run the calibration flow yourself, copy `example-calibration/calibration.yml` into the `calibration/` directory. The defaults in that file assume: two thumbs up for neutral, wrist rotation for steering, and index pointing for brake (accel stays on unless braking).

## Notebooks

The `notebooks/` directory contains exploration and rule-based gesture analysis. These are optional and not required for running the app.
