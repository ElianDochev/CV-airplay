# Plan

## Stage 1 - Requirements and gestures
- Define gestures and priority rules (steer, accelerate, brake).
- Map hand states to control outputs.
- Decide confidence thresholds and fallback behavior.

## Stage 2 - Hand tracking pipeline
- Select/confirm MediaPipe or alternative hand tracking stack.
- Normalize landmarks and estimate steering rotation.
- Build gesture classifiers for grip, open palm, and index-point.

## Stage 3 - Controller abstraction
- Define a unified control state (steer, accel, brake).
- Enforce mutual exclusivity of accel vs brake.
- Support simultaneous steering with accel or brake.

## Stage 4 - Runtime app
- Implement real-time loop, smoothing, and debouncing.
- Add calibration and optional on-screen debug overlay.
- Provide CLI or config file for tuning.

## Stage 5 - Virtual controller integration
- Windows: vJoy/ViGEm or equivalent.
- Linux: uinput/evdev virtual gamepad.
- Keep overlay invisible and do not disable keyboard/mouse.

## Stage 6 - Containerization
- Docker Compose to run AI + virtual controller services.
- Document manual device assignment steps per OS.
- Validate latency and stability.

## Stage 7 - Tests and documentation
- Add unit tests for gesture logic.
- Add integration tests for control output.
- Update README with setup and usage.
