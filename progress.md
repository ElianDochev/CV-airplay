# Progress Log

This document summarizes the calibration and controller changes attempted to fix behavior issues and improve reliability.

## Controller startup and toggling
- Changed the controller env var to only control startup state after calibration.
- Added SPACE toggle with a 3-second enable countdown and UI messaging.
- Allowed controller enablement based on controls config (keyboard/gamepad), not just controller_backend.
- Updated README to reflect new controller behavior and SPACE toggle.

## UI and Docker behavior
- Found controller UI was hidden in Docker because controller_backend was set to none.
- Switched controller_backend to null so it no longer disables controller output by default.

## Input issues and keyboard backend
- Confirmed accel/steer values were changing, but keys were not registering.
- Added uinput-based keyboard backend and auto-selection (uinput when available).
- Added keyboard backend selection to controls.yml.
- Added debug prints to report selected keyboard backend and keys.

## Steering stability and key hold
- Added steer_hold_threshold to treat large steering values as held key presses.
- Kept brake priority after smoothing (accel cleared when both are true).

## Calibration changes
- Added a new two-hand neutral calibration stage.
- Reordered calibration to learn brake/accel first, then neutral, then steering.
- Calculated two-hand max steer based on the explicit neutral stage.
- Excluded hands performing accel/brake gestures from steering calculation.
- Added warnings during neutral step when accel/brake gestures are detected.
- Prevented completing neutral stage if warning blocks sampling.

## Two-hand thumb-up rework
- Switched steering to two-hand wrist rotation only (no one-hand steering).
- Made accel default-on and brake trigger via index-pointing gesture.
- Replaced calibration stages with neutral, left/right steering, then brake in neutral/left/right.
- Added separate brake patterns for neutral/left/right to keep steering active while braking.
- Updated example calibration and README to match the new flow.

## Duplicates and cleanup
- Removed unused helper functions and unused state.
- Reduced duplicate calibration save logic.

## Current status
- Controller UI now shows enabled/disabled state.
- Keyboard input should use uinput in Docker when available.
- Calibration flow includes explicit neutral and warning when accel/brake gestures appear.

## Next checks
- Recalibrate after the new stage order and neutral step changes.
- Verify the warning appears only when accel/brake gestures are seen during neutral.
- Confirm uinput backend is selected in Docker and keys are held as expected.
