from __future__ import annotations

import argparse

from .gesture_controller import run_camera_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AirDrive gesture controller")
    parser.add_argument("--config", default=None, help="Path to config file")
    parser.add_argument("--controls-config", default=None, help="Path to controls config file")
    parser.add_argument("--camera", type=int, default=None, help="Camera index")
    parser.add_argument(
        "--controller",
        choices=["on", "off"],
        default="off",
        help="Start controller active after calibration",
    )
    parser.add_argument("--show-ui", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--draw-landmarks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mirror-input", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-fps", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_camera_loop(
        config_path=args.config,
        controls_config_path=args.controls_config,
        camera_index=args.camera,
        show_ui=args.show_ui,
        draw_landmarks=args.draw_landmarks,
        mirror_input=args.mirror_input,
        backend=None,
        show_fps=args.show_fps,
        controller_start_active=args.controller == "on",
    )


if __name__ == "__main__":
    main()
