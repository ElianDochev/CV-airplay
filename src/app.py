from __future__ import annotations

import argparse

from .gesture_controller import run_camera_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AirDrive gesture controller")
    parser.add_argument("--config", default=None, help="Path to config file")
    parser.add_argument("--camera", type=int, default=None, help="Camera index")
    parser.add_argument(
        "--backend",
        choices=["none", "uinput", "vgamepad"],
        default=None,
        help="Virtual controller backend",
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
        camera_index=args.camera,
        show_ui=args.show_ui,
        draw_landmarks=args.draw_landmarks,
        mirror_input=args.mirror_input,
        backend=args.backend,
        show_fps=args.show_fps,
    )


if __name__ == "__main__":
    main()
