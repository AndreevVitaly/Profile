"""CLI for canonical pseudo-3D reconstruction and projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portrait_core.archive.common import read_json, write_json
from portrait_core.reconstruction_3d.export import build_reconstruction
from portrait_core.reconstruction_3d.frame_selection import SelectionConfig
from portrait_core.reconstruction_3d.projection import PROJECTION_PRESETS, project_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ORION canonical relative-depth pseudo-3D research tools")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build canonical pseudo-3D from Dataset PFR files")
    build.add_argument("--dataset", required=True)
    build.add_argument("--scale-mode", choices=("unit_ipd", "unit_face_width"), default="unit_ipd")
    build.add_argument("--fusion-method", choices=("median", "weighted_mean"), default="median")
    build.add_argument("--min-frames", type=int, default=3)
    build.add_argument("--max-frames", type=int, default=21)
    build.add_argument("--max-frames-per-pose-bin", type=int, default=3)
    build.add_argument("--force-3d", action="store_true")
    build.add_argument("--no-projections", action="store_true")
    project = commands.add_parser("project", help="Project an existing canonical model")
    project.add_argument("--model", required=True)
    project.add_argument("--preset", choices=tuple(PROJECTION_PRESETS), default="frontal_orthographic")
    project.add_argument("--projection-type", choices=("orthographic", "perspective"), default="orthographic")
    project.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build":
        config = SelectionConfig(
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            max_frames_per_pose_bin=args.max_frames_per_pose_bin,
        )
        result = build_reconstruction(
            args.dataset, scale_mode=args.scale_mode, fusion_method=args.fusion_method,
            selection_config=config, generate_projections=not args.no_projections,
            force=args.force_3d,
        )
    else:
        model_path = Path(args.model)
        result = project_model(read_json(model_path), args.preset, projection_type=args.projection_type)
        output = Path(args.output) if args.output else model_path.parent / "projections" / f"{args.preset}.json"
        write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
