"""CLI for Profile Engine."""

from __future__ import annotations

import argparse
import json

from profile_engine.runner import run_profile_engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Engine coordinator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run Profile Engine for a DS-* archive")
    run.add_argument("--dataset", required=True, help="Path to DS-* archive")
    run.add_argument("--force-pfr", action="store_true")
    run.add_argument("--force-invariants", action="store_true")
    run.add_argument("--stages", help="Comma-separated stages, e.g. invariants,invariant_stats")
    run.add_argument("--skip-pfr", action="store_true")
    run.add_argument("--skip-invariants", action="store_true")
    run.add_argument("--skip-lic", action="store_true")
    run.add_argument("--skip-report-pack", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--backend", default="mediapipe")
    run.add_argument("--model", dest="model_path")
    run.add_argument("--topology", dest="topology_path")
    run.add_argument("--force-3d", action="store_true")
    run.add_argument("--min-3d-frames", type=int, default=3)
    run.add_argument("--max-3d-frames", type=int, default=21)
    run.add_argument("--max-frames-per-pose-bin", type=int, default=3)
    run.add_argument("--scale-mode", choices=("unit_ipd", "unit_face_width"), default="unit_ipd")
    run.add_argument("--fusion-method", choices=("median", "weighted_mean"), default="median")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        config = {
            "force_pfr": args.force_pfr,
            "force_invariants": args.force_invariants,
            "stages": args.stages,
            "skip_pfr": args.skip_pfr,
            "skip_invariants": args.skip_invariants,
            "skip_lic": args.skip_lic,
            "skip_report_pack": args.skip_report_pack,
            "dry_run": args.dry_run,
            "backend": args.backend,
            "model_path": args.model_path,
            "topology_path": args.topology_path,
            "force_3d": args.force_3d,
            "min_3d_frames": args.min_3d_frames,
            "max_3d_frames": args.max_3d_frames,
            "max_frames_per_pose_bin": args.max_frames_per_pose_bin,
            "scale_mode": args.scale_mode,
            "fusion_method": args.fusion_method,
        }
        result = run_profile_engine(args.dataset, config=config)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
