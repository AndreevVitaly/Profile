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
        }
        result = run_profile_engine(args.dataset, config=config)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
