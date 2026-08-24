"""CLI entrypoint for Dataset Builder."""

import argparse
import json
import sys

from apps.dataset_builder.builder import main as build_main
from apps.dataset_builder.preflight import run_preflight


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "preflight":
        parser = argparse.ArgumentParser(description="ORION Dataset Builder preflight")
        parser.add_argument("command")
        parser.add_argument("input_path")
        parser.add_argument("--output", default="dataset")
        parser.add_argument("--backend", choices=("mediapipe", "onnx"), default="mediapipe")
        parser.add_argument("--model", dest="model_path")
        parser.add_argument("--topology", dest="topology_path")
        args = parser.parse_args()
        report = run_preflight(args.input_path, args.output, backend=args.backend, model_path=args.model_path, topology_path=args.topology_path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["status"] != "ready":
            raise SystemExit(2)
        return
    build_main()


if __name__ == "__main__":
    main()
