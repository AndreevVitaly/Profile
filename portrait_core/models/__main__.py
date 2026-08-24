from __future__ import annotations

import argparse
import json

from .manager import ModelManager
from .registry import MODEL_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description="ORION model diagnostics")
    parser.add_argument("command", choices=("list", "validate", "install"))
    parser.add_argument("model_id", nargs="?", default="mediapipe_face_landmarker")
    parser.add_argument("--model")
    args = parser.parse_args()
    manager = ModelManager()
    if args.command == "install":
        result = manager.install(args.model_id, destination=args.model)
        print(json.dumps({"schema": "orion.models.v1", "models": [result.to_dict(project_root=manager.project_root)]}, ensure_ascii=False, indent=2))
        if not result.valid:
            raise SystemExit(2)
        return
    results = []
    for model_id in MODEL_REGISTRY:
        resolved = manager.resolve(model_id, explicit_path=args.model)
        checked = manager.validate(resolved, initialize_backend=args.command == "validate")
        results.append(checked.to_dict(project_root=manager.project_root))
    print(json.dumps({"schema": "orion.models.v1", "models": results}, ensure_ascii=False, indent=2))
    if args.command == "validate" and not all(item["valid"] for item in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
