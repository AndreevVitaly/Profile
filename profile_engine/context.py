"""Runtime context for the Profile Engine coordinator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from portrait_core.archive.common import current_utc_iso, make_record_id, read_json


@dataclass
class ProfileEngineContext:
    dataset_path: Path
    config: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: make_record_id("PER"))
    started_at: str = field(default_factory=current_utc_iso)
    dataset_id: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        self.paths = {
            "dataset_json": self.dataset_path / "dataset.json",
            "images_dir": self.dataset_path / "images",
            "pfr_dir": self.dataset_path / "pfr",
            "invariants_dir": self.dataset_path / "invariants",
            "experiments_dir": self.dataset_path / "experiments",
            "report_pack_dir": self.dataset_path / "experiments",
            "reconstruction_3d_dir": self.dataset_path / "reconstruction_3d",
        }
        if self.paths["dataset_json"].exists():
            try:
                dataset = read_json(self.paths["dataset_json"])
                self.dataset_id = dataset.get("id")
            except Exception as error:  # noqa: BLE001 - manifest should record it.
                self.add_warning(f"dataset.json could not be read during context init: {error}")

    def add_warning(self, message: str) -> None:
        self.warnings.append(str(message))

    def add_error(self, message: str) -> None:
        self.errors.append(str(message))

    def add_artifact(
        self,
        name: str,
        path: str | Path,
        *,
        kind: str = "file",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        artifact = {
            "name": name,
            "path": self.relative_path(path),
            "kind": kind,
        }
        if metadata:
            artifact["metadata"] = metadata
        self.artifacts.append(artifact)

    def add_stage_result(self, result: dict[str, Any]) -> None:
        self.stages.append(result)

    def relative_path(self, path: str | Path) -> str:
        value = Path(path)
        try:
            value = value.relative_to(self.dataset_path)
        except ValueError:
            pass
        return value.as_posix()

    def read_dataset(self) -> dict[str, Any]:
        dataset = read_json(self.paths["dataset_json"])
        self.dataset_id = dataset.get("id") or self.dataset_id
        return dataset
