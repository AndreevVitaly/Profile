"""Profile Engine stages.

Stages orchestrate existing Profile components; they do not implement new face
analysis logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from portrait_core.archive.common import as_posix, read_json, write_json
from portrait_core.archive.dataset import write_dataset_files
from portrait_core.archive.validation import validate_dataset_archive
from portrait_core.invariants import build_invariant_stats, build_invariants_for_portrait
from portrait_core.lic_experiment import analyze_lic_stability
from portrait_core.lic_stability_report import build_lic_stability_report
from portrait_core.report_pack import build_report_pack, render_markdown
from profile_engine.context import ProfileEngineContext


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class StageFatalError(RuntimeError):
    """Raised when the engine cannot continue safely."""


class BaseStage:
    name = "base"

    def run(self, context: ProfileEngineContext) -> dict[str, Any]:
        raise NotImplementedError

    def result(
        self,
        status: str,
        *,
        actions: list[str] | None = None,
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
        stats: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": status,
            "actions": actions or [],
            "warnings": warnings or [],
            "errors": errors or [],
            "stats": stats or {},
            "artifacts": artifacts or [],
        }


class ValidateDatasetStage(BaseStage):
    name = "validate_dataset"

    def run(self, context: ProfileEngineContext) -> dict[str, Any]:
        warnings: list[str] = []
        errors: list[str] = []
        dataset_path = context.dataset_path

        if not dataset_path.exists() or not dataset_path.is_dir():
            errors.append(f"dataset directory not found: {dataset_path}")
        for key in ("dataset_json", "images_dir", "pfr_dir"):
            path = context.paths[key]
            if not path.exists():
                errors.append(f"required path missing: {context.relative_path(path)}")

        dataset = None
        if context.paths["dataset_json"].exists():
            try:
                dataset = context.read_dataset()
            except Exception as error:  # noqa: BLE001
                errors.append(f"dataset.json is not readable JSON: {error}")
        if dataset:
            context.dataset_id = dataset.get("id")

        validation = validate_dataset_archive(dataset_path)
        validation_errors = validation.get("errors") or []
        warnings.extend(validation.get("warnings") or [])
        warnings.extend(f"dataset validation warning: {error}" for error in validation_errors)

        for warning in warnings:
            context.add_warning(warning)
        for error in errors:
            context.add_error(error)

        if errors:
            result = self.result("failed", warnings=warnings, errors=errors)
            raise StageFatalError(str(result))
        return self.result(
            "completed",
            actions=["dataset archive structurally validated"],
            warnings=warnings,
            stats={
                "items_checked": validation.get("items_checked", 0),
                "validator_valid": validation.get("valid"),
                "validator_errors": len(validation_errors),
            },
        )


class EnsurePFRStage(BaseStage):
    name = "ensure_pfr"

    def run(self, context: ProfileEngineContext) -> dict[str, Any]:
        if context.config.get("skip_pfr"):
            return self.result("skipped", actions=["skip_pfr enabled"])

        context.paths["pfr_dir"].mkdir(parents=True, exist_ok=True)
        dataset = context.read_dataset()
        items = list(dataset.get("items") or [])
        if not items:
            items = self._items_from_images(context)
            dataset["items"] = items

        created = 0
        skipped = 0
        warnings: list[str] = []
        force = bool(context.config.get("force_pfr"))

        for item in items:
            image_path = self._resolve_item_image(context, item)
            if image_path is None:
                warnings.append(f"item has no image_path: {item}")
                continue
            pfr_path = self._pfr_path(context, item, image_path)
            if pfr_path.exists() and not force:
                skipped += 1
                item["pfr_path"] = as_posix(pfr_path, context.dataset_path)
                continue
            if context.config.get("dry_run"):
                skipped += 1
                continue
            if not image_path.exists():
                warnings.append(f"image not found: {context.relative_path(image_path)}")
                continue

            from portrait_core import create_portrait_report

            report = create_portrait_report(
                str(image_path),
                backend=context.config.get("backend", "mediapipe"),
                model_path=context.config.get("model_path"),
                topology_path=context.config.get("topology_path"),
                input_metadata={
                    "dataset_id": context.dataset_id,
                    "source_frame": image_path.name,
                },
            )
            write_json(pfr_path, report)
            item.update(
                {
                    "pfr_id": report.get("id"),
                    "pfr_uuid": report.get("uuid"),
                    "pfr_path": as_posix(pfr_path, context.dataset_path),
                    "status": report.get("quality", {}).get("status", "warning"),
                    "issues": report.get("quality", {}).get("issues", []),
                }
            )
            context.add_artifact("pfr", pfr_path)
            created += 1

        if not context.config.get("dry_run"):
            write_dataset_files(context.dataset_path, dataset)

        for warning in warnings:
            context.add_warning(warning)
        return self.result(
            "completed",
            actions=["PFR checked"],
            warnings=warnings,
            stats={"created": created, "skipped_existing": skipped},
        )

    def _items_from_images(self, context: ProfileEngineContext) -> list[dict[str, Any]]:
        items = []
        for image_path in sorted(context.paths["images_dir"].iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            items.append(
                {
                    "image_path": as_posix(image_path, context.dataset_path),
                    "pfr_path": None,
                    "status": "pending",
                    "issues": [],
                }
            )
        return items

    def _resolve_item_image(
        self,
        context: ProfileEngineContext,
        item: dict[str, Any],
    ) -> Path | None:
        image_path = item.get("image_path")
        if not image_path:
            return None
        path = Path(str(image_path))
        if not path.is_absolute():
            path = context.dataset_path / path
        return path

    def _pfr_path(
        self,
        context: ProfileEngineContext,
        item: dict[str, Any],
        image_path: Path,
    ) -> Path:
        current = item.get("pfr_path")
        if current:
            path = Path(str(current))
            return path if path.is_absolute() else context.dataset_path / path
        return context.paths["pfr_dir"] / f"{image_path.stem}_portrait.json"


class BuildInvariantsStage(BaseStage):
    name = "build_invariants"

    def run(self, context: ProfileEngineContext) -> dict[str, Any]:
        if context.config.get("skip_invariants"):
            return self.result("skipped", actions=["skip_invariants enabled"])

        context.paths["invariants_dir"].mkdir(parents=True, exist_ok=True)
        dataset = context.read_dataset()
        items = list(dataset.get("items") or [])
        created = 0
        skipped_existing = 0
        failed = 0
        processed = 0
        warnings: list[str] = []
        force = bool(context.config.get("force_invariants"))

        for item in items:
            pfr_value = item.get("pfr_path")
            if not pfr_value:
                if item.get("status") == "rejected":
                    continue
                warnings.append(f"item without pfr_path skipped: {item.get('image_path')}")
                continue
            pfr_path = _resolve_dataset_path(context, pfr_value)
            if not pfr_path.exists():
                warnings.append(f"pfr_path missing: {context.relative_path(pfr_path)}")
                failed += 1
                continue
            processed += 1
            output_path = context.paths["invariants_dir"] / f"{pfr_path.stem}_invariants.json"
            relative_output = as_posix(output_path, context.dataset_path)

            if output_path.exists() and not force:
                try:
                    payload = read_json(output_path)
                    if payload.get("schema") != "profile.invariants.v1":
                        warnings.append(f"incompatible invariants schema skipped: {relative_output}")
                        failed += 1
                        continue
                    item["invariants_path"] = relative_output
                    skipped_existing += 1
                    continue
                except Exception as error:  # noqa: BLE001
                    warnings.append(f"corrupted invariants file: {relative_output}: {error}")
                    failed += 1
                    continue

            if context.config.get("dry_run"):
                skipped_existing += 1
                continue
            try:
                build_invariants_for_portrait(pfr_path, output_path)
                item["invariants_path"] = relative_output
                context.add_artifact("invariants", output_path)
                created += 1
            except Exception as error:  # noqa: BLE001
                warnings.append(f"invariants failed for {context.relative_path(pfr_path)}: {error}")
                failed += 1

        if processed == 0:
            warnings.append("no valid PFR files found for invariant build")
        if not context.config.get("dry_run"):
            write_dataset_files(context.dataset_path, dataset)
        for warning in warnings:
            context.add_warning(warning)
        return self.result(
            "completed" if failed == 0 else "warning",
            actions=["invariants checked", "dataset.json invariants_path updated"],
            warnings=warnings,
            stats={
                "processed": processed,
                "created": created,
                "skipped_existing": skipped_existing,
                "failed": failed,
            },
        )


class BuildInvariantStatsStage(BaseStage):
    name = "build_invariant_stats"

    def run(self, context: ProfileEngineContext) -> dict[str, Any]:
        if context.config.get("skip_invariants"):
            return self.result("skipped", actions=["skip_invariants enabled"])

        entries = _invariant_entries(context)
        if not entries:
            warning = "no invariants files found for stats"
            context.add_warning(warning)
            return self.result("skipped", warnings=[warning])

        samples = {
            "all_valid_pfr": [entry for entry in entries if entry["status"] in {"passed", "warning"}],
            "passed_only": [entry for entry in entries if entry["status"] == "passed"],
            "passed_and_warning": [entry for entry in entries if entry["status"] in {"passed", "warning"}],
        }
        stats_by_sample: dict[str, Any] = {}
        warnings: list[str] = []
        for name, sample_entries in samples.items():
            if not sample_entries:
                warnings.append(f"stats sample empty: {name}")
                continue
            stats_by_sample[name] = build_invariant_stats(
                [entry["path"] for entry in sample_entries],
                sample_name=name,
            )

        output_path = context.dataset_path / "invariant_stats.json"
        legacy_path = context.paths["invariants_dir"] / "stats.json"
        payload = {
            "schema": "profile.invariants.dataset_stats.v1",
            "dataset_id": context.dataset_id,
            "samples": stats_by_sample,
            "quality_samples": {name: len(value) for name, value in samples.items()},
            "warnings": warnings,
        }
        if not context.config.get("dry_run"):
            write_json(output_path, payload)
            write_json(legacy_path, stats_by_sample.get("all_valid_pfr") or payload)
            context.add_artifact("invariant_stats", output_path)
            context.add_artifact("invariant_stats_legacy", legacy_path)
        for warning in warnings:
            context.add_warning(warning)
        ratios_analyzed = len((stats_by_sample.get("all_valid_pfr") or {}).get("stats", {}))
        return self.result(
            "completed" if stats_by_sample else "warning",
            actions=["invariant stats built"],
            warnings=warnings,
            stats={
                "input_files": len(entries),
                "ratios_total": ratios_analyzed,
                "ratios_analyzed": ratios_analyzed,
                "samples": {name: len(value) for name, value in samples.items()},
            },
            artifacts=[context.relative_path(output_path)],
        )


class InvariantReadinessStage(BaseStage):
    name = "build_invariant_readiness"

    def run(self, context: ProfileEngineContext) -> dict[str, Any]:
        entries = _invariant_entries(context)
        dataset = context.read_dataset()
        items = list(dataset.get("items") or [])
        stats_path = context.dataset_path / "invariant_stats.json"
        stats_payload = read_json(stats_path) if stats_path.exists() else {}
        all_stats = ((stats_payload.get("samples") or {}).get("all_valid_pfr") or {}).get("stats") or {}
        ratios_available = sorted(
            name for name, stat in all_stats.items() if stat.get("count_valid", stat.get("count", 0)) > 0
        )
        ratios_with_sufficient_samples = sorted(
            name for name, stat in all_stats.items() if stat.get("count_valid", stat.get("count", 0)) >= 3
        )
        quality_distribution: dict[str, int] = {}
        for item in items:
            status = str(item.get("status") or "unknown")
            quality_distribution[status] = quality_distribution.get(status, 0) + 1

        reasons: list[str] = []
        limitations: list[str] = [
            "single dataset can show within-series stability only",
            "no claim of universal validated facial invariants is made",
        ]
        if not entries:
            reasons.append("no invariants files available")
        if len(entries) < 3:
            reasons.append("fewer than 3 invariant observations")
        if not ratios_with_sufficient_samples:
            reasons.append("no ratio has sufficient samples for stability statistics")

        if not entries or not ratios_with_sufficient_samples:
            status = "not_ready"
        elif reasons or quality_distribution.get("warning", 0):
            status = "partially_ready"
        else:
            status = "ready"

        payload = {
            "schema": "profile.invariant_readiness.v1",
            "dataset_id": context.dataset_id,
            "pfr_total": sum(1 for item in items if item.get("pfr_path")),
            "pfr_processed": len(entries),
            "pfr_failed": sum(1 for item in items if item.get("pfr_path") and not item.get("invariants_path")),
            "invariants_created": len(entries),
            "ratios_available": ratios_available,
            "ratios_with_sufficient_samples": ratios_with_sufficient_samples,
            "quality_distribution": quality_distribution,
            "research_readiness": {
                "status": status,
                "reasons": reasons,
                "limitations": limitations,
            },
        }
        output_path = context.dataset_path / "invariant_readiness.json"
        if not context.config.get("dry_run"):
            write_json(output_path, payload)
            context.add_artifact("invariant_readiness", output_path)
        return self.result(
            "completed",
            actions=["invariant readiness report built"],
            stats={
                "status": status,
                "pfr_processed": len(entries),
                "ratios_available": len(ratios_available),
                "ratios_with_sufficient_samples": len(ratios_with_sufficient_samples),
            },
            artifacts=[context.relative_path(output_path)],
        )

class LICStage(BaseStage):
    name = "lic"

    def run(self, context: ProfileEngineContext) -> dict[str, Any]:
        if context.config.get("skip_lic"):
            return self.result("skipped", actions=["skip_lic enabled"])
        if not _pfr_paths(context):
            warning = "no PFR files found for LIC stage"
            context.add_warning(warning)
            return self.result("skipped", warnings=[warning])

        context.paths["experiments_dir"].mkdir(parents=True, exist_ok=True)
        lic_path = context.paths["experiments_dir"] / "lic_stability.json"
        points_path = context.paths["experiments_dir"] / "lic_point_stability.json"
        if not context.config.get("dry_run"):
            write_json(lic_path, analyze_lic_stability(str(context.dataset_path)))
            write_json(points_path, build_lic_stability_report(str(context.dataset_path)))
            context.add_artifact("lic_stability", lic_path)
            context.add_artifact("lic_point_stability", points_path)
        return self.result(
            "completed",
            actions=["LIC stability reports built"],
            artifacts=[context.relative_path(lic_path), context.relative_path(points_path)],
        )


class ReportPackStage(BaseStage):
    name = "report_pack"

    def run(self, context: ProfileEngineContext) -> dict[str, Any]:
        if context.config.get("skip_report_pack"):
            return self.result("skipped", actions=["skip_report_pack enabled"])
        if not _pfr_paths(context):
            warning = "no PFR files found for report pack"
            context.add_warning(warning)
            return self.result("skipped", warnings=[warning])

        context.paths["report_pack_dir"].mkdir(parents=True, exist_ok=True)
        output_path = context.paths["report_pack_dir"] / "report_pack.json"
        markdown_path = context.paths["report_pack_dir"] / "report_pack.md"
        if not context.config.get("dry_run"):
            pack = build_report_pack(str(context.dataset_path), dataset_id=context.dataset_id)
            write_json(output_path, pack)
            markdown_path.write_text(render_markdown(pack), encoding="utf-8")
            context.add_artifact("report_pack", output_path)
            context.add_artifact("report_pack_markdown", markdown_path)
        return self.result(
            "completed",
            actions=["report pack built"],
            artifacts=[context.relative_path(output_path), context.relative_path(markdown_path)],
        )


def default_stages(*, include_reconstruction_3d: bool = False) -> list[BaseStage]:
    stages: list[BaseStage] = [
        ValidateDatasetStage(),
        EnsurePFRStage(),
        BuildInvariantsStage(),
        BuildInvariantStatsStage(),
        InvariantReadinessStage(),
        LICStage(),
        ReportPackStage(),
    ]
    if include_reconstruction_3d:
        from profile_engine.reconstruction_stages import reconstruction_stages

        stages.extend(reconstruction_stages())
    return stages


def _pfr_paths(context: ProfileEngineContext) -> list[Path]:
    paths: list[Path] = []
    try:
        dataset = context.read_dataset()
    except Exception:  # noqa: BLE001
        dataset = {}
    for item in dataset.get("items") or []:
        pfr_path = item.get("pfr_path")
        if not pfr_path:
            continue
        path = Path(str(pfr_path))
        path = path if path.is_absolute() else context.dataset_path / path
        if path.exists():
            paths.append(path)
    if not paths and context.paths["pfr_dir"].exists():
        paths = sorted(context.paths["pfr_dir"].glob("*_portrait.json"))
    return sorted(dict.fromkeys(paths))


def _invariant_paths(context: ProfileEngineContext) -> list[Path]:
    if not context.paths["invariants_dir"].exists():
        return []
    return sorted(
        path
        for path in context.paths["invariants_dir"].glob("*.json")
        if path.name != "stats.json"
    )


def _resolve_dataset_path(context: ProfileEngineContext, value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else context.dataset_path / path


def _invariant_entries(context: ProfileEngineContext) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        dataset = context.read_dataset()
    except Exception:  # noqa: BLE001
        dataset = {}
    for item in dataset.get("items") or []:
        invariant_path = item.get("invariants_path")
        if not invariant_path:
            continue
        path = _resolve_dataset_path(context, invariant_path)
        if path.exists():
            entries.append({"path": path, "item": item, "status": item.get("status") or "warning"})
    if entries:
        return entries
    return [
        {"path": path, "item": {}, "status": "warning"}
        for path in _invariant_paths(context)
    ]