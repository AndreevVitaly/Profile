"""JSON export helpers for Profile geometric invariants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from portrait_core.archive.common import as_posix, read_json, write_json
from portrait_core.invariants.invariant_models import InvariantSet
from portrait_core.invariants.ratio_engine import build_invariant_set_from_pfr


def default_invariants_path(portrait_json_path: str | Path) -> Path:
    pfr_path = Path(portrait_json_path)
    if pfr_path.parent.name == "pfr" and (pfr_path.parent.parent / "dataset.json").exists():
        stem = "invariants" if pfr_path.stem == "portrait" else f"{pfr_path.stem}_invariants"
        return pfr_path.parent.parent / "invariants" / f"{stem}.json"
    return pfr_path.with_name("invariants.json")


def invariant_set_from_dict(payload: dict[str, Any]) -> InvariantSet:
    from portrait_core.invariants.invariant_models import InvariantRatio

    ratios = {
        name: InvariantRatio(
            name=name,
            numerator=ratio["numerator"],
            denominator=ratio["denominator"],
            value=None if ratio.get("value") is None else float(ratio["value"]),
            category=ratio["category"],
            valid=bool(ratio.get("valid", True)),
            source=ratio.get("source", "measurements"),
            quality=ratio.get("quality", "ok"),
            skipped_reason=ratio.get("skipped_reason"),
            diagnostics=dict(ratio.get("diagnostics") or {}),
        )
        for name, ratio in (payload.get("ratios") or {}).items()
    }
    return InvariantSet(
        portrait_id=payload.get("portrait_id"),
        dataset_id=payload.get("dataset_id"),
        pfr_id=payload.get("pfr_id"),
        pfr_uuid=payload.get("pfr_uuid"),
        ratios=ratios,
        quality=dict(payload.get("quality") or {}),
        diagnostics=dict(payload.get("diagnostics") or {}),
        warnings=list(payload.get("warnings") or []),
        source=dict(payload.get("source") or {}),
        metadata=dict(payload.get("metadata") or {}),
        schema=payload.get("schema", "profile.invariants.v1"),
    )


def save_invariant_set(invariant_set: InvariantSet, output_path: str | Path) -> Path:
    target = Path(output_path)
    write_json(target, invariant_set.to_dict())
    return target


def build_invariants_for_portrait(
    portrait_json_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    pfr_path = Path(portrait_json_path)
    pfr = read_json(pfr_path)
    target = Path(output_path) if output_path is not None else default_invariants_path(pfr_path)
    source_base = pfr_path.parent.parent if pfr_path.parent.name == "pfr" else target.parent if target.parent.exists() else pfr_path.parent
    relative_pfr = as_posix(pfr_path, source_base)
    invariant_set = build_invariant_set_from_pfr(
        pfr,
        source={"portrait_json": relative_pfr, "source_pfr": relative_pfr},
    )
    save_invariant_set(invariant_set, target)
    return invariant_set.to_dict()