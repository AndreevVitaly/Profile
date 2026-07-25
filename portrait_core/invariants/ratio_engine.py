"""Ratio engine for geometric invariant candidates."""

from __future__ import annotations

import math
from typing import Any

from portrait_core.invariants.invariant_models import ENGINE_VERSION, InvariantRatio, InvariantSet
from portrait_core.invariants.registry import INVARIANT_DEFINITIONS, MEASUREMENT_ALIASES


RATIO_DEFINITIONS = INVARIANT_DEFINITIONS


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _get_path(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = mapping
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _unit_for_path(measurements: dict[str, Any], path: tuple[str, ...]) -> str | None:
    if not path:
        return None
    parent = _get_path(measurements, path[:-1]) if len(path) > 1 else measurements
    if not isinstance(parent, dict):
        return None
    units = parent.get("units") or parent.get("unit")
    if isinstance(units, dict):
        value = units.get(path[-1])
        return str(value) if value else None
    if isinstance(units, str):
        return units
    entry = parent.get(path[-1])
    if isinstance(entry, dict):
        value = entry.get("unit") or entry.get("units")
        return str(value) if value else None
    return None


def resolve_measurement(measurements: dict[str, Any], name: str) -> tuple[float | None, dict[str, Any]]:
    aliases = MEASUREMENT_ALIASES.get(name, ((name,),))
    for path in aliases:
        raw_value = _get_path(measurements, path)
        value = _as_number(raw_value)
        if value is not None:
            return value, {
                "canonical_name": name,
                "resolved_path": ".".join(path),
                "alias_used": path != aliases[0],
                "unit": _unit_for_path(measurements, path),
            }
    return None, {
        "canonical_name": name,
        "aliases_checked": [".".join(path) for path in aliases],
    }


def _invalid_ratio(definition, reason: str, diagnostics: dict[str, Any]) -> InvariantRatio:
    return InvariantRatio(
        name=definition.name,
        numerator=definition.numerator,
        denominator=definition.denominator,
        value=None,
        category=definition.category,
        valid=False,
        quality="skipped",
        skipped_reason=reason,
        diagnostics=diagnostics,
    )


def _units_compatible(numerator: dict[str, Any], denominator: dict[str, Any]) -> bool:
    num_unit = numerator.get("unit")
    den_unit = denominator.get("unit")
    return not num_unit or not den_unit or num_unit == den_unit


def build_invariant_set_from_pfr(
    pfr: dict[str, Any],
    *,
    source: dict[str, Any] | None = None,
) -> InvariantSet:
    measurements = pfr.get("measurements") or {}
    warnings: list[str] = []
    ratios: dict[str, InvariantRatio] = {}
    alias_diagnostics: list[dict[str, Any]] = []
    computed_count = 0
    skipped_count = 0

    for definition in RATIO_DEFINITIONS:
        numerator, numerator_diag = resolve_measurement(measurements, definition.numerator)
        denominator, denominator_diag = resolve_measurement(measurements, definition.denominator)
        diagnostics = {
            "numerator": numerator_diag,
            "denominator": denominator_diag,
            "description": definition.description,
        }
        if numerator_diag.get("alias_used"):
            alias_diagnostics.append({"measurement": definition.numerator, **numerator_diag})
        if denominator_diag.get("alias_used"):
            alias_diagnostics.append({"measurement": definition.denominator, **denominator_diag})

        reason = None
        if numerator is None:
            reason = f"missing numerator {definition.numerator}"
        elif denominator is None:
            reason = f"missing denominator {definition.denominator}"
        elif denominator == 0:
            reason = f"zero denominator {definition.denominator}"
        elif not _units_compatible(numerator_diag, denominator_diag):
            reason = "incompatible units"

        if reason:
            warnings.append(f"{definition.name}: {reason}")
            ratios[definition.name] = _invalid_ratio(definition, reason, diagnostics)
            skipped_count += 1
            continue

        value = numerator / denominator
        if not math.isfinite(value):
            reason = "non-finite ratio value"
            warnings.append(f"{definition.name}: {reason}")
            ratios[definition.name] = _invalid_ratio(definition, reason, diagnostics)
            skipped_count += 1
            continue

        ratios[definition.name] = InvariantRatio(
            name=definition.name,
            numerator=definition.numerator,
            denominator=definition.denominator,
            value=round(value, 6),
            category=definition.category,
            valid=True,
            diagnostics=diagnostics,
        )
        computed_count += 1

    pfr_id = pfr.get("id") or pfr.get("metadata", {}).get("pfr_id")
    quality = pfr.get("quality") or {}
    return InvariantSet(
        portrait_id=pfr.get("portrait_id") or pfr_id,
        dataset_id=pfr.get("dataset_id") or pfr.get("metadata", {}).get("dataset_id"),
        pfr_id=pfr_id,
        pfr_uuid=pfr.get("uuid") or pfr.get("metadata", {}).get("pfr_uuid"),
        ratios=ratios,
        quality={
            "source_status": quality.get("status", "warning"),
            "source_issues": list(quality.get("issues") or []),
        },
        diagnostics={
            "computed_count": computed_count,
            "skipped_count": skipped_count,
            "warnings": list(warnings),
            "aliases": alias_diagnostics,
        },
        warnings=warnings,
        source=source or {},
        metadata={
            "created_by": "portrait_core.invariants",
            "version": ENGINE_VERSION,
            "scientific_status": "computable ratio candidate; not a validated universal invariant",
        },
    )