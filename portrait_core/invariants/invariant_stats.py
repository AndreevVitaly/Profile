"""Stability statistics for Profile geometric invariants."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Iterable

from portrait_core.archive.common import read_json, write_json
from portrait_core.invariants.invariant_export import invariant_set_from_dict
from portrait_core.invariants.invariant_models import InvariantSet, InvariantStats
from portrait_core.invariants.registry import INVARIANT_DEFINITIONS, STABILITY_THRESHOLDS, STATS_SCHEMA


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def stability_class(score: float | None, count: int, *, cv: float | None = None) -> str:
    if count < STABILITY_THRESHOLDS["minimum_count"] or score is None or cv is None:
        return "insufficient_data"
    if cv < STABILITY_THRESHOLDS["excellent_cv"]:
        return "excellent"
    if cv < STABILITY_THRESHOLDS["stable_cv"]:
        return "stable"
    if cv < STABILITY_THRESHOLDS["moderate_cv"]:
        return "moderate"
    return "unstable"


def stability_score_from_cv(cv: float | None, count: int) -> float | None:
    if cv is None or count < STABILITY_THRESHOLDS["minimum_count"]:
        return None
    return max(0.0, min(1.0, 1.0 - cv))


def _load_invariant_set(value: str | Path | InvariantSet | dict[str, Any]) -> InvariantSet:
    if isinstance(value, InvariantSet):
        return value
    if isinstance(value, dict):
        return invariant_set_from_dict(value)
    return invariant_set_from_dict(read_json(value))


def _valid_ratio_value(item: InvariantSet, name: str) -> float | None:
    ratio = item.ratios.get(name)
    if ratio is None or not ratio.valid or ratio.value is None:
        return None
    return ratio.value


def compute_invariant_stats(
    invariant_sets: Iterable[str | Path | InvariantSet | dict[str, Any]],
) -> dict[str, InvariantStats]:
    sets = [_load_invariant_set(item) for item in invariant_sets]
    ratio_names = sorted({definition.name for definition in INVARIANT_DEFINITIONS} | {name for item in sets for name in item.ratios})
    result: dict[str, InvariantStats] = {}

    for name in ratio_names:
        values = [value for item in sets if (value := _valid_ratio_value(item, name)) is not None]
        count_valid = len(values)
        count_total = len(sets)
        missing_count = count_total - count_valid
        if values:
            mean = statistics.fmean(values)
            median = statistics.median(values)
            variance = statistics.pvariance(values) if count_valid > 1 else 0.0
            std = variance**0.5
            cv = None if mean == 0 else abs(std / mean)
            mad = statistics.median([abs(value - median) for value in values])
            min_value = min(values)
            max_value = max(values)
        else:
            mean = median = variance = std = cv = mad = min_value = max_value = None
        score = stability_score_from_cv(cv, count_valid)
        result[name] = InvariantStats(
            ratio_name=name,
            mean=_round(mean),
            median=_round(median),
            std=_round(std),
            variance=_round(variance),
            cv=_round(cv),
            mad=_round(mad),
            min=_round(min_value),
            max=_round(max_value),
            count_total=count_total,
            count_valid=count_valid,
            missing_count=missing_count,
            stability_score=_round(score),
            stability_class=stability_class(score, count_valid, cv=cv),
        )
    return result


def build_invariant_stats(
    invariants_paths: Iterable[str | Path | InvariantSet | dict[str, Any]],
    output_path: str | Path | None = None,
    *,
    sample_name: str = "all_valid_pfr",
) -> dict[str, Any]:
    stats = compute_invariant_stats(invariants_paths)
    payload = {
        "schema": "profile.invariants.stats.v1",
        "schema_info": dict(STATS_SCHEMA),
        "sample": sample_name,
        "stats": {name: item.to_dict() for name, item in sorted(stats.items())},
        "stability_formula": "stability_score = clamp(1 - cv, 0, 1); classes use documented cv thresholds",
        "stability_thresholds": dict(STABILITY_THRESHOLDS),
        "scientific_status": {
            "computable_ratio": True,
            "candidate_invariant": True,
            "stable_within_dataset": "derived from stability_class per ratio",
            "stable_across_conditions": False,
            "validated_invariant": False,
        },
        "metadata": {
            "created_by": "portrait_core.invariants",
            "version": "0.2.0",
        },
    }
    if output_path is not None:
        write_json(output_path, payload)
    return payload