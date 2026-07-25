"""Data models for the Profile geometric invariant engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from portrait_core.invariants.registry import INVARIANT_SCHEMA


SCHEMA_VERSION = "profile.invariants.v1"
ENGINE_VERSION = "0.2.0"


@dataclass(frozen=True)
class RatioDefinition:
    name: str
    numerator: str
    denominator: str
    category: str


@dataclass
class InvariantRatio:
    name: str
    numerator: str
    denominator: str
    value: float | None
    category: str
    valid: bool = True
    source: str = "measurements"
    quality: str = "ok"
    skipped_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "value": self.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "category": self.category,
            "valid": self.valid,
            "source": self.source,
            "quality": self.quality,
        }
        if self.skipped_reason:
            payload["skipped_reason"] = self.skipped_reason
        if self.diagnostics:
            payload["diagnostics"] = dict(self.diagnostics)
        return payload


@dataclass
class InvariantSet:
    portrait_id: str | None
    dataset_id: str | None
    pfr_id: str | None
    pfr_uuid: str | None = None
    ratios: dict[str, InvariantRatio] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_info": dict(INVARIANT_SCHEMA),
            "portrait_id": self.portrait_id,
            "dataset_id": self.dataset_id,
            "pfr_id": self.pfr_id,
            "pfr_uuid": self.pfr_uuid,
            "source": self.source,
            "ratios": {
                name: ratio.to_dict()
                for name, ratio in sorted(self.ratios.items())
            },
            "quality": dict(self.quality),
            "diagnostics": dict(self.diagnostics),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass
class InvariantStats:
    ratio_name: str
    mean: float | None
    median: float | None
    std: float | None
    variance: float | None
    cv: float | None
    mad: float | None
    min: float | None
    max: float | None
    count_total: int
    count_valid: int
    missing_count: int
    stability_score: float | None
    stability_class: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ratio_name": self.ratio_name,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "variance": self.variance,
            "cv": self.cv,
            "mad": self.mad,
            "min": self.min,
            "max": self.max,
            "count": self.count_valid,
            "count_total": self.count_total,
            "count_valid": self.count_valid,
            "missing_count": self.missing_count,
            "count_missing": self.missing_count,
            "stability_score": self.stability_score,
            "stability_class": self.stability_class,
        }