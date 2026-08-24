"""Dependency-free statistical helpers for the A/B runner."""

from __future__ import annotations

import math
import statistics
from typing import Iterable


def describe(values: Iterable[float]) -> dict:
    xs = [float(x) for x in values if not isinstance(x, bool) and math.isfinite(float(x))]
    keys = ("count", "mean", "median", "std", "variance", "cv", "mad", "min", "max", "p10", "p90")
    if not xs:
        return {key: (0 if key == "count" else None) for key in keys}
    xs.sort()
    mean = statistics.fmean(xs)
    variance = statistics.pvariance(xs)
    median = statistics.median(xs)

    def percentile(q: float) -> float:
        pos = (len(xs) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)

    return {"count": len(xs), "mean": mean, "median": median, "std": math.sqrt(variance),
            "variance": variance, "cv": math.sqrt(variance) / abs(mean) if mean else None,
            "mad": statistics.median(abs(x - median) for x in xs), "min": xs[0], "max": xs[-1],
            "p10": percentile(.1), "p90": percentile(.9)}


def relative_change(a: float | None, b: float | None) -> float | None:
    return None if a in (None, 0) or b is None else (b - a) / abs(a)
