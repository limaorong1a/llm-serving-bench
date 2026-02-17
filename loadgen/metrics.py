from __future__ import annotations

from dataclasses import dataclass


def percentile(sorted_vals: list[float], p: float) -> float | None:
    """Percentile with linear interpolation (expects sorted input)."""
    if not sorted_vals:
        return None
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    n = len(sorted_vals)
    pos = (p / 100.0) * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


@dataclass(frozen=True)
class LatencySummary:
    mean_s: float | None
    p50_s: float | None
    p95_s: float | None
    p99_s: float | None
    min_s: float | None
    max_s: float | None


def summarize_latencies(latencies: list[float]) -> LatencySummary:
    if not latencies:
        return LatencySummary(None, None, None, None, None, None)
    lats_sorted = sorted(latencies)
    mean_s = sum(latencies) / len(latencies)
    return LatencySummary(
        mean_s=mean_s,
        p50_s=percentile(lats_sorted, 50),
        p95_s=percentile(lats_sorted, 95),
        p99_s=percentile(lats_sorted, 99),
        min_s=min(latencies),
        max_s=max(latencies),
    )


def tokens_per_second(total_tokens: int, wall_time_s: float) -> float | None:
    if wall_time_s <= 0 or total_tokens <= 0:
        return None
    return total_tokens / wall_time_s
