"""Spatial statistics with conservative, explicit interpretations."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .fourier import polygon_mask
from .models import ROIStatistics, StatisticalResult


def spatial_statistics(
    timepoint: int,
    centroids_rc: NDArray[np.floating],
    values: NDArray[np.generic],
    image_height: int,
    bin_size: int,
) -> StatisticalResult:
    """Test linear association with row and equality of row-bin means.

    A non-significant result is reported as a failure to reject the relevant null;
    it is not evidence that variables are independent or distributions uniform.
    """

    points = np.asarray(centroids_rc, dtype=float)
    magnitudes = np.abs(np.asarray(values))
    if points.shape != (len(magnitudes), 2):
        raise ValueError("centroids_rc and values must have compatible lengths.")
    if image_height <= 0 or bin_size <= 0:
        raise ValueError("image_height and bin_size must be positive.")

    rows = points[:, 0] if len(points) else np.empty(0, dtype=float)
    valid = np.isfinite(rows) & np.isfinite(magnitudes)
    rows = rows[valid]
    magnitudes = np.asarray(magnitudes[valid], dtype=float)

    bins = np.arange(0, image_height + bin_size, bin_size, dtype=float)
    if bins[-1] < image_height:
        bins = np.append(bins, float(image_height))
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bin_means = np.full(len(bin_centers), np.nan, dtype=float)
    bin_counts = np.zeros(len(bin_centers), dtype=int)
    groups: List[NDArray[np.floating]] = []

    for index, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
        in_bin = (rows >= lower) & (rows < upper)
        group = magnitudes[in_bin]
        bin_counts[index] = len(group)
        if len(group):
            bin_means[index] = float(np.mean(group))
        if len(group) >= 2:
            groups.append(group)

    correlation: Optional[float] = None
    correlation_p: Optional[float] = None
    if (
        len(rows) >= 3
        and not np.allclose(rows, rows[0])
        and not np.allclose(magnitudes, magnitudes[0])
    ):
        result = stats.pearsonr(rows, magnitudes)
        if np.isfinite(result.statistic) and np.isfinite(result.pvalue):
            correlation = float(result.statistic)
            correlation_p = float(result.pvalue)

    f_statistic: Optional[float] = None
    anova_p: Optional[float] = None
    pooled = np.concatenate(groups) if groups else np.empty(0, dtype=float)
    if len(groups) >= 2 and not np.allclose(pooled, pooled[0]):
        result = stats.f_oneway(*groups)
        if not np.isnan(result.statistic) and np.isfinite(result.pvalue):
            f_statistic = float(result.statistic)
            anova_p = float(result.pvalue)

    return StatisticalResult(
        timepoint=int(timepoint),
        correlation=correlation,
        correlation_p_value=correlation_p,
        anova_f_statistic=f_statistic,
        anova_p_value=anova_p,
        bin_centers=bin_centers,
        bin_means=bin_means,
        bin_counts=bin_counts,
    )


def benjamini_hochberg(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Return Benjamini-Hochberg adjusted q-values, preserving missing entries."""

    valid = [
        (index, float(value))
        for index, value in enumerate(p_values)
        if value is not None and math.isfinite(float(value)) and 0 <= float(value) <= 1
    ]
    adjusted: List[Optional[float]] = [None] * len(p_values)
    if not valid:
        return adjusted

    ordered = sorted(valid, key=lambda item: item[1])
    count = len(ordered)
    running = 1.0
    for rank_from_end in range(count, 0, -1):
        original_index, p_value = ordered[rank_from_end - 1]
        candidate = p_value * count / rank_from_end
        running = min(running, candidate)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def adjust_timepoint_statistics(results: Sequence[StatisticalResult]) -> None:
    """Attach BH q-values to a collection of per-timepoint results in place."""

    correlation_q = benjamini_hochberg([result.correlation_p_value for result in results])
    anova_q = benjamini_hochberg([result.anova_p_value for result in results])
    for result, corr_value, anova_value in zip(results, correlation_q, anova_q):
        result.correlation_q_value = corr_value
        result.anova_q_value = anova_value


def pearson_conclusion(result: StatisticalResult, alpha: float = 0.05) -> str:
    """Describe only what the Pearson test supports."""

    value = result.correlation_q_value
    label = "BH-adjusted q-value"
    if value is None:
        value = result.correlation_p_value
        label = "p-value"
    if value is None:
        return "Pearson test not performed (insufficient or constant data)."
    if value <= alpha:
        return f"Evidence of a linear association ({label} ≤ {alpha:g})."
    return f"Failed to reject no linear association ({label} > {alpha:g})."


def anova_conclusion(result: StatisticalResult, alpha: float = 0.05) -> str:
    """Describe only what the one-way ANOVA supports."""

    value = result.anova_q_value
    label = "BH-adjusted q-value"
    if value is None:
        value = result.anova_p_value
        label = "p-value"
    if value is None:
        return "ANOVA not performed (insufficient or constant groups)."
    if value <= alpha:
        return f"Evidence that at least one row-bin mean differs ({label} ≤ {alpha:g})."
    return f"Failed to reject equality of row-bin means ({label} > {alpha:g})."


def roi_psi_statistics(
    timepoint: int,
    psi_magnitude_map: NDArray[np.floating],
    polygon_xy: Iterable[Tuple[float, float]],
) -> ROIStatistics:
    """Average finite scalar ψ magnitudes in an ROI, excluding NaN background."""

    psi_map = np.asarray(psi_magnitude_map, dtype=float)
    if psi_map.ndim != 2:
        raise ValueError(f"psi_magnitude_map must be 2-D; got {psi_map.shape}.")
    polygon = tuple((float(x), float(y)) for x, y in polygon_xy)
    mask = polygon_mask(psi_map.shape, polygon)
    valid = mask & np.isfinite(psi_map)
    valid_count = int(np.count_nonzero(valid))
    average = float(np.mean(psi_map[valid])) if valid_count else None
    return ROIStatistics(
        timepoint=int(timepoint),
        polygon_xy=polygon,
        average_psi_magnitude=average,
        geometric_area_pixels=int(np.count_nonzero(mask)),
        valid_psi_pixels=valid_count,
    )


def write_statistics_csv(results: Sequence[StatisticalResult], path: Path) -> Path:
    """Write raw and BH-adjusted spatial-test values."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timepoint",
                "correlation",
                "correlation_p_value",
                "correlation_q_value_bh",
                "anova_f_statistic",
                "anova_p_value",
                "anova_q_value_bh",
                "pearson_conclusion",
                "anova_conclusion",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.timepoint,
                    _csv_value(result.correlation),
                    _csv_value(result.correlation_p_value),
                    _csv_value(result.correlation_q_value),
                    _csv_value(result.anova_f_statistic),
                    _csv_value(result.anova_p_value),
                    _csv_value(result.anova_q_value),
                    pearson_conclusion(result),
                    anova_conclusion(result),
                ]
            )
    return path


def _csv_value(value: Optional[float]) -> object:
    return "" if value is None else value
