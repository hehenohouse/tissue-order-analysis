"""Spatial statistics with conservative, explicit interpretations."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .fourier import polygon_mask
from .models import (
    BootstrapResult,
    ComplexPsiSummary,
    HarmonicResult,
    ROIStatistics,
    StatisticalResult,
)


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


def summarize_complex_psi(
    psi: NDArray[np.complexfloating],
    n_fold: int,
    weights: Optional[NDArray[np.floating]] = None,
) -> ComplexPsiSummary:
    """Summarize complex ψ without assigning an angle to zero resultant order."""

    values = np.asarray(psi, dtype=np.complex128)
    if values.ndim != 1:
        raise ValueError("psi must be one-dimensional.")
    if n_fold <= 0:
        raise ValueError("n_fold must be positive.")
    sample_weights = (
        np.ones(len(values), dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    if sample_weights.shape != values.shape:
        raise ValueError("weights and psi must have matching one-dimensional shapes.")
    valid = (
        np.isfinite(values.real)
        & np.isfinite(values.imag)
        & np.isfinite(sample_weights)
        & (sample_weights > 0)
    )
    if not np.any(valid):
        return ComplexPsiSummary(None, None, None, None, None, None)
    selected = values[valid]
    selected_weights = sample_weights[valid]
    weight_sum = float(np.sum(selected_weights))
    weighted_sum = complex(np.sum(selected_weights * selected))
    mean_value = weighted_sum / weight_sum
    resultant = float(abs(mean_value))
    magnitude_weight = float(np.sum(selected_weights * np.abs(selected)))
    coherence = float(abs(weighted_sum) / magnitude_weight) if magnitude_weight > 0 else None
    orientation_radians: Optional[float] = None
    orientation_degrees: Optional[float] = None
    if resultant > np.finfo(float).eps:
        period = 2.0 * np.pi / n_fold
        orientation_radians = float((np.angle(mean_value) / n_fold) % period)
        orientation_degrees = float(np.degrees(orientation_radians))
    return ComplexPsiSummary(
        mean_real=float(mean_value.real),
        mean_imag=float(mean_value.imag),
        resultant_magnitude=resultant,
        orientation_radians=orientation_radians,
        orientation_degrees=orientation_degrees,
        phase_coherence=coherence,
    )


def harmonic_association(
    timepoint: int,
    centroids_rc: NDArray[np.floating],
    values: NDArray[np.generic],
    image_height: int,
    *,
    basis: str,
    periodic_y: bool = True,
) -> HarmonicResult:
    """Fit a first harmonic to values along a vertically periodic coordinate."""

    points = np.asarray(centroids_rc, dtype=float)
    magnitudes = np.abs(np.asarray(values, dtype=float))
    if points.shape != (len(magnitudes), 2):
        raise ValueError("centroids_rc and values must have compatible lengths.")
    if image_height <= 0:
        raise ValueError("image_height must be positive.")
    empty = HarmonicResult(int(timepoint), basis, None, None, None, None, None, None, None)
    if not periodic_y:
        return empty
    rows = points[:, 0] if len(points) else np.empty(0, dtype=float)
    valid = np.isfinite(rows) & np.isfinite(magnitudes)
    rows = rows[valid]
    magnitudes = magnitudes[valid]
    if len(rows) < 4 or np.allclose(magnitudes, magnitudes[0]):
        return empty

    phase = 2.0 * np.pi * rows / image_height
    design = np.column_stack((np.ones(len(rows)), np.cos(phase), np.sin(phase)))
    coefficients, _, _, _ = np.linalg.lstsq(design, magnitudes, rcond=None)
    fitted = design @ coefficients
    residual_sum = float(np.sum((magnitudes - fitted) ** 2))
    total_sum = float(np.sum((magnitudes - np.mean(magnitudes)) ** 2))
    if total_sum <= 0:
        return empty
    r_squared = float(np.clip(1.0 - residual_sum / total_sum, 0.0, 1.0))
    numerator = max(0.0, total_sum - residual_sum) / 2.0
    if residual_sum <= np.finfo(float).eps:
        f_statistic = float("inf") if numerator > 0 else 0.0
        p_value = 0.0 if numerator > 0 else 1.0
    else:
        f_statistic = float(numerator / (residual_sum / (len(rows) - 3)))
        p_value = float(stats.f.sf(f_statistic, 2, len(rows) - 3))
    cosine = float(coefficients[1])
    sine = float(coefficients[2])
    amplitude = float(np.hypot(cosine, sine))
    peak_phase = float(np.arctan2(sine, cosine) % (2.0 * np.pi))
    return HarmonicResult(
        timepoint=int(timepoint),
        basis=basis,
        cosine_coefficient=cosine,
        sine_coefficient=sine,
        amplitude=amplitude,
        peak_row=float(peak_phase * image_height / (2.0 * np.pi)),
        r_squared=r_squared,
        f_statistic=f_statistic,
        p_value=p_value,
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


def adjust_harmonic_statistics(results: Sequence[HarmonicResult]) -> None:
    """Apply BH correction separately to each harmonic value basis."""

    for basis in sorted({result.basis for result in results}):
        selected = [result for result in results if result.basis == basis]
        adjusted = benjamini_hochberg([result.p_value for result in selected])
        for result, q_value in zip(selected, adjusted):
            result.q_value = q_value


def spatial_block_bootstrap(
    timepoint: int,
    centroids_rc: NDArray[np.floating],
    psi: NDArray[np.complexfloating],
    component_areas: NDArray[np.floating],
    *,
    n_fold: int,
    block_size_pixels: int,
    replicates: int,
    confidence_level: float,
    seed: int,
    image_height: int,
    periodic_y: bool,
) -> Tuple[BootstrapResult, ...]:
    """Return deterministic exploratory confidence intervals using spatial blocks."""

    points = np.asarray(centroids_rc, dtype=float)
    complex_values = np.asarray(psi, dtype=np.complex128)
    areas = np.asarray(component_areas, dtype=float)
    if points.shape != (len(complex_values), 2) or areas.shape != (len(complex_values),):
        raise ValueError("centroids_rc, psi, and component_areas must describe the same regions.")
    if not len(points):
        return ()
    if block_size_pixels <= 0 or replicates <= 0:
        raise ValueError("block_size_pixels and replicates must be positive.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")

    block_ids = np.floor(points / block_size_pixels).astype(int)
    _, inverse = np.unique(block_ids, axis=0, return_inverse=True)
    occupied = int(np.max(inverse)) + 1
    members = [np.flatnonzero(inverse == index) for index in range(occupied)]
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(timepoint)]))
    magnitudes = np.abs(complex_values)
    samples: Dict[str, List[float]] = {
        "cell_mean_psi_magnitude": [],
        "area_weighted_mean_psi_magnitude": [],
        "phase_coherence_cell": [],
    }
    if periodic_y:
        samples["harmonic_r_squared_raw"] = []

    for _ in range(replicates):
        selected_blocks = rng.integers(0, occupied, size=occupied)
        indices = np.concatenate([members[index] for index in selected_blocks])
        samples["cell_mean_psi_magnitude"].append(float(np.mean(magnitudes[indices])))
        samples["area_weighted_mean_psi_magnitude"].append(
            float(np.average(magnitudes[indices], weights=areas[indices]))
        )
        summary = summarize_complex_psi(complex_values[indices], n_fold)
        samples["phase_coherence_cell"].append(
            np.nan if summary.phase_coherence is None else summary.phase_coherence
        )
        if periodic_y:
            harmonic = harmonic_association(
                timepoint,
                points[indices],
                magnitudes[indices],
                image_height,
                basis="raw_psi_magnitude",
                periodic_y=True,
            )
            samples["harmonic_r_squared_raw"].append(
                np.nan if harmonic.r_squared is None else harmonic.r_squared
            )

    original_summary = summarize_complex_psi(complex_values, n_fold)
    estimates: Dict[str, Optional[float]] = {
        "cell_mean_psi_magnitude": float(np.mean(magnitudes)),
        "area_weighted_mean_psi_magnitude": float(np.average(magnitudes, weights=areas)),
        "phase_coherence_cell": original_summary.phase_coherence,
    }
    if periodic_y:
        estimates["harmonic_r_squared_raw"] = harmonic_association(
            timepoint,
            points,
            magnitudes,
            image_height,
            basis="raw_psi_magnitude",
            periodic_y=True,
        ).r_squared

    alpha = (1.0 - confidence_level) / 2.0
    output = []
    for metric, metric_samples in samples.items():
        finite_samples = np.asarray(metric_samples, dtype=float)
        finite_samples = finite_samples[np.isfinite(finite_samples)]
        if finite_samples.size:
            low, high = np.quantile(finite_samples, [alpha, 1.0 - alpha])
            confidence_low: Optional[float] = float(low)
            confidence_high: Optional[float] = float(high)
        else:
            confidence_low = None
            confidence_high = None
        output.append(
            BootstrapResult(
                timepoint=int(timepoint),
                metric=metric,
                estimate=estimates[metric],
                confidence_low=confidence_low,
                confidence_high=confidence_high,
                occupied_blocks=occupied,
                replicates=replicates,
            )
        )
    return tuple(output)


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
