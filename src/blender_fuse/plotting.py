"""Static, accessible plotting helpers for Blender Fuse outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .fourier import reconstruct_selected_frequencies
from .models import FourierResult, ROIStatistics, StatisticalResult

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
CRITICAL = "#d03b3b"
_BLUE_RAMP = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)


def sequential_blue_colormap() -> LinearSegmentedColormap:
    """Return the validated light-to-dark magnitude ramp."""

    colormap = LinearSegmentedColormap.from_list("blender_fuse_blue", _BLUE_RAMP)
    colormap.set_bad(BASELINE)
    return colormap


def _prepare_axis(axis: object) -> None:
    axis.set_facecolor(SURFACE)
    axis.tick_params(colors=TEXT_SECONDARY)
    for spine in axis.spines.values():
        spine.set_color(BASELINE)
        spine.set_linewidth(0.8)
    axis.grid(True, color=GRID, linewidth=0.7, linestyle="-")
    axis.set_axisbelow(True)


def _new_figure(figsize: Tuple[float, float] = (8.0, 7.0)) -> Tuple[object, object]:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=figsize, facecolor=SURFACE)
    _prepare_axis(axis)
    return figure, axis


def _save(figure: object, path: Path, dpi: int = 180) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    return path


def save_scalar_map(
    scalar_map: np.ndarray,
    path: Path,
    *,
    title: str,
    colorbar_label: str,
    polygon_xy: Optional[Iterable[Tuple[float, float]]] = None,
) -> Path:
    """Save a scalar magnitude map; NaN background is rendered neutrally."""

    figure, axis = _new_figure()
    image = axis.imshow(
        np.asarray(scalar_map, dtype=float),
        cmap=sequential_blue_colormap(),
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        interpolation="nearest",
    )
    if polygon_xy is not None:
        polygon = np.asarray(tuple(polygon_xy), dtype=float)
        if len(polygon):
            closed = np.vstack((polygon, polygon[0]))
            axis.plot(
                closed[:, 0],
                closed[:, 1],
                color=ORANGE,
                linewidth=2.0,
                solid_capstyle="round",
                label="ROI boundary",
            )
            axis.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    axis.set_title(title, color=TEXT_PRIMARY)
    axis.set_xlabel("X (pixels)", color=TEXT_SECONDARY)
    axis.set_ylabel("Y (pixels)", color=TEXT_SECONDARY)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label, color=TEXT_SECONDARY)
    return _save(figure, path)


def save_histogram(
    result: StatisticalResult,
    path: Path,
    *,
    title: str,
    ylabel: str,
) -> Path:
    """Save mean ψ magnitude by spatial row bin."""

    figure, axis = _new_figure((8.5, 5.5))
    finite = np.isfinite(result.bin_means)
    if len(result.bin_centers) > 1:
        width = float(np.min(np.diff(result.bin_centers))) * 0.75
    else:
        width = 1.0
    axis.bar(
        result.bin_centers[finite],
        result.bin_means[finite],
        width=width,
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=2.0,
    )
    axis.set_title(title, color=TEXT_PRIMARY)
    axis.set_xlabel("Y row (pixels)", color=TEXT_SECONDARY)
    axis.set_ylabel(ylabel, color=TEXT_SECONDARY)
    axis.set_ylim(0.0, 1.0)
    return _save(figure, path)


def save_fourier_map(
    result: FourierResult,
    path: Path,
    *,
    title: str,
    intensity_override: Optional[np.ndarray] = None,
    vmax: Optional[float] = None,
    show_selected_bins: bool = True,
) -> Path:
    """Save a sequential Fourier intensity map with optional selected-bin markers."""

    display_intensity = (
        result.intensity
        if intensity_override is None
        else np.asarray(intensity_override, dtype=float)
    )
    figure, axis = _new_figure()
    image = axis.imshow(
        display_intensity,
        cmap=sequential_blue_colormap(),
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax,
    )
    if show_selected_bins and result.bin_indices_rc.size:
        axis.scatter(
            result.bin_indices_rc[:, 1],
            result.bin_indices_rc[:, 0],
            s=64,
            facecolors="none",
            edgecolors=ORANGE,
            linewidths=2.0,
            label="Selected frequency bins",
        )
        axis.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    axis.set_title(title, color=TEXT_PRIMARY)
    axis.set_xlabel("Frequency column", color=TEXT_SECONDARY)
    axis.set_ylabel("Frequency row", color=TEXT_SECONDARY)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Intensity |FT|²", color=TEXT_SECONDARY)
    return _save(figure, path)


def save_reciprocal_overlay(
    scalar_map: np.ndarray,
    result: FourierResult,
    path: Path,
    *,
    title: str,
) -> Path:
    """Overlay a selected-frequency reconstruction on the analyzed ψ region."""

    x1, x2, y1, y2 = result.rect
    region = np.asarray(scalar_map, dtype=float)[y1:y2, x1:x2]
    lattice = reconstruct_selected_frequencies(result)
    figure, axis = _new_figure((6.5, 6.0))
    axis.imshow(
        region,
        cmap=sequential_blue_colormap(),
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        interpolation="nearest",
    )
    axis.imshow(
        lattice,
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        alpha=0.45,
        origin="upper",
        interpolation="nearest",
    )
    axis.set_title(title, color=TEXT_PRIMARY)
    axis.set_xlabel("ROI X (pixels)", color=TEXT_SECONDARY)
    axis.set_ylabel("ROI Y (pixels)", color=TEXT_SECONDARY)
    return _save(figure, path)


def save_statistical_summary(
    results: Sequence[StatisticalResult], path: Path, *, alpha: float = 0.05
) -> Path:
    """Save Pearson and ANOVA adjusted values as two aligned small multiples."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 8.0),
        sharex=True,
        facecolor=SURFACE,
    )
    series = (
        (
            axes[0],
            [result.correlation_q_value for result in results],
            BLUE,
            "Pearson: no linear association",
        ),
        (
            axes[1],
            [result.anova_q_value for result in results],
            ORANGE,
            "ANOVA: equal row-bin means",
        ),
    )
    timepoints = np.asarray([result.timepoint for result in results], dtype=int)
    positive_values = [
        float(value)
        for _, values, _, _ in series
        for value in values
        if value is not None and np.isfinite(value) and value > 0
    ]
    floor = min(positive_values, default=1e-12)
    floor = max(floor * 0.5, np.finfo(float).tiny)

    for axis, values, color, title in series:
        _prepare_axis(axis)
        valid = np.asarray(
            [value is not None and np.isfinite(value) for value in values], dtype=bool
        )
        plotted = np.asarray(
            [max(float(value), floor) if value is not None else np.nan for value in values]
        )
        axis.plot(
            timepoints[valid],
            plotted[valid],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=8.0,
            markeredgecolor=SURFACE,
            markeredgewidth=2.0,
        )
        axis.axhline(alpha, color=CRITICAL, linewidth=1.5, linestyle="--")
        axis.text(
            0.99,
            alpha,
            f"  α = {alpha:g}",
            color=TEXT_SECONDARY,
            va="bottom",
            ha="right",
            transform=axis.get_yaxis_transform(),
        )
        axis.set_yscale("log")
        axis.set_ylim(floor, 1.0)
        axis.set_ylabel("BH-adjusted q-value", color=TEXT_SECONDARY)
        axis.set_title(title, color=TEXT_PRIMARY, loc="left")
    axes[1].set_xlabel("Timepoint", color=TEXT_SECONDARY)
    figure.suptitle("Spatial statistical tests", color=TEXT_PRIMARY, y=0.97)
    figure.subplots_adjust(left=0.12, right=0.96, bottom=0.10, top=0.88, hspace=0.36)
    return _save(figure, path)


def save_roi_series(results: Sequence[ROIStatistics], path: Path) -> Path:
    """Plot ROI ψ magnitude only; area and valid counts remain in the CSV table."""

    figure, axis = _new_figure((10.0, 5.5))
    valid_results = [result for result in results if result.average_psi_magnitude is not None]
    if valid_results:
        axis.plot(
            [result.timepoint for result in valid_results],
            [result.average_psi_magnitude for result in valid_results],
            color=BLUE,
            linewidth=2.0,
            marker="o",
            markersize=8.0,
            markeredgecolor=SURFACE,
            markeredgewidth=2.0,
        )
    axis.set_title("ROI mean |ψₙ| over time", color=TEXT_PRIMARY)
    axis.set_xlabel("Timepoint", color=TEXT_SECONDARY)
    axis.set_ylabel("Mean |ψₙ|", color=TEXT_SECONDARY)
    axis.set_ylim(0.0, 1.0)
    return _save(figure, path)


def create_video(image_paths: Sequence[Path], output_path: Path, fps: float) -> Path:
    """Create an MP4 from existing image frames using the optional video extra."""

    if not image_paths:
        raise ValueError("Cannot create a video without image frames.")
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Video support is not installed; use `pip install blender-fuse[video]`."
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps, format="FFMPEG") as writer:
        for image_path in image_paths:
            writer.append_data(imageio.imread(image_path))
    return output_path
