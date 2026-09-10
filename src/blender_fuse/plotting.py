"""Static, accessible plotting helpers for Blender Fuse outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .fourier import reconstruct_selected_frequencies
from .models import (
    FourierResult,
    FourierV2Result,
    FrameSummary,
    ROIStatistics,
    StatisticalResult,
)

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
    return colormap.with_extremes(bad=BASELINE)


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


def save_fourier_v2_map(result: FourierV2Result, path: Path, *, title: str) -> Path:
    """Save Fourier v2 power on physical frequency axes."""

    from matplotlib.patches import Circle

    figure, axis = _new_figure()
    frequency_x = result.frequency_x_cycles_per_pixel
    frequency_y = result.frequency_y_cycles_per_pixel
    x_step = 1.0 / result.intensity.shape[1]
    y_step = 1.0 / result.intensity.shape[0]
    image = axis.imshow(
        np.log1p(result.intensity),
        cmap=sequential_blue_colormap(),
        origin="lower",
        interpolation="nearest",
        extent=(
            frequency_x[0] - x_step / 2.0,
            frequency_x[-1] + x_step / 2.0,
            frequency_y[0] - y_step / 2.0,
            frequency_y[-1] + y_step / 2.0,
        ),
        aspect="equal",
    )
    eligible_frequency = result.radial_frequency_cycles_per_pixel[result.eligible_mask]
    if eligible_frequency.size:
        radii = (float(np.min(eligible_frequency)), float(np.max(eligible_frequency)))
        for index, radius in enumerate(radii):
            axis.add_patch(
                Circle(
                    (0.0, 0.0),
                    radius,
                    fill=False,
                    edgecolor=TEXT_SECONDARY,
                    linewidth=1.5,
                    linestyle="--",
                    label="Eligible band boundaries" if index == 0 else None,
                )
            )
    peak_x = []
    peak_y = []
    for pair in result.peak_pairs:
        for index in (pair.first_index_rc, pair.second_index_rc):
            if index is not None:
                peak_y.append(float(frequency_y[index[0]]))
                peak_x.append(float(frequency_x[index[1]]))
    if peak_x:
        axis.scatter(
            peak_x,
            peak_y,
            s=64,
            facecolors="none",
            edgecolors=ORANGE,
            linewidths=2.0,
            label="Selected conjugate-pair members",
        )
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, frameon=False, labelcolor=TEXT_PRIMARY)
    axis.set_title(title, color=TEXT_PRIMARY)
    axis.set_xlabel("Frequency x (cycles/pixel)", color=TEXT_SECONDARY)
    axis.set_ylabel("Frequency y (cycles/pixel)", color=TEXT_SECONDARY)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("log(1 + normalized intensity)", color=TEXT_SECONDARY)
    return _save(figure, path)


def save_fourier_v2_radial(
    result: FourierV2Result, path: Path, *, title: str
) -> Path:
    """Save the annular mean-power profile and configured eligible band."""

    figure, axis = _new_figure((8.5, 5.5))
    profile = result.radial_profile
    finite = np.isfinite(profile.mean_intensity)
    axis.plot(
        profile.centers_cycles_per_pixel[finite],
        profile.mean_intensity[finite],
        color=BLUE,
        linewidth=2.0,
    )
    eligible_frequencies = result.radial_frequency_cycles_per_pixel[result.eligible_mask]
    if eligible_frequencies.size:
        lower = float(np.min(eligible_frequencies))
        upper = float(np.max(eligible_frequencies))
        axis.axvspan(lower, upper, color=BLUE, alpha=0.10, label="Eligible band")
        axis.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    if result.dominant_wavelength_pixels is not None and result.peak_pairs:
        frequency = result.peak_pairs[0].radial_frequency_cycles_per_pixel
        axis.axvline(frequency, color=ORANGE, linewidth=1.5, linestyle="--")
        axis.annotate(
            f"{result.dominant_wavelength_pixels:.2f} px",
            (frequency, result.peak_pairs[0].mean_intensity),
            xytext=(6, 6),
            textcoords="offset points",
            color=TEXT_PRIMARY,
        )
    axis.set_title(title, color=TEXT_PRIMARY)
    axis.set_xlabel("Radial frequency (cycles/pixel)", color=TEXT_SECONDARY)
    axis.set_ylabel("Mean normalized intensity", color=TEXT_SECONDARY)
    return _save(figure, path)


def save_fourier_v2_series(
    frames: Sequence[FrameSummary],
    path: Path,
    *,
    start_t: int,
    frame_interval_minutes: float,
    min_frequency_cycles_per_pixel: float,
    max_frequency_cycles_per_pixel: float,
) -> Path:
    """Plot v2 concentration and dominant wavelength as aligned small multiples."""

    import matplotlib.pyplot as plt

    selected = [frame for frame in frames if frame.fourier_v2_metric is not None]
    if not selected:
        raise ValueError("Cannot plot Fourier v2 series without v2 frame summaries.")
    minutes = np.asarray(
        [(frame.timepoint - start_t) * frame_interval_minutes for frame in selected],
        dtype=float,
    )
    metrics = np.asarray([frame.fourier_v2_metric for frame in selected], dtype=float)
    wavelengths = np.asarray(
        [
            np.nan
            if frame.fourier_v2_wavelength_pixels is None
            else frame.fourier_v2_wavelength_pixels
            for frame in selected
        ],
        dtype=float,
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.5),
        sharex=True,
        facecolor=SURFACE,
    )
    for axis in axes:
        _prepare_axis(axis)
    axes[0].plot(minutes, metrics, color=BLUE, linewidth=2.0)
    axes[0].set_ylabel("Eligible max / mean", color=TEXT_SECONDARY)
    axes[0].set_title("Band-limited spectral concentration", color=TEXT_PRIMARY, loc="left")

    finite = np.isfinite(wavelengths)
    if np.any(finite):
        axes[1].plot(minutes[finite], wavelengths[finite], color=ORANGE, linewidth=2.0)
    lower_wavelength = 1.0 / max_frequency_cycles_per_pixel
    upper_wavelength = (
        1.0 / min_frequency_cycles_per_pixel
        if min_frequency_cycles_per_pixel > 0
        else None
    )
    if upper_wavelength is not None:
        axes[1].axhspan(
            lower_wavelength,
            upper_wavelength,
            color=BLUE,
            alpha=0.10,
            label="Configured eligible wavelengths",
        )
        axes[1].legend(frameon=False, labelcolor=TEXT_PRIMARY)
    axes[1].set_ylabel("Dominant wavelength (pixels)", color=TEXT_SECONDARY)
    axes[1].set_xlabel("Time (minutes)", color=TEXT_SECONDARY)
    axes[1].set_title("Dominant conjugate pair", color=TEXT_PRIMARY, loc="left")
    figure.suptitle("Fourier v2 over time", color=TEXT_PRIMARY, y=0.97)
    figure.subplots_adjust(left=0.12, right=0.96, bottom=0.10, top=0.88, hspace=0.34)
    return _save(figure, path)


def _rgb_surface() -> np.ndarray:
    return np.asarray([int(SURFACE[index : index + 2], 16) for index in (1, 3, 5)], dtype=np.uint8)


def _as_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """Normalize grayscale/RGB/RGBA input without resizing its pixels."""

    array = np.asarray(frame)
    if np.issubdtype(array.dtype, np.floating):
        maximum = float(np.nanmax(array, initial=0.0))
        scale = 255.0 if maximum <= 1.0 else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError(f"Video frame must be 2-D or 3-D; got shape {array.shape}.")
    if array.shape[2] == 1:
        return np.repeat(array, 3, axis=2)
    if array.shape[2] == 3:
        return array
    if array.shape[2] == 4:
        alpha = array[:, :, 3:4].astype(float) / 255.0
        surface = _rgb_surface().reshape(1, 1, 3).astype(float)
        composited = array[:, :, :3].astype(float) * alpha + surface * (1.0 - alpha)
        return np.rint(composited).astype(np.uint8)
    raise ValueError(f"Video frame must have 1, 3, or 4 channels; got shape {array.shape}.")


def _encoder_dimension(value: int) -> int:
    return ((int(value) + 15) // 16) * 16


def create_video(image_paths: Sequence[Path], output_path: Path, fps: float) -> Path:
    """Create an atomic, constant-canvas MP4 from possibly variable PNG sizes."""

    if not image_paths:
        raise ValueError("Cannot create a video without image frames.")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive.")
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Video support is not installed; use `pip install blender-fuse[video]`."
        ) from exc

    paths = [Path(path) for path in image_paths]
    maximum_height = 0
    maximum_width = 0
    for path in paths:
        try:
            frame = _as_uint8_rgb(imageio.imread(path))
        except Exception as exc:
            raise ValueError(f"Could not normalize video frame {path}: {exc}") from exc
        maximum_height = max(maximum_height, frame.shape[0])
        maximum_width = max(maximum_width, frame.shape[1])
    target_height = _encoder_dimension(maximum_height)
    target_width = _encoder_dimension(maximum_width)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    try:
        with imageio.get_writer(
            temporary,
            fps=float(fps),
            format="FFMPEG",
            macro_block_size=None,
        ) as writer:
            for path in paths:
                frame = _as_uint8_rgb(imageio.imread(path))
                canvas = np.empty((target_height, target_width, 3), dtype=np.uint8)
                canvas[...] = _rgb_surface()
                row = (target_height - frame.shape[0]) // 2
                column = (target_width - frame.shape[1]) // 2
                canvas[row : row + frame.shape[0], column : column + frame.shape[1]] = frame
                writer.append_data(canvas)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
