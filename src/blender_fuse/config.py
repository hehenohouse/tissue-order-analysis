"""Validated configuration for Blender Fuse analyses."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

PointXY = Tuple[float, float]
FourierRectangle = Tuple[int, int, int, int]


@dataclass
class OutputOptions:
    """Control which analysis artifacts are written to disk."""

    save_order_images: bool = True
    save_order_arrays: bool = True
    save_smoothed_images: bool = True
    save_histograms: bool = True
    save_smoothed_histograms: bool = True
    save_fourier_images: bool = True
    save_fourier_arrays: bool = True
    save_frequency_bins: bool = True
    save_metrics_csv: bool = True
    save_reciprocal_overlay: bool = True
    save_statistical_summary: bool = True
    save_videos: bool = False
    retain_frame_data: bool = False


@dataclass
class PIVConfig:
    """Optional PIVlab and evolving-ROI settings.

    Public polygon vertices always use ``(x, y)`` coordinates.
    """

    enabled: bool = False
    mat_path: Optional[Path] = None
    roi_polygon: Tuple[PointXY, ...] = (
        (800.0, 800.0),
        (800.0, 1000.0),
        (1000.0, 1000.0),
        (1000.0, 800.0),
    )
    interpolation: str = "bilinear"
    save_roi_images: bool = True
    save_roi_fourier: bool = True
    save_roi_statistics: bool = True
    roi_fourier_crop_size: int = 100
    roi_fourier_vmax: Optional[float] = 1e7

    def __post_init__(self) -> None:
        if self.mat_path is not None:
            self.mat_path = Path(self.mat_path)
        self.roi_polygon = tuple((float(point[0]), float(point[1])) for point in self.roi_polygon)

    def validate(self) -> None:
        methods = {"bilinear", "bicubic", "inverse_distance", "nearest"}
        if self.interpolation not in methods:
            choices = ", ".join(sorted(methods))
            raise ValueError(
                f"Unsupported PIV interpolation {self.interpolation!r}; choose {choices}."
            )
        if len(self.roi_polygon) < 3:
            raise ValueError("PIV ROI polygon must contain at least three (x, y) vertices.")
        if any(not math.isfinite(value) for point in self.roi_polygon for value in point):
            raise ValueError("PIV ROI polygon coordinates must be finite.")
        if self.roi_fourier_crop_size <= 0:
            raise ValueError("roi_fourier_crop_size must be positive.")
        if self.roi_fourier_vmax is not None and self.roi_fourier_vmax <= 0:
            raise ValueError("roi_fourier_vmax must be positive or None.")
        if self.enabled and self.mat_path is None:
            raise ValueError("piv.mat_path is required when PIV analysis is enabled.")


@dataclass
class AnalysisConfig:
    """Configuration for a segmentation time-series analysis."""

    data_dir: Path
    output_dir: Optional[Path] = None
    start_t: int = 1
    end_t: Optional[int] = None
    filename_pattern: str = "*T{timepoint:04d}_Simple Segmentation.h5"
    dataset_name: Optional[str] = None
    n_fold: int = 6
    smooth_neighbors: int = 6
    smooth_include_self: bool = False
    periodic_y: bool = True
    vertical_shift: int = 0
    background_values: Tuple[float, ...] = (0.0, 2.0)
    fourier_rect: Optional[FourierRectangle] = (700, 900, 800, 1000)
    top_frequency_bins: int = 10
    histogram_bin_size: int = 25
    frame_interval_minutes: float = 0.5
    fps: float = 2.0
    strict_missing: bool = False
    outputs: OutputOptions = field(default_factory=OutputOptions)
    piv: PIVConfig = field(default_factory=PIVConfig)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        self.background_values = tuple(float(value) for value in self.background_values)

    @property
    def resolved_output_dir(self) -> Path:
        """Return the explicit output directory or a safe default beside the data."""

        return self.output_dir or self.data_dir / "analysis_output"

    def validate(self) -> None:
        """Validate settings without touching the filesystem."""

        if not isinstance(self.start_t, int) or isinstance(self.start_t, bool):
            raise TypeError("start_t must be an integer.")
        if self.end_t is not None and (
            not isinstance(self.end_t, int) or isinstance(self.end_t, bool)
        ):
            raise TypeError("end_t must be an integer or None.")
        if self.start_t < 0:
            raise ValueError("start_t must be non-negative.")
        if self.end_t is not None and self.end_t < self.start_t:
            raise ValueError("end_t must be greater than or equal to start_t.")
        if "{timepoint" not in self.filename_pattern:
            raise ValueError("filename_pattern must contain a {timepoint...} field.")
        if self.n_fold <= 0:
            raise ValueError("n_fold must be positive.")
        if self.smooth_neighbors < 0:
            raise ValueError("smooth_neighbors cannot be negative.")
        if self.top_frequency_bins <= 0:
            raise ValueError("top_frequency_bins must be positive.")
        if self.histogram_bin_size <= 0:
            raise ValueError("histogram_bin_size must be positive.")
        if self.frame_interval_minutes <= 0:
            raise ValueError("frame_interval_minutes must be positive.")
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if not self.background_values:
            raise ValueError("background_values cannot be empty.")
        if self.fourier_rect is not None:
            if any(
                not isinstance(value, int) or isinstance(value, bool) for value in self.fourier_rect
            ):
                raise TypeError("fourier_rect coordinates must be integers.")
            x1, x2, y1, y2 = self.fourier_rect
            if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
                raise ValueError(
                    "fourier_rect must be (x1, x2, y1, y2) with non-negative "
                    "origins and increasing bounds."
                )
        self.piv.validate()


def polygon_from_sequence(points: Sequence[Sequence[float]]) -> Tuple[PointXY, ...]:
    """Convert a generic sequence into an immutable public ``(x, y)`` polygon."""

    polygon = tuple((float(point[0]), float(point[1])) for point in points)
    if len(polygon) < 3:
        raise ValueError("A polygon requires at least three vertices.")
    return polygon
