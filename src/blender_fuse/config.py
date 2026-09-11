"""Validated configuration for Blender Fuse analyses."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Optional, Sequence, Tuple

PointXY = Tuple[float, float]
FourierRectangle = Tuple[int, int, int, int]


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")


def _require_int(name: str, value: object, *, minimum: int = 0) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")


def _require_finite(name: str, value: object, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")
    if positive and float(value) <= 0:
        raise ValueError(f"{name} must be positive.")


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
    save_segmentation_qc: bool = True
    save_extended_metrics: bool = True
    save_harmonic_statistics: bool = True
    save_bootstrap_statistics: bool = True
    save_fourier_v2_arrays: bool = True
    save_fourier_v2_images: bool = True
    save_fourier_v2_tables: bool = True

    def validate(self) -> None:
        for name, value in vars(self).items():
            _require_bool(f"outputs.{name}", value)
        if self.save_videos and not (
            self.save_order_images
            or self.save_smoothed_images
            or self.save_fourier_images
            or self.save_fourier_v2_images
        ):
            raise ValueError("Video output requires at least one enabled image output.")


@dataclass
class SegmentationFilterConfig:
    """Optional connected-component filters; all are disabled by default."""

    min_area_pixels: Optional[int] = None
    max_area_pixels: Optional[int] = None
    exclude_border: bool = False

    @property
    def enabled(self) -> bool:
        return (
            self.min_area_pixels is not None
            or self.max_area_pixels is not None
            or self.exclude_border
        )

    def validate(self) -> None:
        if self.min_area_pixels is not None:
            _require_int("segmentation_filter.min_area_pixels", self.min_area_pixels, minimum=1)
        if self.max_area_pixels is not None:
            _require_int("segmentation_filter.max_area_pixels", self.max_area_pixels, minimum=1)
        if (
            self.min_area_pixels is not None
            and self.max_area_pixels is not None
            and self.max_area_pixels < self.min_area_pixels
        ):
            raise ValueError(
                "segmentation_filter.max_area_pixels must be at least min_area_pixels."
            )
        _require_bool("segmentation_filter.exclude_border", self.exclude_border)


@dataclass
class FourierV2Config:
    """Opt-in windowed, band-limited rectangular Fourier analysis."""

    enabled: bool = False
    window: str = "hann"
    min_frequency_cycles_per_pixel: Optional[float] = None
    max_frequency_cycles_per_pixel: Optional[float] = None
    top_pairs: int = 5
    radial_bin_width_cycles_per_pixel: Optional[float] = None

    def validate(self, *, fourier_rect: Optional[FourierRectangle]) -> None:
        _require_bool("fourier_v2.enabled", self.enabled)
        windows = {"none", "hann", "hamming", "blackman"}
        if self.window not in windows:
            choices = ", ".join(sorted(windows))
            raise ValueError(f"Unsupported Fourier v2 window {self.window!r}; choose {choices}.")
        _require_int("fourier_v2.top_pairs", self.top_pairs, minimum=1)
        for name, value in (
            ("min_frequency_cycles_per_pixel", self.min_frequency_cycles_per_pixel),
            ("max_frequency_cycles_per_pixel", self.max_frequency_cycles_per_pixel),
        ):
            if value is not None:
                _require_finite(f"fourier_v2.{name}", value)
                if float(value) < 0:
                    raise ValueError(f"fourier_v2.{name} cannot be negative.")
                if float(value) > math.sqrt(0.5):
                    raise ValueError(f"fourier_v2.{name} exceeds the 2-D Nyquist limit.")
        if self.radial_bin_width_cycles_per_pixel is not None:
            _require_finite(
                "fourier_v2.radial_bin_width_cycles_per_pixel",
                self.radial_bin_width_cycles_per_pixel,
                positive=True,
            )
        if self.enabled:
            if fourier_rect is None:
                raise ValueError("fourier_rect is required when Fourier v2 is enabled.")
            if self.min_frequency_cycles_per_pixel is None:
                raise ValueError(
                    "fourier_v2.min_frequency_cycles_per_pixel is required when enabled."
                )
            if self.max_frequency_cycles_per_pixel is None:
                raise ValueError(
                    "fourier_v2.max_frequency_cycles_per_pixel is required when enabled."
                )
            if self.max_frequency_cycles_per_pixel <= self.min_frequency_cycles_per_pixel:
                raise ValueError("Fourier v2 maximum frequency must exceed minimum frequency.")


@dataclass
class SpatialBootstrapConfig:
    """Optional spatial block-bootstrap settings."""

    enabled: bool = False
    replicates: int = 1000
    block_size_pixels: int = 100
    confidence_level: float = 0.95
    seed: int = 0

    def validate(self) -> None:
        _require_bool("spatial_bootstrap.enabled", self.enabled)
        _require_int("spatial_bootstrap.replicates", self.replicates, minimum=1)
        _require_int("spatial_bootstrap.block_size_pixels", self.block_size_pixels, minimum=1)
        _require_int("spatial_bootstrap.seed", self.seed, minimum=0)
        _require_finite("spatial_bootstrap.confidence_level", self.confidence_level)
        if not 0 < self.confidence_level < 1:
            raise ValueError(
                "spatial_bootstrap.confidence_level must lie strictly between 0 and 1."
            )


@dataclass
class ResumeConfig:
    """Validated checkpoint-resume settings."""

    enabled: bool = False

    def validate(self) -> None:
        _require_bool("resume.enabled", self.enabled)


@dataclass
class ProvenanceConfig:
    """Run-manifest and checksum settings."""

    enabled: bool = True
    hash_inputs: bool = True
    hash_outputs: bool = True

    def validate(self) -> None:
        _require_bool("provenance.enabled", self.enabled)
        _require_bool("provenance.hash_inputs", self.hash_inputs)
        _require_bool("provenance.hash_outputs", self.hash_outputs)


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
        for name in ("enabled", "save_roi_images", "save_roi_fourier", "save_roi_statistics"):
            _require_bool(f"piv.{name}", getattr(self, name))
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
        _require_int("piv.roi_fourier_crop_size", self.roi_fourier_crop_size, minimum=1)
        if self.roi_fourier_vmax is not None:
            _require_finite("piv.roi_fourier_vmax", self.roi_fourier_vmax, positive=True)
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
    segmentation_filter: SegmentationFilterConfig = field(
        default_factory=SegmentationFilterConfig
    )
    fourier_v2: FourierV2Config = field(default_factory=FourierV2Config)
    spatial_bootstrap: SpatialBootstrapConfig = field(
        default_factory=SpatialBootstrapConfig
    )
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in self.background_values
        ):
            raise TypeError("background_values must contain only numeric values.")
        self.background_values = tuple(float(value) for value in self.background_values)
        if self.fourier_rect is not None:
            self.fourier_rect = tuple(self.fourier_rect)  # type: ignore[assignment]

    @property
    def resolved_output_dir(self) -> Path:
        """Return the explicit output directory or a safe default beside the data."""

        return self.output_dir or self.data_dir / "analysis_output"

    def validate(self) -> None:
        """Validate settings without touching the filesystem."""

        _require_int("start_t", self.start_t)
        if self.end_t is not None:
            _require_int("end_t", self.end_t)
        if self.end_t is not None and self.end_t < self.start_t:
            raise ValueError("end_t must be greater than or equal to start_t.")
        if not isinstance(self.filename_pattern, str) or "{timepoint" not in self.filename_pattern:
            raise ValueError("filename_pattern must contain a {timepoint...} field.")
        if self.dataset_name is not None and (
            not isinstance(self.dataset_name, str) or not self.dataset_name.strip()
        ):
            raise ValueError("dataset_name must be a non-empty string or None.")
        _require_int("n_fold", self.n_fold, minimum=1)
        _require_int("smooth_neighbors", self.smooth_neighbors)
        _require_bool("smooth_include_self", self.smooth_include_self)
        _require_bool("periodic_y", self.periodic_y)
        if not isinstance(self.vertical_shift, Integral) or isinstance(self.vertical_shift, bool):
            raise TypeError("vertical_shift must be an integer.")
        _require_int("top_frequency_bins", self.top_frequency_bins, minimum=1)
        _require_int("histogram_bin_size", self.histogram_bin_size, minimum=1)
        _require_finite("frame_interval_minutes", self.frame_interval_minutes, positive=True)
        _require_finite("fps", self.fps, positive=True)
        _require_bool("strict_missing", self.strict_missing)
        if not self.background_values:
            raise ValueError("background_values cannot be empty.")
        if any(not math.isfinite(value) for value in self.background_values):
            raise ValueError("background_values must be finite.")
        if self.fourier_rect is not None:
            if len(self.fourier_rect) != 4 or any(
                not isinstance(value, Integral) or isinstance(value, bool)
                for value in self.fourier_rect
            ):
                raise TypeError("fourier_rect coordinates must be four integers.")
            x1, x2, y1, y2 = self.fourier_rect
            if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
                raise ValueError(
                    "fourier_rect must be (x1, x2, y1, y2) with non-negative "
                    "origins and increasing bounds."
                )
        self.outputs.validate()
        self.piv.validate()
        self.segmentation_filter.validate()
        self.fourier_v2.validate(fourier_rect=self.fourier_rect)
        self.spatial_bootstrap.validate()
        self.resume.validate()
        self.provenance.validate()
        if self.resume.enabled and self.piv.enabled:
            raise ValueError("Resume with PIV is not supported in Blender Fuse 0.2.")
        effective_video_sources = (
            self.outputs.save_order_images
            or (self.outputs.save_smoothed_images and self.smooth_neighbors > 0)
            or (self.outputs.save_fourier_images and self.fourier_rect is not None)
            or (self.outputs.save_fourier_v2_images and self.fourier_v2.enabled)
        )
        if self.outputs.save_videos and not effective_video_sources:
            raise ValueError(
                "Video output requires an image sequence enabled by the analysis "
                "configuration."
            )


def polygon_from_sequence(points: Sequence[Sequence[float]]) -> Tuple[PointXY, ...]:
    """Convert a generic sequence into an immutable public ``(x, y)`` polygon."""

    polygon = tuple((float(point[0]), float(point[1])) for point in points)
    if len(polygon) < 3:
        raise ValueError("A polygon requires at least three vertices.")
    return polygon
