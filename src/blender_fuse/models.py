"""Typed numerical results returned by Blender Fuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
ComplexArray = NDArray[np.complexfloating]
IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]


@dataclass
class FourierResult:
    """Fourier spectrum, intensity, metric, and selected frequency bins."""

    rect: Tuple[int, int, int, int]
    spectrum: ComplexArray
    intensity: FloatArray
    metric: float
    bin_indices_rc: IntArray
    offsets_rc: FloatArray
    bin_values: FloatArray
    distances: FloatArray

    @property
    def average_distance(self) -> float:
        if self.distances.size == 0:
            return float("nan")
        return float(np.mean(self.distances))


@dataclass(frozen=True)
class FourierPeakPair:
    """One deterministic conjugate frequency pair from Fourier v2."""

    rank: int
    first_index_rc: Tuple[int, int]
    second_index_rc: Optional[Tuple[int, int]]
    frequency_y_cycles_per_pixel: float
    frequency_x_cycles_per_pixel: float
    radial_frequency_cycles_per_pixel: float
    wavelength_pixels: float
    first_intensity: float
    second_intensity: Optional[float]
    mean_intensity: float
    total_power: float
    member_count: int


@dataclass
class FourierRadialProfile:
    """Annular Fourier v2 power profile in cycles per pixel."""

    edges_cycles_per_pixel: FloatArray
    centers_cycles_per_pixel: FloatArray
    mean_intensity: FloatArray
    total_power: FloatArray
    counts: IntArray


@dataclass
class FourierV2Result:
    """Windowed, band-limited Fourier analysis with physical frequency units."""

    rect: Tuple[int, int, int, int]
    schema_version: str
    window: str
    removed_mean: float
    window_power_normalization: float
    spectrum: ComplexArray
    intensity: FloatArray
    frequency_y_cycles_per_pixel: FloatArray
    frequency_x_cycles_per_pixel: FloatArray
    radial_frequency_cycles_per_pixel: FloatArray
    eligible_mask: BoolArray
    eligible_max_intensity: float
    eligible_mean_intensity: float
    metric: float
    peak_pairs: Tuple[FourierPeakPair, ...]
    radial_profile: FourierRadialProfile
    low_frequency_power_fraction: Optional[float]
    eligible_power_fraction: Optional[float]

    @property
    def dominant_wavelength_pixels(self) -> Optional[float]:
        return self.peak_pairs[0].wavelength_pixels if self.peak_pairs else None


@dataclass
class SegmentationQC:
    """Raw and retained connected-component quality measurements."""

    image_height: int
    image_width: int
    raw_foreground_pixels: int
    retained_foreground_pixels: int
    raw_component_count: int
    retained_component_count: int
    rejected_any_count: int
    rejected_min_area_count: int
    rejected_max_area_count: int
    rejected_border_count: int
    raw_area_min: Optional[float]
    raw_area_median: Optional[float]
    raw_area_mean: Optional[float]
    raw_area_max: Optional[float]
    retained_area_min: Optional[float]
    retained_area_median: Optional[float]
    retained_area_mean: Optional[float]
    retained_area_max: Optional[float]


@dataclass
class ComplexPsiSummary:
    """Weighted complex ψ mean, orientation, and phase coherence."""

    mean_real: Optional[float]
    mean_imag: Optional[float]
    resultant_magnitude: Optional[float]
    orientation_radians: Optional[float]
    orientation_degrees: Optional[float]
    phase_coherence: Optional[float]


@dataclass
class HarmonicResult:
    """Association with periodic vertical position."""

    timepoint: int
    basis: str
    cosine_coefficient: Optional[float]
    sine_coefficient: Optional[float]
    amplitude: Optional[float]
    peak_row: Optional[float]
    r_squared: Optional[float]
    f_statistic: Optional[float]
    p_value: Optional[float]
    q_value: Optional[float] = None


@dataclass
class BootstrapResult:
    """Exploratory spatial block-bootstrap confidence intervals."""

    timepoint: int
    metric: str
    estimate: Optional[float]
    confidence_low: Optional[float]
    confidence_high: Optional[float]
    occupied_blocks: int
    replicates: int


@dataclass
class StatisticalResult:
    """Per-frame spatial tests with deliberately narrow interpretations."""

    timepoint: int
    correlation: Optional[float]
    correlation_p_value: Optional[float]
    anova_f_statistic: Optional[float]
    anova_p_value: Optional[float]
    correlation_q_value: Optional[float] = None
    anova_q_value: Optional[float] = None
    bin_centers: FloatArray = field(default_factory=lambda: np.empty(0, dtype=float))
    bin_means: FloatArray = field(default_factory=lambda: np.empty(0, dtype=float))
    bin_counts: IntArray = field(default_factory=lambda: np.empty(0, dtype=int))


@dataclass
class ROIStatistics:
    """Scalar ψ summary for one ROI at one timepoint."""

    timepoint: int
    polygon_xy: Tuple[Tuple[float, float], ...]
    average_psi_magnitude: Optional[float]
    geometric_area_pixels: int
    valid_psi_pixels: int


@dataclass
class FrameAnalysis:
    """Complete numerical analysis of one segmentation frame."""

    timepoint: int
    source_path: Path
    segmentation: BoolArray
    labels: IntArray
    centroids_rc: FloatArray
    psi: ComplexArray
    psi_magnitude: FloatArray
    psi_map: FloatArray
    smoothed_magnitude: Optional[FloatArray] = None
    smoothed_map: Optional[FloatArray] = None
    fourier: Optional[FourierResult] = None
    statistics: Optional[StatisticalResult] = None
    analysis_segmentation: Optional[BoolArray] = None
    component_areas: FloatArray = field(default_factory=lambda: np.empty(0, dtype=float))
    segmentation_qc: Optional[SegmentationQC] = None
    fourier_v2: Optional[FourierV2Result] = None
    harmonic_raw: Optional[HarmonicResult] = None
    harmonic_smoothed: Optional[HarmonicResult] = None
    complex_psi_cell: Optional[ComplexPsiSummary] = None
    complex_psi_area: Optional[ComplexPsiSummary] = None
    bootstrap_results: Tuple[BootstrapResult, ...] = ()


@dataclass
class FrameSummary:
    """Lightweight information retained by a streaming pipeline."""

    timepoint: int
    source_path: Path
    region_count: int
    mean_psi_magnitude: Optional[float]
    mean_smoothed_psi_magnitude: Optional[float]
    fourier_metric: Optional[float]
    raw_region_count: Optional[int] = None
    area_weighted_mean_psi_magnitude: Optional[float] = None
    area_weighted_mean_smoothed_psi_magnitude: Optional[float] = None
    fourier_v2_metric: Optional[float] = None
    fourier_v2_wavelength_pixels: Optional[float] = None
    fourier_v2_radial_frequency_cycles_per_pixel: Optional[float] = None
    fourier_v2_low_frequency_power_fraction: Optional[float] = None
    fourier_v2_eligible_power_fraction: Optional[float] = None
    complex_psi_cell: Optional[ComplexPsiSummary] = None
    complex_psi_area: Optional[ComplexPsiSummary] = None


@dataclass
class AnalysisResult:
    """Structured result of a complete analysis run."""

    output_dir: Path
    frames: List[FrameSummary] = field(default_factory=list)
    missing_timepoints: List[int] = field(default_factory=list)
    statistics: List[StatisticalResult] = field(default_factory=list)
    roi_statistics: List[ROIStatistics] = field(default_factory=list)
    written_paths: List[Path] = field(default_factory=list)
    retained_frames: Dict[int, FrameAnalysis] = field(default_factory=dict)
    harmonic_statistics: List[HarmonicResult] = field(default_factory=list)
    segmentation_qc: Dict[int, SegmentationQC] = field(default_factory=dict)
    bootstrap_statistics: List[BootstrapResult] = field(default_factory=list)
    resumed_timepoints: List[int] = field(default_factory=list)

    @property
    def processed_timepoints(self) -> List[int]:
        return [frame.timepoint for frame in self.frames]
