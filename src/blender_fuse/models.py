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


@dataclass
class FrameSummary:
    """Lightweight information retained by a streaming pipeline."""

    timepoint: int
    source_path: Path
    region_count: int
    mean_psi_magnitude: Optional[float]
    mean_smoothed_psi_magnitude: Optional[float]
    fourier_metric: Optional[float]


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

    @property
    def processed_timepoints(self) -> List[int]:
        return [frame.timepoint for frame in self.frames]
