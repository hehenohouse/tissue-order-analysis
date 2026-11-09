"""Regression tests for periodic labeling, roll-invariant statistics, and Voronoi edges."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blender_fuse import AnalysisConfig, analyze_frame
from blender_fuse.config import SegmentationFilterConfig
from blender_fuse.order import (
    compute_periodic_order_parameter,
    filter_labeled_components,
    label_segmentation,
    smooth_order_magnitudes,
    voronoi_boundary_mask,
)
from blender_fuse.statistics import (
    pearson_conclusion,
    spatial_block_bootstrap,
    spatial_statistics,
)


def _wrapping_blob() -> np.ndarray:
    segmentation = np.zeros((100, 60), dtype=bool)
    segmentation[:5, 20:30] = True
    segmentation[-5:, 20:30] = True
    segmentation[40:45, 40:45] = True
    return segmentation


def test_periodic_labeling_merges_component_across_vertical_edge() -> None:
    labels, centroids = label_segmentation(_wrapping_blob(), periodic_y=True)
    assert labels.max() == 2
    assert len(centroids) == 2
    # Rows 95..99 and 0..4 unwrap to -5..4, whose mean is -0.5 -> 99.5.
    wrapped = centroids[np.argmin(np.abs(centroids[:, 1] - 24.5))]
    assert wrapped == pytest.approx([99.5, 24.5])
    assert labels[0, 20] == labels[-1, 20]


def test_periodic_labeling_joins_diagonal_contact_across_edge() -> None:
    segmentation = np.zeros((10, 10), dtype=bool)
    segmentation[0, 4] = True
    segmentation[-1, 5] = True
    labels, _ = label_segmentation(segmentation, periodic_y=True)
    assert labels.max() == 1


def test_non_periodic_labeling_is_unchanged() -> None:
    labels, centroids = label_segmentation(_wrapping_blob(), periodic_y=False)
    assert labels.max() == 3
    assert centroids[:, 0].min() == pytest.approx(2.0)


def test_filter_reports_merged_component_once() -> None:
    result = filter_labeled_components(
        _wrapping_blob(), SegmentationFilterConfig(), periodic_y=True
    )
    assert result.quality.raw_component_count == 2
    assert sorted(result.component_areas.tolist()) == [25.0, 100.0]


def test_pearson_is_skipped_for_periodic_rows_and_anova_is_roll_invariant() -> None:
    rng = np.random.default_rng(0)
    height = 400
    points = rng.uniform(0, height, (300, 2))
    values = 0.5 + 0.3 * np.cos(2 * np.pi * points[:, 0] / height)
    periodic = spatial_statistics(1, points, values, height, 50, periodic_y=True)
    assert periodic.correlation is None
    assert periodic.correlation_skipped_reason == "periodic row coordinate"
    assert "periodic" in pearson_conclusion(periodic)
    assert periodic.anova_p_value is not None

    flat = spatial_statistics(1, points, values, height, 50)
    assert flat.correlation is not None


def test_analyze_frame_statistics_do_not_depend_on_vertical_shift(
    tmp_path: Path, write_segmentation
) -> None:
    rng = np.random.default_rng(3)
    cells = {(int(r), int(c)) for r, c in rng.integers(1, 63, size=(120, 2))}
    write_segmentation(tmp_path, 1, cells, shape=(64, 64))
    path = next(tmp_path.glob("*.h5"))
    results = []
    for shift in (0, 17, 40):
        config = AnalysisConfig(data_dir=tmp_path, fourier_rect=None, vertical_shift=shift)
        results.append(analyze_frame(path, 1, config).statistics)
    for other in results[1:]:
        assert other.anova_p_value == pytest.approx(results[0].anova_p_value)
        assert other.correlation is None


def _jittered_hex_lattice(seed: int = 1) -> tuple[np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    spacing = 10.0
    rows, columns = 40, 30
    points = np.array(
        [
            (i * spacing * np.sqrt(3) / 2, 5.0 + j * spacing + (spacing / 2 if i % 2 else 0))
            for i in range(rows)
            for j in range(columns)
        ]
    )
    points += rng.normal(0, 1.5, points.shape)
    height = int(round(rows * spacing * np.sqrt(3) / 2))
    width = int(columns * spacing + 10)
    points[:, 0] %= height
    return points, height, width


def test_voronoi_boundary_mask_flags_only_lateral_edges_when_periodic() -> None:
    points, height, width = _jittered_hex_lattice()
    flagged = voronoi_boundary_mask(points, (height, width), periodic_y=True)
    columns = points[:, 1]
    assert flagged[columns < 8].all()
    assert flagged[columns > columns.max() - 3].all()
    interior = (columns > 30) & (columns < columns.max() - 30)
    assert not flagged[interior].any()


def test_voronoi_boundary_mask_flags_top_and_bottom_without_periodicity() -> None:
    points, height, width = _jittered_hex_lattice()
    flagged = voronoi_boundary_mask(points, (height, width), periodic_y=False)
    rows = points[:, 0]
    assert flagged[rows < 3].all()
    assert flagged[rows > height - 3].all()


def test_excluding_boundary_removes_edge_inflation_of_psi() -> None:
    points, height, width = _jittered_hex_lattice()
    magnitudes = np.abs(compute_periodic_order_parameter(points, 6, height, True))
    flagged = voronoi_boundary_mask(points, (height, width), periodic_y=True)
    # Flagged cells are biased upward relative to the bulk.
    assert magnitudes[flagged].mean() > magnitudes[~flagged].mean() + 0.05


def test_smoothing_ignores_excluded_regions() -> None:
    points = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float)
    psi = np.array([1.0, np.nan, 0.5, 0.0], dtype=complex)
    psi[1] = complex(np.nan, np.nan)
    smoothed = smooth_order_magnitudes(points, psi, 1)
    assert np.isnan(smoothed[1])
    assert smoothed[0] == pytest.approx(0.5)
    # Region 2's nearest valid neighbor is region 3 (distance 1), not region 0.
    assert smoothed[2] == pytest.approx(0.0)


def test_bootstrap_ignores_excluded_regions() -> None:
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 100, (50, 2))
    psi = np.full(50, 0.5 + 0j)
    psi[:5] = complex(np.nan, np.nan)
    results = spatial_block_bootstrap(
        1,
        points,
        psi,
        np.ones(50),
        n_fold=6,
        block_size_pixels=25,
        replicates=20,
        confidence_level=0.9,
        seed=0,
        image_height=100,
        periodic_y=False,
    )
    by_metric = {item.metric: item for item in results}
    assert by_metric["cell_mean_psi_magnitude"].estimate == pytest.approx(0.5)


def test_analyze_frame_marks_boundary_cells_and_can_keep_legacy(
    tmp_path: Path, write_segmentation
) -> None:
    cells = [(row, column) for row in range(2, 62, 6) for column in range(2, 62, 6)]
    write_segmentation(tmp_path, 1, cells, shape=(64, 64))
    path = next(tmp_path.glob("*.h5"))
    excluded = analyze_frame(path, 1, AnalysisConfig(data_dir=tmp_path, fourier_rect=None))
    assert excluded.segmentation_qc.voronoi_boundary_count > 0
    assert np.isnan(excluded.psi_magnitude).sum() == (
        excluded.segmentation_qc.voronoi_boundary_count
    )
    legacy = analyze_frame(
        path,
        1,
        AnalysisConfig(data_dir=tmp_path, fourier_rect=None, exclude_voronoi_boundary=False),
    )
    assert np.isfinite(legacy.psi_magnitude).all()
    assert legacy.segmentation_qc.voronoi_boundary_count == 0
