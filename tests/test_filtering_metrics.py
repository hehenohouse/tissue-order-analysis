import numpy as np
import pytest

from blender_fuse.config import SegmentationFilterConfig
from blender_fuse.order import filter_labeled_components
from blender_fuse.statistics import (
    harmonic_association,
    spatial_block_bootstrap,
    summarize_complex_psi,
)


def test_filtering_is_disabled_by_default_and_preserves_labels() -> None:
    mask = np.asarray([[True, False], [False, True]], dtype=bool)
    result = filter_labeled_components(
        mask, SegmentationFilterConfig(), periodic_y=True
    )
    assert result.labels.max() == 1
    np.testing.assert_array_equal(result.analysis_segmentation, mask)
    assert result.quality.raw_component_count == result.quality.retained_component_count == 1


def test_area_filtering_is_inclusive_and_relabels_sequentially() -> None:
    mask = np.zeros((8, 12), dtype=bool)
    mask[1, 1] = True
    mask[2:4, 4:6] = True
    mask[4:7, 8:11] = True
    result = filter_labeled_components(
        mask,
        SegmentationFilterConfig(min_area_pixels=4, max_area_pixels=9),
        periodic_y=False,
    )
    assert result.quality.raw_component_count == 3
    assert result.quality.retained_component_count == 2
    assert result.quality.rejected_min_area_count == 1
    assert result.quality.rejected_max_area_count == 0
    assert set(np.unique(result.labels)) == {0, 1, 2}
    np.testing.assert_array_equal(result.component_areas, [4, 9])


def test_periodic_y_border_filter_ignores_top_bottom_but_not_sides() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, 3] = True
    mask[4, 0] = True
    result = filter_labeled_components(
        mask,
        SegmentationFilterConfig(exclude_border=True),
        periodic_y=True,
    )
    assert result.quality.rejected_border_count == 1
    assert result.quality.retained_component_count == 1
    assert result.analysis_segmentation[0, 3]


def test_nonperiodic_border_filter_includes_top_bottom() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, 3] = True
    mask[4, 4] = True
    result = filter_labeled_components(
        mask,
        SegmentationFilterConfig(exclude_border=True),
        periodic_y=False,
    )
    assert result.quality.rejected_border_count == 1
    assert result.quality.retained_component_count == 1


def test_filtering_every_component_returns_stable_empty_products() -> None:
    mask = np.eye(4, dtype=bool)
    result = filter_labeled_components(
        mask,
        SegmentationFilterConfig(min_area_pixels=10),
        periodic_y=False,
    )
    assert result.labels.max(initial=0) == 0
    assert result.centroids_rc.shape == (0, 2)
    assert result.component_areas.size == 0


def test_complex_summary_distinguishes_cell_and_area_weighting() -> None:
    psi = np.asarray([1 + 0j, 1j], dtype=complex)
    cell = summarize_complex_psi(psi, 1)
    area = summarize_complex_psi(psi, 1, np.asarray([1.0, 9.0]))
    assert cell.mean_real == pytest.approx(0.5)
    assert cell.mean_imag == pytest.approx(0.5)
    assert area.mean_real == pytest.approx(0.1)
    assert area.mean_imag == pytest.approx(0.9)
    assert 0 <= area.phase_coherence <= 1


def test_harmonic_association_recovers_periodic_peak() -> None:
    rows = np.arange(0, 100, 2, dtype=float)
    points = np.column_stack((rows, np.zeros_like(rows)))
    peak_row = 25.0
    values = 2.0 + 0.5 * np.cos(2 * np.pi * (rows - peak_row) / 100)
    result = harmonic_association(
        3, points, values, 100, basis="raw", periodic_y=True
    )
    assert result.r_squared == pytest.approx(1.0)
    assert result.peak_row == pytest.approx(peak_row)
    assert result.p_value == 0.0


def test_harmonic_is_unperformed_without_periodic_y() -> None:
    points = np.asarray([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    result = harmonic_association(
        0, points, np.arange(4), 10, basis="raw", periodic_y=False
    )
    assert result.r_squared is None
    assert result.p_value is None


def test_spatial_block_bootstrap_is_deterministic() -> None:
    points = np.asarray([[1, 1], [2, 2], [11, 1], [12, 2]], dtype=float)
    psi = np.asarray([1, 0.5j, -1, -0.5j], dtype=complex)
    areas = np.asarray([1, 2, 3, 4], dtype=float)
    first = spatial_block_bootstrap(
        5,
        points,
        psi,
        areas,
        n_fold=6,
        block_size_pixels=10,
        replicates=30,
        confidence_level=0.9,
        seed=42,
        image_height=20,
        periodic_y=True,
    )
    second = spatial_block_bootstrap(
        5,
        points,
        psi,
        areas,
        n_fold=6,
        block_size_pixels=10,
        replicates=30,
        confidence_level=0.9,
        seed=42,
        image_height=20,
        periodic_y=True,
    )
    assert first == second
    assert all(item.confidence_low <= item.confidence_high for item in first)


def test_bootstrap_does_not_impute_undefined_metrics_as_zero() -> None:
    points = np.asarray([[1, 1], [2, 2], [11, 1], [12, 2]], dtype=float)
    results = spatial_block_bootstrap(
        0,
        points,
        np.zeros(4, dtype=complex),
        np.ones(4),
        n_fold=6,
        block_size_pixels=10,
        replicates=10,
        confidence_level=0.9,
        seed=1,
        image_height=20,
        periodic_y=True,
    )
    by_metric = {result.metric: result for result in results}
    assert by_metric["phase_coherence_cell"].estimate is None
    assert by_metric["phase_coherence_cell"].confidence_low is None
    assert by_metric["harmonic_r_squared_raw"].estimate is None
    assert by_metric["harmonic_r_squared_raw"].confidence_high is None
