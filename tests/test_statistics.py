from pathlib import Path

import numpy as np

from blender_fuse.models import StatisticalResult
from blender_fuse.statistics import (
    adjust_timepoint_statistics,
    anova_conclusion,
    benjamini_hochberg,
    pearson_conclusion,
    roi_psi_statistics,
    spatial_statistics,
    write_statistics_csv,
)


def test_anova_reports_difference_without_claiming_dependence() -> None:
    rng = np.random.default_rng(4)
    rows = np.repeat([10.0, 40.0, 70.0], 15)
    columns = rng.uniform(0, 100, len(rows))
    values = np.concatenate(
        (
            rng.normal(0.2, 0.01, 15),
            rng.normal(0.5, 0.01, 15),
            rng.normal(0.8, 0.01, 15),
        )
    )
    result = spatial_statistics(
        1, np.column_stack((rows, columns)), values, image_height=90, bin_size=30
    )

    assert result.anova_p_value is not None
    assert result.anova_p_value < 0.05
    conclusion = anova_conclusion(result)
    assert "at least one row-bin mean differs" in conclusion
    assert "independ" not in conclusion.lower()
    assert "uniform" not in conclusion.lower()


def test_constant_data_returns_unperformed_tests() -> None:
    points = np.asarray([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=float)
    result = spatial_statistics(1, points, np.ones(4), image_height=10, bin_size=5)
    assert result.correlation is None
    assert result.anova_p_value is None
    assert "not performed" in pearson_conclusion(result)


def test_benjamini_hochberg_preserves_order_and_missing_values() -> None:
    adjusted = benjamini_hochberg([0.01, 0.02, 0.20, None])
    np.testing.assert_allclose(adjusted[:3], [0.03, 0.03, 0.20])
    assert adjusted[3] is None


def test_adjusted_values_are_written_to_csv(tmp_path: Path) -> None:
    results = [
        StatisticalResult(1, 0.4, 0.01, 3.0, 0.02),
        StatisticalResult(2, 0.1, 0.5, 0.2, 0.8),
    ]
    adjust_timepoint_statistics(results)
    path = write_statistics_csv(results, tmp_path / "stats.csv")
    text = path.read_text(encoding="utf-8")
    assert "correlation_q_value_bh" in text
    assert "Failed to reject no linear association" in text


def test_roi_average_uses_scalar_values_and_excludes_nan_background() -> None:
    psi_map = np.full((12, 20), np.nan)
    psi_map[2:6, 10:16] = 0.75
    polygon = ((10, 2), (16, 2), (16, 6), (10, 6))

    result = roi_psi_statistics(3, psi_map, polygon)

    assert result.average_psi_magnitude == 0.75
    assert result.valid_psi_pixels > 0
    assert result.geometric_area_pixels >= result.valid_psi_pixels
