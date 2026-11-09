"""Tests for coordination defects and the g6(r) orientational correlation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from blender_fuse import AnalysisConfig, run_analysis
from blender_fuse.checkpoint import restore_structure
from blender_fuse.config import OutputOptions, StructureConfig
from blender_fuse.order import compute_periodic_order_parameter
from blender_fuse.profiles import analysis_config_from_dict
from blender_fuse.structure import (
    analyze_structure,
    correlation_lengths,
    defect_summary,
    orientational_correlation,
    voronoi_neighbors,
)


def _hex_lattice(rows: int = 20, columns: int = 20, spacing: float = 10.0, noise: float = 0.0):
    rng = np.random.default_rng(0)
    row_step = spacing * np.sqrt(3) / 2
    points = np.array(
        [
            (i * row_step, 5.0 + j * spacing + (spacing / 2 if i % 2 else 0.0))
            for i in range(rows)
            for j in range(columns)
        ]
    )
    points += rng.normal(0.0, noise, points.shape)
    height = int(round(rows * row_step))
    width = int(columns * spacing + 10)
    points[:, 0] %= height
    return points, height, width


def test_perfect_lattice_interior_is_six_coordinated_with_no_defects() -> None:
    points, height, width = _hex_lattice()
    neighbors, boundary = voronoi_neighbors(points, (height, width), periodic_y=True)
    interior = np.flatnonzero(
        ~boundary & np.array([not boundary[n].any() for n in neighbors])
    )
    assert interior.size > 0
    assert all(len(neighbors[i]) == 6 for i in interior)
    summary = defect_summary(neighbors, boundary)
    assert summary["defect_fraction"] == 0.0
    assert summary["mean_coordination"] == pytest.approx(6.0)
    assert summary["dislocation_fraction"] is None


def test_periodic_neighbors_map_back_across_seam() -> None:
    points, height, width = _hex_lattice()
    neighbors, _ = voronoi_neighbors(points, (height, width), periodic_y=True)
    top = int(np.argmin(np.abs(points[:, 0]) + np.abs(points[:, 1] - 105)))
    rows = points[neighbors[top], 0]
    assert np.any(rows > height / 2)  # at least one neighbor sits across the seam
    assert np.all(neighbors[top] < len(points))


def test_defect_summary_counts_charges_and_dislocation_pairs() -> None:
    neighbors = [
        np.array([1, 2, 3, 4, 5]),  # z = 5, touches region 1 (z = 7)
        np.array([0, 2, 3, 4, 5, 6, 7]),  # z = 7
        np.array([0, 1, 3, 4, 5, 6]),  # z = 6
        np.array([0, 1, 2, 4, 5, 6]),
        np.array([0, 1, 2, 3, 5, 6]),
        np.array([0, 1, 2, 3, 4, 6]),
        np.array([1, 2, 3, 4, 5, 7]),
        np.array([1, 6, 8]),  # touches boundary region 8, so excluded
        np.array([7]),
    ]
    boundary = np.zeros(9, dtype=bool)
    boundary[8] = True
    summary = defect_summary(neighbors, boundary)
    assert summary["interior_regions"] == 7
    assert summary["fraction_z5"] == pytest.approx(1 / 7)
    assert summary["fraction_z7"] == pytest.approx(1 / 7)
    assert summary["defect_fraction"] == pytest.approx(2 / 7)
    assert summary["net_charge_per_region"] == pytest.approx(0.0)
    assert summary["dislocation_fraction"] == pytest.approx(1.0)


def test_g6_is_flat_for_perfect_lattice_and_decays_with_disorder() -> None:
    ordered, height, width = _hex_lattice(rows=30, columns=30)
    psi = compute_periodic_order_parameter(ordered, 6, height, True)
    centers, mean, counts = orientational_correlation(
        ordered, psi, (height, width), periodic_y=True, max_distance=60.0, bin_width=5.0
    )
    filled = counts > 0
    assert np.nanmin(mean[filled & (centers > 8)]) > 0.9

    disordered, height, width = _hex_lattice(rows=30, columns=30, noise=3.0)
    psi = compute_periodic_order_parameter(disordered, 6, height, True)
    _, noisy, _ = orientational_correlation(
        disordered, psi, (height, width), periodic_y=True, max_distance=60.0, bin_width=5.0
    )
    assert noisy[-1] < noisy[2]


def test_g6_pairs_wrap_vertically_only_when_periodic() -> None:
    points = np.array([[1.0, 50.0], [99.0, 50.0]])
    psi = np.array([1.0 + 0j, 1.0 + 0j])
    _, _, periodic = orientational_correlation(
        points, psi, (100, 100), periodic_y=True, max_distance=10.0, bin_width=5.0
    )
    _, _, flat = orientational_correlation(
        points, psi, (100, 100), periodic_y=False, max_distance=10.0, bin_width=5.0
    )
    assert periodic.sum() == 1
    assert flat.sum() == 0


def test_g6_ignores_non_finite_psi() -> None:
    points = np.array([[10.0, 10.0], [10.0, 13.0], [10.0, 16.0]])
    psi = np.array([1.0 + 0j, complex(np.nan, np.nan), 1.0 + 0j])
    _, mean, counts = orientational_correlation(
        points, psi, (40, 40), periodic_y=False, max_distance=10.0, bin_width=10.0
    )
    assert counts.sum() == 1
    assert mean[0] == pytest.approx(1.0)


def test_correlation_lengths_recover_exponential_decay() -> None:
    spacing = 10.0
    r = np.arange(1.25, 150, 2.5)
    g = np.exp(-r / 30.0)
    lengths = correlation_lengths(r, g, spacing, fit_floor=0.01)
    assert lengths["xi_1e_pixels"] == pytest.approx(30.0, rel=0.05)
    assert lengths["exp_xi_pixels"] == pytest.approx(30.0, rel=1e-6)
    assert lengths["exp_r_squared"] == pytest.approx(1.0)
    assert lengths["exp_r_squared"] > lengths["power_r_squared"]


def test_analyze_structure_reports_spacing_in_pixels() -> None:
    points, height, width = _hex_lattice(noise=1.0)
    psi = compute_periodic_order_parameter(points, 6, height, True)
    result = analyze_structure(1, points, psi, (height, width), periodic_y=True)
    assert result.mean_spacing_pixels == pytest.approx(np.sqrt(height * width / len(points)))
    assert len(result.g6_r_pixels) == len(result.g6_normalized) == len(result.g6_pair_counts)
    assert max(result.g6_r_pixels) <= height / 2


def test_structure_config_validation_and_profile_round_trip() -> None:
    with pytest.raises(ValueError):
        StructureConfig(bin_width_spacings=20.0, max_distance_spacings=10.0).validate()
    config = analysis_config_from_dict(
        {"data_dir": ".", "structure": {"enabled": True, "max_distance_spacings": 8.0}}
    )
    assert config.structure.enabled
    assert config.structure.max_distance_spacings == 8.0
    with pytest.raises(ValueError):
        analysis_config_from_dict({"data_dir": ".", "structure": {"unknown": 1}})


def test_pipeline_writes_structure_outputs_and_resumes(tmp_path: Path, write_segmentation) -> None:
    rng = np.random.default_rng(5)
    data_dir = tmp_path / "data"
    for timepoint in (1, 2):
        cells = {(int(r), int(c)) for r, c in rng.integers(1, 63, size=(150, 2))}
        write_segmentation(data_dir, timepoint, cells, shape=(64, 64))
    output_dir = tmp_path / "out"
    outputs = OutputOptions(
        save_order_images=False,
        save_order_arrays=False,
        save_smoothed_images=False,
        save_histograms=False,
        save_smoothed_histograms=False,
        save_reciprocal_overlay=False,
        save_statistical_summary=False,
    )
    config = AnalysisConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        start_t=1,
        end_t=2,
        fourier_rect=None,
        outputs=outputs,
        structure=StructureConfig(enabled=True),
    )
    first = run_analysis(config)
    assert [item.timepoint for item in first.structure_results] == [1, 2]
    with (output_dir / "data" / "structure_metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["timepoint"] for row in rows] == ["1", "2"]
    assert float(rows[0]["mean_coordination"]) > 0
    assert (output_dir / "data" / "g6_correlation.csv").stat().st_size > 0
    assert (output_dir / "images" / "structure_summary.png").exists()
    assert (output_dir / "images" / "g6_curves.png").exists()

    config.resume.enabled = True
    resumed = run_analysis(config)
    assert resumed.resumed_timepoints == [1, 2]
    assert resumed.structure_results == first.structure_results


def test_restore_structure_handles_missing_record() -> None:
    assert restore_structure({}) is None


def test_correlation_length_is_flagged_as_upper_bound_when_decay_is_immediate() -> None:
    r = np.arange(0.125, 5, 0.25) * 10.0
    lengths = correlation_lengths(r, np.exp(-r / 3.0), 10.0)
    assert lengths["xi_1e_is_upper_bound"] is True
    assert lengths["xi_1e_pixels"] == pytest.approx(8.75)


def test_spacing_qc_detects_missing_points_but_not_uniform_expansion() -> None:
    from blender_fuse.structure import spacing_qc

    rng = np.random.default_rng(2)
    lattice, height, width = _hex_lattice(rows=60, columns=60, spacing=20.0, noise=1.0)
    full = spacing_qc(lattice, (height, width))
    assert 0.9 < full["nn_distance_ratio"] < 1.1
    assert full["hole_area_fraction"] < 0.02

    dropped = lattice[rng.random(len(lattice)) >= 0.2]
    holes = spacing_qc(dropped, (height, width))
    assert holes["median_nn_distance_pixels"] == pytest.approx(
        full["median_nn_distance_pixels"], rel=0.05
    )
    assert holes["nn_distance_ratio"] < full["nn_distance_ratio"] - 0.08
    assert holes["hole_area_fraction"] > 0.3

    expanded = spacing_qc(lattice * 1.3, (int(height * 1.3), int(width * 1.3)))
    assert expanded["nn_distance_ratio"] == pytest.approx(full["nn_distance_ratio"], abs=0.03)
    assert expanded["hole_area_fraction"] < 0.02


def test_spacing_qc_is_empty_for_tiny_inputs() -> None:
    from blender_fuse.structure import spacing_qc

    result = spacing_qc(np.zeros((3, 2)), (10, 10))
    assert result["hole_area_fraction"] is None
