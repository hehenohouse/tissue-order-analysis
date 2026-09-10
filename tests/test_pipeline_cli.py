from pathlib import Path

import numpy as np
import pytest

from blender_fuse import AnalysisConfig, OutputOptions, analyze_frame, run_analysis
from blender_fuse.cli import build_parser, main


def minimal_outputs(**updates) -> OutputOptions:
    values = dict(
        save_order_images=False,
        save_order_arrays=True,
        save_smoothed_images=False,
        save_histograms=False,
        save_smoothed_histograms=False,
        save_fourier_images=False,
        save_fourier_arrays=True,
        save_frequency_bins=True,
        save_metrics_csv=True,
        save_reciprocal_overlay=False,
        save_statistical_summary=True,
        save_videos=False,
        retain_frame_data=True,
    )
    values.update(updates)
    return OutputOptions(**values)


def test_end_to_end_pipeline_writes_numeric_not_rgb_psi(tmp_path: Path, write_segmentation) -> None:
    data_dir = tmp_path / "data"
    points = [
        (5, 5),
        (5, 15),
        (5, 25),
        (15, 5),
        (15, 15),
        (15, 25),
        (25, 5),
        (25, 15),
        (25, 25),
    ]
    for timepoint in (1, 2):
        write_segmentation(data_dir, timepoint, points, shape=(32, 32))

    output_dir = tmp_path / "result"
    config = AnalysisConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        start_t=1,
        end_t=2,
        fourier_rect=(0, 32, 0, 32),
        outputs=minimal_outputs(),
    )
    result = run_analysis(config)

    assert result.processed_timepoints == [1, 2]
    assert all(isinstance(path, Path) for path in result.written_paths)
    assert set(result.retained_frames) == {1, 2}
    frame = result.retained_frames[1]
    assert frame.psi_map.ndim == 2
    assert np.isnan(frame.psi_map[0, 0])

    array_path = output_dir / "data" / "order" / "order_T0001.npz"
    with np.load(array_path) as saved:
        assert set(saved.files) == {
            "labels",
            "centroids_rc",
            "psi",
            "psi_magnitude",
            "psi_magnitude_map",
            "smoothed_magnitude",
            "smoothed_magnitude_map",
        }
        assert saved["psi_magnitude_map"].ndim == 2
        assert saved["smoothed_magnitude_map"].ndim == 2
        assert saved["psi"].ndim == 1
        assert np.iscomplexobj(saved["psi"])
    assert (output_dir / "data" / "spatial_statistics.csv").exists()
    assert (output_dir / "data" / "frame_metrics.csv").exists()


def test_analyze_frame_has_no_output_side_effects(tmp_path: Path, write_segmentation) -> None:
    data_dir = tmp_path / "data"
    path = write_segmentation(
        data_dir,
        1,
        [(4, 4), (4, 10), (10, 4), (10, 10)],
        shape=(16, 16),
    )
    config = AnalysisConfig(data_dir=data_dir, output_dir=tmp_path / "never", fourier_rect=None)

    frame = analyze_frame(path, 1, config)

    assert frame.psi_map.shape == (16, 16)
    assert not config.resolved_output_dir.exists()


def test_strict_missing_fails_before_output(tmp_path: Path, write_segmentation) -> None:
    data_dir = tmp_path / "data"
    write_segmentation(data_dir, 1, [(3, 3)])
    output_dir = tmp_path / "result"
    config = AnalysisConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        start_t=1,
        end_t=2,
        strict_missing=True,
        fourier_rect=None,
    )
    with pytest.raises(FileNotFoundError, match="T0002"):
        run_analysis(config)
    assert not output_dir.exists()


def test_cli_parser_and_help_are_available(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["analyze", "data", "--no-fourier"])
    assert args.command == "analyze"
    assert args.no_fourier
    assert main([]) == 0
    assert "Analyze tissue order" in capsys.readouterr().out
