from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from blender_fuse import AnalysisConfig, OutputOptions, run_analysis
from blender_fuse.config import ProvenanceConfig, ResumeConfig


def quiet_outputs() -> OutputOptions:
    return OutputOptions(
        save_order_images=False,
        save_smoothed_images=False,
        save_histograms=False,
        save_smoothed_histograms=False,
        save_fourier_images=False,
        save_reciprocal_overlay=False,
        save_statistical_summary=True,
        save_videos=False,
        save_fourier_v2_images=False,
    )


def test_resume_rehydrates_complete_frame_and_statistics_state(
    tmp_path: Path, write_segmentation
) -> None:
    data = tmp_path / "data"
    points = [(3, 3), (3, 8), (8, 3), (8, 8), (13, 13)]
    for timepoint in (1, 2):
        write_segmentation(data, timepoint, points, shape=(20, 20))
    config = AnalysisConfig(
        data_dir=data,
        output_dir=tmp_path / "output",
        start_t=1,
        end_t=2,
        fourier_rect=(0, 20, 0, 20),
        outputs=quiet_outputs(),
        provenance=ProvenanceConfig(hash_inputs=True, hash_outputs=True),
    )
    first = run_analysis(config)
    resumed = run_analysis(replace(config, resume=ResumeConfig(enabled=True)))
    assert first.processed_timepoints == resumed.processed_timepoints == [1, 2]
    assert resumed.resumed_timepoints == [1, 2]
    assert len(resumed.statistics) == 2
    assert [item.correlation_q_value for item in first.statistics] == [
        item.correlation_q_value for item in resumed.statistics
    ]
    manifest = config.resolved_output_dir / "run_manifest.json"
    assert '"status": "complete"' in manifest.read_text(encoding="utf-8")


def test_resume_rejects_modified_frame_artifact(tmp_path: Path, write_segmentation) -> None:
    data = tmp_path / "data"
    write_segmentation(
        data,
        1,
        [(3, 3), (3, 8), (8, 3), (8, 8)],
        shape=(16, 16),
    )
    config = AnalysisConfig(
        data_dir=data,
        output_dir=tmp_path / "output",
        start_t=1,
        end_t=1,
        fourier_rect=(0, 16, 0, 16),
        outputs=quiet_outputs(),
    )
    run_analysis(config)
    array_path = config.resolved_output_dir / "data" / "order" / "order_T0001.npz"
    array_path.write_bytes(array_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="size changed"):
        run_analysis(replace(config, resume=ResumeConfig(enabled=True)))


def test_manifest_failure_state_is_written(
    tmp_path: Path, write_segmentation, monkeypatch
) -> None:
    data = tmp_path / "data"
    write_segmentation(data, 1, [(3, 3)], shape=(10, 10))
    config = AnalysisConfig(
        data_dir=data,
        output_dir=tmp_path / "output",
        start_t=1,
        end_t=1,
        fourier_rect=None,
        outputs=quiet_outputs(),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr("blender_fuse.pipeline.analyze_frame", fail)
    with pytest.raises(RuntimeError, match="injected"):
        run_analysis(config)
    manifest = config.resolved_output_dir / "run_manifest.json"
    text = manifest.read_text(encoding="utf-8")
    assert '"status": "failed"' in text
    assert "injected failure" in text


def test_default_run_keeps_legacy_order_npz_schema(tmp_path: Path, write_segmentation) -> None:
    data = tmp_path / "data"
    write_segmentation(
        data,
        1,
        [(3, 3), (3, 8), (8, 3), (8, 8)],
        shape=(16, 16),
    )
    config = AnalysisConfig(
        data_dir=data,
        output_dir=tmp_path / "output",
        start_t=1,
        end_t=1,
        fourier_rect=(0, 16, 0, 16),
        outputs=quiet_outputs(),
    )
    run_analysis(config)
    with np.load(config.resolved_output_dir / "data/order/order_T0001.npz") as saved:
        assert set(saved.files) == {
            "labels",
            "centroids_rc",
            "psi",
            "psi_magnitude",
            "psi_magnitude_map",
            "smoothed_magnitude",
            "smoothed_magnitude_map",
        }


def test_resume_rejects_configuration_and_input_changes(
    tmp_path: Path, write_segmentation
) -> None:
    data = tmp_path / "data"
    source = write_segmentation(
        data,
        1,
        [(3, 3), (3, 8), (8, 3), (8, 8)],
        shape=(16, 16),
    )
    config = AnalysisConfig(
        data_dir=data,
        output_dir=tmp_path / "output",
        start_t=1,
        end_t=1,
        fourier_rect=(0, 16, 0, 16),
        outputs=quiet_outputs(),
    )
    run_analysis(config)
    with pytest.raises(ValueError, match="configuration, input files"):
        run_analysis(replace(config, n_fold=4, resume=ResumeConfig(enabled=True)))

    # Recreate the valid checkpoint, then modify the source while retaining its path.
    run_analysis(config)
    with source.open("r+b") as handle:
        handle.seek(-1, 2)
        final_byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([final_byte[0] ^ 1]))
    with pytest.raises(ValueError, match="configuration, input files"):
        run_analysis(replace(config, resume=ResumeConfig(enabled=True)))


def test_resume_rejects_nonempty_directory_without_run_checkpoint(
    tmp_path: Path, write_segmentation
) -> None:
    data = tmp_path / "data"
    write_segmentation(data, 1, [(3, 3)], shape=(10, 10))
    output = tmp_path / "output"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep", encoding="utf-8")
    config = AnalysisConfig(
        data_dir=data,
        output_dir=output,
        start_t=1,
        end_t=1,
        fourier_rect=None,
        outputs=quiet_outputs(),
        resume=ResumeConfig(enabled=True),
    )
    with pytest.raises(ValueError, match="nonempty"):
        run_analysis(config)
    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "keep"


def test_manifest_ledgers_include_source_version_and_run_checkpoint(
    tmp_path: Path, write_segmentation
) -> None:
    import json

    data = tmp_path / "data"
    write_segmentation(data, 1, [(3, 3)], shape=(10, 10))
    config = AnalysisConfig(
        data_dir=data,
        output_dir=tmp_path / "output",
        start_t=1,
        end_t=1,
        fourier_rect=None,
        outputs=quiet_outputs(),
    )
    run_analysis(config)
    manifest = json.loads(
        (config.resolved_output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["software"]["dependencies"]["blender-fuse"] == "0.2.0"
    assert "checkpoints/run.json" in {
        record["path"] for record in manifest["outputs"]
    }
    assert (config.resolved_output_dir / "provenance/input_checksums.sha256").is_file()
    assert (config.resolved_output_dir / "provenance/output_checksums.sha256").is_file()


def test_completed_v2_resume_rebuilds_aggregate_files_without_frame_recomputation(
    tmp_path: Path, write_segmentation, monkeypatch
) -> None:
    from blender_fuse import FourierV2Config

    data = tmp_path / "data"
    points = [(row, column) for row in range(2, 16, 4) for column in range(2, 16, 4)]
    for timepoint in (1, 2):
        write_segmentation(data, timepoint, points, shape=(20, 20))
    outputs = replace(
        quiet_outputs(),
        save_fourier_v2_arrays=True,
        save_fourier_v2_tables=True,
    )
    config = AnalysisConfig(
        data_dir=data,
        output_dir=tmp_path / "output",
        start_t=1,
        end_t=2,
        fourier_rect=(0, 20, 0, 20),
        fourier_v2=FourierV2Config(
            enabled=True,
            window="hann",
            min_frequency_cycles_per_pixel=0.05,
            max_frequency_cycles_per_pixel=0.25,
        ),
        outputs=outputs,
    )
    run_analysis(config)
    aggregate_paths = [
        config.resolved_output_dir / "data/frame_metrics.csv",
        config.resolved_output_dir / "data/frame_metrics_extended.csv",
        config.resolved_output_dir / "data/segmentation_qc.csv",
        config.resolved_output_dir / "data/spatial_statistics.csv",
        config.resolved_output_dir / "data/spatial_statistics_harmonic.csv",
        config.resolved_output_dir / "data/fourier_v2_metrics.csv",
    ]
    expected = {path: path.read_bytes() for path in aggregate_paths}

    def unexpected(*args, **kwargs):
        raise AssertionError("completed frames must not be recomputed")

    monkeypatch.setattr("blender_fuse.pipeline.analyze_frame", unexpected)
    resumed = run_analysis(replace(config, resume=ResumeConfig(enabled=True)))
    assert resumed.resumed_timepoints == [1, 2]
    assert {path: path.read_bytes() for path in aggregate_paths} == expected
