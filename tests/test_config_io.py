from pathlib import Path

import h5py
import numpy as np
import pytest

from blender_fuse import AnalysisConfig
from blender_fuse.io import discover_timepoint_files, load_segmentation_h5


def test_config_validates_without_touching_filesystem(tmp_path: Path) -> None:
    config = AnalysisConfig(data_dir=tmp_path, fourier_rect=(0, 10, 1, 11))
    config.validate()
    assert config.resolved_output_dir == tmp_path / "analysis_output"
    assert not config.resolved_output_dir.exists()


@pytest.mark.parametrize(
    "updates",
    [
        {"n_fold": 0},
        {"end_t": 0},
        {"histogram_bin_size": 0},
        {"fourier_rect": (10, 5, 0, 2)},
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, updates: dict) -> None:
    config = AnalysisConfig(data_dir=tmp_path, **updates)
    with pytest.raises(ValueError):
        config.validate()


def test_discovery_reports_gaps(tmp_path: Path, write_segmentation) -> None:
    first = write_segmentation(tmp_path, 1, [(3, 3)])
    third = write_segmentation(tmp_path, 3, [(4, 4)])

    files, missing = discover_timepoint_files(
        tmp_path,
        "*T{timepoint:04d}_Simple Segmentation.h5",
        start_t=1,
        end_t=3,
    )

    assert files == {1: first, 3: third}
    assert missing == [2]


def test_discovery_fails_when_no_inputs_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No segmentation H5 files"):
        discover_timepoint_files(
            tmp_path,
            "*T{timepoint:04d}_Simple Segmentation.h5",
            start_t=1,
        )


def test_h5_loader_selects_dataset_and_background(tmp_path: Path) -> None:
    path = tmp_path / "frame.h5"
    data = np.asarray([[2, 1, 0], [1, 1, 2]], dtype=np.uint8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("group/labels", data=data[None, ...])

    segmentation = load_segmentation_h5(path, dataset_name="group/labels")

    np.testing.assert_array_equal(
        segmentation,
        np.asarray([[False, True, False], [True, True, False]]),
    )


def test_h5_loader_rejects_non_2d_data(tmp_path: Path) -> None:
    path = tmp_path / "frame.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("labels", data=np.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="2 dimensions"):
        load_segmentation_h5(path)


def test_background_values_reject_booleans(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="background_values"):
        AnalysisConfig(data_dir=tmp_path, background_values=(True, 2.0))


def test_resume_rejects_piv_and_video_requires_images(tmp_path: Path) -> None:
    from blender_fuse import OutputOptions, PIVConfig, ResumeConfig

    with pytest.raises(ValueError, match="Resume with PIV"):
        AnalysisConfig(
            data_dir=tmp_path,
            fourier_rect=None,
            piv=PIVConfig(enabled=True, mat_path=tmp_path / "piv.mat"),
            resume=ResumeConfig(enabled=True),
        ).validate()
    with pytest.raises(ValueError, match="Video output"):
        AnalysisConfig(
            data_dir=tmp_path,
            fourier_rect=None,
            outputs=OutputOptions(
                save_order_images=False,
                save_smoothed_images=False,
                save_fourier_images=False,
                save_fourier_v2_images=False,
                save_videos=True,
            ),
        ).validate()


def test_disabled_v2_images_do_not_count_as_video_source(tmp_path: Path) -> None:
    from blender_fuse import OutputOptions

    outputs = OutputOptions(
        save_order_images=False,
        save_smoothed_images=False,
        save_fourier_images=False,
        save_fourier_v2_images=True,
        save_videos=True,
    )
    with pytest.raises(ValueError, match="image sequence"):
        AnalysisConfig(
            data_dir=tmp_path,
            fourier_rect=None,
            outputs=outputs,
        ).validate()
