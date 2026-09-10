from pathlib import Path

import numpy as np
from scipy.io import savemat

from blender_fuse import AnalysisConfig, OutputOptions, PIVConfig, run_analysis
from blender_fuse.piv import (
    PIVData,
    VelocityInterpolator,
    evolve_polygon,
    load_pivlab_data,
    polygon_from_display,
    polygon_to_display,
)


def constant_piv(frame_count: int = 2, u: float = 1.0, v: float = 0.0) -> PIVData:
    y_coords, x_coords = np.mgrid[0:20:5j, 0:20:5j]
    shape = x_coords.shape
    return PIVData(
        u_frames=[np.full(shape, u) for _ in range(frame_count)],
        v_frames=[np.full(shape, v) for _ in range(frame_count)],
        x_coords=x_coords,
        y_coords=y_coords,
    )


def test_loads_object_array_pivlab_mat_file(tmp_path: Path) -> None:
    data = constant_piv()
    shape = (data.frame_count, 1)
    u_cells = np.empty(shape, dtype=object)
    v_cells = np.empty(shape, dtype=object)
    x_cells = np.empty(shape, dtype=object)
    y_cells = np.empty(shape, dtype=object)
    for index in range(data.frame_count):
        u_cells[index, 0] = data.u_frames[index]
        v_cells[index, 0] = data.v_frames[index]
        x_cells[index, 0] = data.x_coords
        y_cells[index, 0] = data.y_coords

    path = tmp_path / "pivlab.mat"
    savemat(
        path,
        {
            "u_original": u_cells,
            "v_original": v_cells,
            "x": x_cells,
            "y": y_cells,
        },
    )
    loaded = load_pivlab_data(path)

    assert loaded.frame_count == 2
    np.testing.assert_allclose(loaded.u_frames[0], 1.0)
    np.testing.assert_allclose(loaded.x_coords, data.x_coords)


def test_constant_velocity_advances_polygon_once() -> None:
    data = constant_piv(u=1.5, v=2.0)
    interpolator = VelocityInterpolator(
        data.u_frames[0],
        data.v_frames[0],
        data.x_coords,
        data.y_coords,
        method="bilinear",
    )
    polygon = ((2.0, 3.0), (5.0, 3.0), (5.0, 6.0), (2.0, 6.0))
    evolved = evolve_polygon(polygon, interpolator, (20, 20))
    np.testing.assert_allclose(evolved, np.asarray(polygon) + [1.5, 2.0])


def test_periodic_y_wraps_and_display_transform_round_trips() -> None:
    data = constant_piv(u=0.0, v=3.0)
    interpolator = VelocityInterpolator(
        data.u_frames[0],
        data.v_frames[0],
        data.x_coords,
        data.y_coords,
        method="nearest",
    )
    polygon = ((2.0, 18.0), (5.0, 18.0), (5.0, 19.0))
    evolved = evolve_polygon(polygon, interpolator, (20, 20), periodic_y=True)
    assert evolved[0][1] == 1.0

    displayed = polygon_to_display(evolved, 20, 7)
    restored = polygon_from_display(displayed, 20, 7)
    np.testing.assert_allclose(restored, evolved)


def test_pipeline_associates_evolved_roi_with_next_frame(
    tmp_path: Path, write_segmentation, monkeypatch
) -> None:
    data_dir = tmp_path / "data"
    points = [(3, 3), (3, 8), (8, 3), (8, 8), (13, 13)]
    for timepoint in (1, 2, 3):
        write_segmentation(data_dir, timepoint, points, shape=(20, 20))

    monkeypatch.setattr("blender_fuse.pipeline.load_pivlab_data", lambda _: constant_piv())
    outputs = OutputOptions(
        save_order_images=False,
        save_order_arrays=False,
        save_smoothed_images=False,
        save_histograms=False,
        save_smoothed_histograms=False,
        save_fourier_images=False,
        save_fourier_arrays=False,
        save_frequency_bins=False,
        save_metrics_csv=False,
        save_reciprocal_overlay=False,
        save_statistical_summary=False,
        save_videos=False,
    )
    piv = PIVConfig(
        enabled=True,
        mat_path=tmp_path / "placeholder.mat",
        roi_polygon=((1, 1), (10, 1), (10, 10), (1, 10)),
        interpolation="bilinear",
        save_roi_images=False,
        save_roi_fourier=False,
        save_roi_statistics=False,
    )
    config = AnalysisConfig(
        data_dir=data_dir,
        output_dir=tmp_path / "output",
        start_t=1,
        end_t=3,
        fourier_rect=None,
        outputs=outputs,
        piv=piv,
    )

    result = run_analysis(config)

    assert [item.timepoint for item in result.roi_statistics] == [1, 2, 3]
    assert [item.polygon_xy[0][0] for item in result.roi_statistics] == [1.0, 2.0, 3.0]
    assert len(result.roi_statistics) == 3
