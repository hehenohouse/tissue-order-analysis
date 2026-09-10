"""Streaming orchestration for Blender Fuse analyses."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import AnalysisConfig
from .fourier import (
    analyze_centroid_polygon_fourier,
    analyze_fourier,
    analyze_polygon_fourier,
    center_crop,
)
from .io import discover_timepoint_files, load_segmentation_h5
from .models import AnalysisResult, FourierResult, FrameAnalysis, FrameSummary
from .order import (
    compute_periodic_order_parameter,
    label_segmentation,
    shift_image_rows,
    shift_points_for_display,
    smooth_order_magnitudes,
    values_to_label_map,
)
from .piv import (
    PIVData,
    VelocityInterpolator,
    evolve_polygon,
    load_pivlab_data,
    polygon_to_display,
)
from .plotting import (
    create_video,
    save_fourier_map,
    save_histogram,
    save_reciprocal_overlay,
    save_roi_series,
    save_scalar_map,
    save_statistical_summary,
)
from .statistics import (
    adjust_timepoint_statistics,
    roi_psi_statistics,
    spatial_statistics,
    write_statistics_csv,
)

LOGGER = logging.getLogger(__name__)


def analyze_frame(path: Path, timepoint: int, config: AnalysisConfig) -> FrameAnalysis:
    """Analyze one H5 frame without writing files."""

    config.validate()
    segmentation = load_segmentation_h5(
        path,
        dataset_name=config.dataset_name,
        background_values=config.background_values,
    )
    labels, centroids = label_segmentation(segmentation)
    psi = compute_periodic_order_parameter(
        centroids,
        config.n_fold,
        image_height=segmentation.shape[0],
        periodic_y=config.periodic_y,
    )
    magnitudes = np.asarray(np.abs(psi), dtype=float)
    psi_map = values_to_label_map(labels, magnitudes)

    smoothed_magnitudes: Optional[np.ndarray] = None
    smoothed_map: Optional[np.ndarray] = None
    needs_smoothing = config.smooth_neighbors > 0
    if needs_smoothing:
        smoothed_magnitudes = smooth_order_magnitudes(
            centroids,
            psi,
            config.smooth_neighbors,
            image_height=segmentation.shape[0],
            periodic_y=config.periodic_y,
            include_self=config.smooth_include_self,
        )
        smoothed_map = values_to_label_map(labels, smoothed_magnitudes)

    display_segmentation = shift_image_rows(segmentation, config.vertical_shift)
    fourier: Optional[FourierResult] = None
    if config.fourier_rect is not None:
        fourier = analyze_fourier(
            display_segmentation.astype(float),
            config.fourier_rect,
            top_bins=config.top_frequency_bins,
        )

    values_for_statistics = smoothed_magnitudes if smoothed_magnitudes is not None else magnitudes
    display_centroids = shift_points_for_display(
        centroids, segmentation.shape[0], config.vertical_shift
    )
    statistics = spatial_statistics(
        timepoint,
        display_centroids,
        values_for_statistics,
        segmentation.shape[0],
        config.histogram_bin_size,
    )

    return FrameAnalysis(
        timepoint=int(timepoint),
        source_path=Path(path),
        segmentation=segmentation,
        labels=labels,
        centroids_rc=centroids,
        psi=psi,
        psi_magnitude=magnitudes,
        psi_map=psi_map,
        smoothed_magnitude=smoothed_magnitudes,
        smoothed_map=smoothed_map,
        fourier=fourier,
        statistics=statistics,
    )


def run_analysis(config: AnalysisConfig) -> AnalysisResult:
    """Run a validated, streaming analysis and return structured summaries."""

    config.validate()
    files, missing = discover_timepoint_files(
        config.data_dir,
        config.filename_pattern,
        start_t=config.start_t,
        end_t=config.end_t,
    )
    if missing and config.strict_missing:
        formatted = ", ".join(f"T{timepoint:04d}" for timepoint in missing)
        raise FileNotFoundError(f"Missing requested segmentation timepoints: {formatted}")
    if missing:
        LOGGER.warning("Skipping missing timepoints: %s", missing)

    output_dir = config.resolved_output_dir
    result = AnalysisResult(output_dir=output_dir, missing_timepoints=missing)

    piv_data: Optional[PIVData] = None
    current_polygon = config.piv.roi_polygon
    polygon_time = config.start_t
    if config.piv.enabled:
        assert config.piv.mat_path is not None
        piv_data = load_pivlab_data(config.piv.mat_path)
        last_timepoint = max(files)
        required_fields = max(0, last_timepoint - config.start_t)
        if piv_data.frame_count < required_fields:
            raise ValueError(
                f"PIV data has {piv_data.frame_count} fields, but {required_fields} "
                f"are required to evolve the ROI through T{last_timepoint:04d}."
            )

    psi_image_paths: List[Path] = []
    smoothed_image_paths: List[Path] = []
    fourier_image_paths: List[Path] = []
    roi_polygons: Dict[int, Tuple[Tuple[float, float], ...]] = {}

    for timepoint, path in sorted(files.items()):
        frame = analyze_frame(path, timepoint, config)
        height, width = frame.segmentation.shape

        if piv_data is not None:
            while polygon_time < timepoint:
                piv_index = polygon_time - config.start_t
                interpolator = VelocityInterpolator(
                    piv_data.u_frames[piv_index],
                    piv_data.v_frames[piv_index],
                    piv_data.x_coords,
                    piv_data.y_coords,
                    method=config.piv.interpolation,
                )
                current_polygon = evolve_polygon(
                    current_polygon,
                    interpolator,
                    frame.segmentation.shape,
                    periodic_y=config.periodic_y,
                )
                polygon_time += 1

        # Re-clamp a PIV polygon to the actual image and record one polygon per frame.
        if piv_data is not None:
            current_polygon = tuple(
                (
                    float(np.clip(x, 0.0, width - 1.0)),
                    float(y % height if config.periodic_y else np.clip(y, 0.0, height - 1.0)),
                )
                for x, y in current_polygon
            )
            roi_polygons[timepoint] = current_polygon
            roi_result = roi_psi_statistics(timepoint, frame.psi_map, current_polygon)
            result.roi_statistics.append(roi_result)

        written = _write_frame_outputs(
            frame,
            config,
            output_dir,
            roi_polygon=current_polygon if piv_data is not None else None,
        )
        result.written_paths.extend(written.values())
        psi_image_paths.extend(path for name, path in written.items() if name == "psi_image")
        smoothed_image_paths.extend(
            path for name, path in written.items() if name == "smoothed_image"
        )
        fourier_image_paths.extend(
            path for name, path in written.items() if name == "fourier_image"
        )

        if frame.statistics is not None:
            result.statistics.append(frame.statistics)
        result.frames.append(
            FrameSummary(
                timepoint=timepoint,
                source_path=path,
                region_count=len(frame.centroids_rc),
                mean_psi_magnitude=_finite_mean(frame.psi_magnitude),
                mean_smoothed_psi_magnitude=_finite_mean(frame.smoothed_magnitude),
                fourier_metric=(frame.fourier.metric if frame.fourier else None),
            )
        )
        if config.outputs.retain_frame_data:
            result.retained_frames[timepoint] = frame

        # Advance exactly once from this frame to the next timepoint. The evolved
        # polygon is therefore associated with t+1, never with t.
        if piv_data is not None and polygon_time == timepoint:
            piv_index = timepoint - config.start_t
            if piv_index < piv_data.frame_count:
                interpolator = VelocityInterpolator(
                    piv_data.u_frames[piv_index],
                    piv_data.v_frames[piv_index],
                    piv_data.x_coords,
                    piv_data.y_coords,
                    method=config.piv.interpolation,
                )
                current_polygon = evolve_polygon(
                    current_polygon,
                    interpolator,
                    frame.segmentation.shape,
                    periodic_y=config.periodic_y,
                )
                polygon_time += 1

    adjust_timepoint_statistics(result.statistics)
    result.written_paths.extend(
        _write_run_outputs(
            result,
            config,
            output_dir,
            roi_polygons,
            psi_image_paths,
            smoothed_image_paths,
            fourier_image_paths,
        )
    )
    return result


def _write_frame_outputs(
    frame: FrameAnalysis,
    config: AnalysisConfig,
    output_dir: Path,
    *,
    roi_polygon: Optional[Sequence[Tuple[float, float]]],
) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    time_label = f"T{frame.timepoint:04d}"
    minutes = (frame.timepoint - config.start_t) * config.frame_interval_minutes
    display_psi = shift_image_rows(frame.psi_map, config.vertical_shift)
    display_smoothed = (
        shift_image_rows(frame.smoothed_map, config.vertical_shift)
        if frame.smoothed_map is not None
        else None
    )

    if config.outputs.save_order_arrays:
        path = output_dir / "data" / "order" / f"order_{time_label}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            labels=frame.labels,
            centroids_rc=frame.centroids_rc,
            psi=frame.psi,
            psi_magnitude=frame.psi_magnitude,
            psi_magnitude_map=frame.psi_map,
            smoothed_magnitude=(
                frame.smoothed_magnitude
                if frame.smoothed_magnitude is not None
                else np.empty(0, dtype=float)
            ),
            smoothed_magnitude_map=(
                frame.smoothed_map
                if frame.smoothed_map is not None
                else np.empty((0, 0), dtype=float)
            ),
        )
        paths["order_array"] = path

    if config.outputs.save_order_images:
        path = output_dir / "images" / "order" / f"order_{time_label}.png"
        paths["psi_image"] = save_scalar_map(
            display_psi,
            path,
            title=f"{minutes:.1f} min — |ψ{config.n_fold}|",
            colorbar_label=f"|ψ{config.n_fold}|",
        )

    if config.outputs.save_smoothed_images and display_smoothed is not None:
        path = output_dir / "images" / "order_smoothed" / f"smoothed_{time_label}.png"
        paths["smoothed_image"] = save_scalar_map(
            display_smoothed,
            path,
            title=(
                f"{minutes:.1f} min — smoothed |ψ{config.n_fold}| "
                f"({config.smooth_neighbors} neighbors)"
            ),
            colorbar_label=f"Smoothed |ψ{config.n_fold}|",
        )

    if frame.statistics is not None:
        display_centroids = shift_points_for_display(
            frame.centroids_rc, frame.segmentation.shape[0], config.vertical_shift
        )
        raw_statistics = spatial_statistics(
            frame.timepoint,
            display_centroids,
            frame.psi_magnitude,
            frame.segmentation.shape[0],
            config.histogram_bin_size,
        )
        if config.outputs.save_histograms:
            path = output_dir / "images" / "histograms" / f"order_{time_label}.png"
            paths["histogram"] = save_histogram(
                raw_statistics,
                path,
                title=f"{minutes:.1f} min — mean |ψ{config.n_fold}| by Y bin",
                ylabel=f"Mean |ψ{config.n_fold}|",
            )
        if config.outputs.save_smoothed_histograms and frame.smoothed_map is not None:
            path = output_dir / "images" / "histograms_smoothed" / f"smoothed_{time_label}.png"
            paths["smoothed_histogram"] = save_histogram(
                frame.statistics,
                path,
                title=(f"{minutes:.1f} min — mean smoothed |ψ{config.n_fold}| by Y bin"),
                ylabel=f"Mean smoothed |ψ{config.n_fold}|",
            )

    if frame.fourier is not None:
        if config.outputs.save_fourier_arrays:
            path = output_dir / "data" / "fourier" / f"fourier_{time_label}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                spectrum=frame.fourier.spectrum,
                intensity=frame.fourier.intensity,
                metric=frame.fourier.metric,
                bin_indices_rc=frame.fourier.bin_indices_rc,
                offsets_rc=frame.fourier.offsets_rc,
                bin_values=frame.fourier.bin_values,
                distances=frame.fourier.distances,
            )
            paths["fourier_array"] = path
        if config.outputs.save_frequency_bins:
            path = output_dir / "data" / "frequency_bins" / f"bins_{time_label}.csv"
            _write_frequency_bins(frame.fourier, path)
            paths["frequency_bins"] = path
        if config.outputs.save_fourier_images:
            path = output_dir / "images" / "fourier" / f"fourier_{time_label}.png"
            paths["fourier_image"] = save_fourier_map(
                frame.fourier,
                path,
                title=f"{minutes:.1f} min — Fourier intensity",
            )
        if config.outputs.save_reciprocal_overlay:
            path = output_dir / "images" / "reciprocal" / f"reciprocal_{time_label}.png"
            paths["reciprocal_overlay"] = save_reciprocal_overlay(
                display_psi,
                frame.fourier,
                path,
                title=f"{minutes:.1f} min — selected-frequency reconstruction",
            )

    if roi_polygon is not None:
        display_polygon = polygon_to_display(
            roi_polygon, frame.segmentation.shape[0], config.vertical_shift
        )
        if config.piv.save_roi_images:
            source = display_smoothed if display_smoothed is not None else display_psi
            path = output_dir / "images" / "roi" / f"roi_{time_label}.png"
            paths["roi_image"] = save_scalar_map(
                source,
                path,
                title=f"{minutes:.1f} min — ROI on |ψ{config.n_fold}|",
                colorbar_label=f"|ψ{config.n_fold}|",
                polygon_xy=display_polygon,
            )
        if config.piv.save_roi_fourier:
            centroid_result = analyze_centroid_polygon_fourier(
                frame.centroids_rc,
                roi_polygon,
                frame.segmentation.shape,
                top_bins=config.top_frequency_bins,
            )
            masked_result = analyze_polygon_fourier(
                frame.segmentation.astype(float),
                roi_polygon,
                top_bins=config.top_frequency_bins,
            )
            centroid_path = (
                output_dir / "images" / "roi_fourier_centroids" / f"centroids_{time_label}.png"
            )
            masked_path = output_dir / "images" / "roi_fourier_masked" / f"masked_{time_label}.png"
            paths["roi_fourier_centroids"] = save_fourier_map(
                centroid_result,
                centroid_path,
                title=f"{minutes:.1f} min — ROI centroid Fourier intensity",
            )
            cropped_intensity = np.asarray(
                center_crop(masked_result.intensity, config.piv.roi_fourier_crop_size),
                dtype=float,
            )
            paths["roi_fourier_masked"] = save_fourier_map(
                masked_result,
                masked_path,
                title=f"{minutes:.1f} min — ROI masked Fourier intensity",
                intensity_override=cropped_intensity,
                vmax=config.piv.roi_fourier_vmax,
                show_selected_bins=False,
            )

    return paths


def _write_run_outputs(
    result: AnalysisResult,
    config: AnalysisConfig,
    output_dir: Path,
    roi_polygons: Dict[int, Tuple[Tuple[float, float], ...]],
    psi_images: Sequence[Path],
    smoothed_images: Sequence[Path],
    fourier_images: Sequence[Path],
) -> List[Path]:
    paths: List[Path] = []
    if config.outputs.save_metrics_csv:
        path = output_dir / "data" / "frame_metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "timepoint",
                    "source_path",
                    "region_count",
                    "mean_psi_magnitude",
                    "mean_smoothed_psi_magnitude",
                    "fourier_metric",
                ]
            )
            for frame in result.frames:
                writer.writerow(
                    [
                        frame.timepoint,
                        frame.source_path,
                        frame.region_count,
                        _csv_value(frame.mean_psi_magnitude),
                        _csv_value(frame.mean_smoothed_psi_magnitude),
                        _csv_value(frame.fourier_metric),
                    ]
                )
        paths.append(path)

    if config.outputs.save_statistical_summary and result.statistics:
        csv_path = output_dir / "data" / "spatial_statistics.csv"
        write_statistics_csv(result.statistics, csv_path)
        paths.append(csv_path)
        plot_path = output_dir / "images" / "statistical_summary.png"
        paths.append(save_statistical_summary(result.statistics, plot_path))

    if config.piv.enabled and config.piv.save_roi_statistics:
        csv_path = output_dir / "data" / "roi_statistics.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "timepoint",
                    "average_psi_magnitude",
                    "geometric_area_pixels",
                    "valid_psi_pixels",
                    "polygon_xy",
                ]
            )
            for roi_result in result.roi_statistics:
                writer.writerow(
                    [
                        roi_result.timepoint,
                        _csv_value(roi_result.average_psi_magnitude),
                        roi_result.geometric_area_pixels,
                        roi_result.valid_psi_pixels,
                        repr(roi_result.polygon_xy),
                    ]
                )
        paths.append(csv_path)
        if result.roi_statistics:
            plot_path = output_dir / "images" / "roi_psi_series.png"
            paths.append(save_roi_series(result.roi_statistics, plot_path))

        boundary_path = output_dir / "data" / "roi_boundaries.csv"
        boundary_path.parent.mkdir(parents=True, exist_ok=True)
        with boundary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timepoint", "vertex", "x", "y"])
            for timepoint, polygon in sorted(roi_polygons.items()):
                for vertex, (x, y) in enumerate(polygon):
                    writer.writerow([timepoint, vertex, x, y])
        paths.append(boundary_path)

    if config.outputs.save_videos:
        for images, filename in (
            (psi_images, "order.mp4"),
            (smoothed_images, "order_smoothed.mp4"),
            (fourier_images, "fourier.mp4"),
        ):
            if images:
                paths.append(create_video(images, output_dir / "videos" / filename, config.fps))

    return paths


def _write_frequency_bins(result: FourierResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["rank", "row", "column", "row_offset", "column_offset", "intensity", "distance"]
        )
        for rank, (index, offset, value, distance) in enumerate(
            zip(
                result.bin_indices_rc,
                result.offsets_rc,
                result.bin_values,
                result.distances,
            ),
            start=1,
        ):
            writer.writerow([rank, index[0], index[1], offset[0], offset[1], value, distance])
    return path


def _finite_mean(values: Optional[np.ndarray]) -> Optional[float]:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else None


def _csv_value(value: Optional[float]) -> object:
    return "" if value is None else value
