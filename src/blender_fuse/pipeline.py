"""Streaming orchestration for Blender Fuse analyses."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .checkpoint import CheckpointManager, restore_checkpoint
from .config import AnalysisConfig
from .fourier import (
    analyze_centroid_polygon_fourier,
    analyze_fourier,
    analyze_fourier_v2,
    analyze_polygon_fourier,
    center_crop,
)
from .io import discover_timepoint_files, load_segmentation_h5
from .models import (
    AnalysisResult,
    FourierResult,
    FourierV2Result,
    FrameAnalysis,
    FrameSummary,
    HarmonicResult,
    SegmentationQC,
)
from .order import (
    compute_periodic_order_parameter,
    filter_labeled_components,
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
    save_fourier_v2_map,
    save_fourier_v2_radial,
    save_fourier_v2_series,
    save_histogram,
    save_reciprocal_overlay,
    save_roi_series,
    save_scalar_map,
    save_statistical_summary,
)
from .provenance import prepare_run_provenance
from .statistics import (
    adjust_harmonic_statistics,
    adjust_timepoint_statistics,
    harmonic_association,
    roi_psi_statistics,
    spatial_block_bootstrap,
    spatial_statistics,
    summarize_complex_psi,
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
    components = filter_labeled_components(
        segmentation,
        config.segmentation_filter,
        periodic_y=config.periodic_y,
    )
    labels = components.labels
    centroids = components.centroids_rc
    areas = components.component_areas
    analysis_segmentation = components.analysis_segmentation
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
    if config.smooth_neighbors > 0:
        smoothed_magnitudes = smooth_order_magnitudes(
            centroids,
            psi,
            config.smooth_neighbors,
            image_height=segmentation.shape[0],
            periodic_y=config.periodic_y,
            include_self=config.smooth_include_self,
        )
        smoothed_map = values_to_label_map(labels, smoothed_magnitudes)

    display_segmentation = shift_image_rows(analysis_segmentation, config.vertical_shift)
    fourier: Optional[FourierResult] = None
    if config.fourier_rect is not None:
        fourier = analyze_fourier(
            display_segmentation.astype(float),
            config.fourier_rect,
            top_bins=config.top_frequency_bins,
        )

    fourier_v2: Optional[FourierV2Result] = None
    if config.fourier_v2.enabled:
        assert config.fourier_rect is not None
        assert config.fourier_v2.min_frequency_cycles_per_pixel is not None
        assert config.fourier_v2.max_frequency_cycles_per_pixel is not None
        fourier_v2 = analyze_fourier_v2(
            display_segmentation.astype(float),
            config.fourier_rect,
            min_frequency_cycles_per_pixel=(
                config.fourier_v2.min_frequency_cycles_per_pixel
            ),
            max_frequency_cycles_per_pixel=(
                config.fourier_v2.max_frequency_cycles_per_pixel
            ),
            window=config.fourier_v2.window,
            top_pairs=config.fourier_v2.top_pairs,
            radial_bin_width_cycles_per_pixel=(
                config.fourier_v2.radial_bin_width_cycles_per_pixel
            ),
        )

    values_for_statistics = (
        smoothed_magnitudes if smoothed_magnitudes is not None else magnitudes
    )
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
    harmonic_raw = harmonic_association(
        timepoint,
        display_centroids,
        magnitudes,
        segmentation.shape[0],
        basis="raw_psi_magnitude",
        periodic_y=config.periodic_y,
    )
    harmonic_smoothed = None
    if smoothed_magnitudes is not None:
        harmonic_smoothed = harmonic_association(
            timepoint,
            display_centroids,
            smoothed_magnitudes,
            segmentation.shape[0],
            basis="smoothed_psi_magnitude",
            periodic_y=config.periodic_y,
        )
    complex_cell = summarize_complex_psi(psi, config.n_fold)
    complex_area = summarize_complex_psi(psi, config.n_fold, areas)

    bootstrap = ()
    if config.spatial_bootstrap.enabled:
        bootstrap = spatial_block_bootstrap(
            timepoint,
            display_centroids,
            psi,
            areas,
            n_fold=config.n_fold,
            block_size_pixels=config.spatial_bootstrap.block_size_pixels,
            replicates=config.spatial_bootstrap.replicates,
            confidence_level=config.spatial_bootstrap.confidence_level,
            seed=config.spatial_bootstrap.seed,
            image_height=segmentation.shape[0],
            periodic_y=config.periodic_y,
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
        analysis_segmentation=analysis_segmentation,
        component_areas=areas,
        segmentation_qc=components.quality,
        fourier_v2=fourier_v2,
        harmonic_raw=harmonic_raw,
        harmonic_smoothed=harmonic_smoothed,
        complex_psi_cell=complex_cell,
        complex_psi_area=complex_area,
        bootstrap_results=bootstrap,
    )


def _preflight_piv(config: AnalysisConfig, files: Dict[int, Path]) -> Optional[PIVData]:
    if not config.piv.enabled:
        return None
    assert config.piv.mat_path is not None
    piv_data = load_pivlab_data(config.piv.mat_path)
    last_timepoint = max(files)
    required_fields = max(0, last_timepoint - config.start_t)
    if piv_data.frame_count < required_fields:
        raise ValueError(
            f"PIV data has {piv_data.frame_count} fields, but {required_fields} "
            f"are required to evolve the ROI through T{last_timepoint:04d}."
        )
    return piv_data


def _append_restored_frame(
    result: AnalysisResult,
    checkpoint: dict,
    output_dir: Path,
    image_lists: Dict[str, List[Path]],
) -> None:
    summary, statistics, harmonic, quality, bootstrap, artifacts = restore_checkpoint(
        checkpoint
    )
    result.frames.append(summary)
    if statistics is not None:
        result.statistics.append(statistics)
    result.harmonic_statistics.extend(harmonic)
    if quality is not None:
        result.segmentation_qc[summary.timepoint] = quality
    result.bootstrap_statistics.extend(bootstrap)
    result.resumed_timepoints.append(summary.timepoint)
    for relative in artifacts:
        path = output_dir / relative
        result.written_paths.append(path)
        normalized = path.as_posix()
        if "/images/order/order_" in normalized:
            image_lists["psi"].append(path)
        elif "/images/order_smoothed/smoothed_" in normalized:
            image_lists["smoothed"].append(path)
        elif "/images/fourier/fourier_" in normalized:
            image_lists["fourier"].append(path)
        elif "/images/fourier_v2/fourier_v2_" in normalized:
            image_lists["fourier_v2"].append(path)


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
    piv_data = _preflight_piv(config, files)

    provenance = prepare_run_provenance(config, files, missing)
    output_dir = config.resolved_output_dir
    checkpoints = CheckpointManager(
        output_dir,
        provenance.signature,
        sorted(files),
        resume=config.resume.enabled,
    )
    checkpoints.initialize()
    result = AnalysisResult(output_dir=output_dir, missing_timepoints=missing)
    result.written_paths.append(checkpoints.run_path)
    if config.provenance.enabled:
        result.written_paths.append(provenance.write_running())

    current_polygon = config.piv.roi_polygon
    polygon_time = config.start_t
    roi_polygons: Dict[int, Tuple[Tuple[float, float], ...]] = {}
    image_lists: Dict[str, List[Path]] = {
        "psi": [],
        "smoothed": [],
        "fourier": [],
        "fourier_v2": [],
    }
    started = time.perf_counter()

    try:
        for timepoint, path in sorted(files.items()):
            restored = checkpoints.load(timepoint)
            if restored is not None:
                _append_restored_frame(result, restored, output_dir, image_lists)
                result.written_paths.append(checkpoints.frame_path(timepoint))
                if config.outputs.retain_frame_data:
                    result.retained_frames[timepoint] = analyze_frame(path, timepoint, config)
                continue

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
                current_polygon = tuple(
                    (
                        float(np.clip(x, 0.0, width - 1.0)),
                        float(
                            y % height
                            if config.periodic_y
                            else np.clip(y, 0.0, height - 1.0)
                        ),
                    )
                    for x, y in current_polygon
                )
                roi_polygons[timepoint] = current_polygon
                result.roi_statistics.append(
                    roi_psi_statistics(timepoint, frame.psi_map, current_polygon)
                )

            written = _write_frame_outputs(
                frame,
                config,
                output_dir,
                roi_polygon=current_polygon if piv_data is not None else None,
            )
            result.written_paths.extend(written.values())
            for role, key in (
                ("psi", "psi_image"),
                ("smoothed", "smoothed_image"),
                ("fourier", "fourier_image"),
                ("fourier_v2", "fourier_v2_image"),
            ):
                if key in written:
                    image_lists[role].append(written[key])

            if frame.statistics is not None:
                result.statistics.append(frame.statistics)
            harmonic = tuple(
                item
                for item in (frame.harmonic_raw, frame.harmonic_smoothed)
                if item is not None
            )
            result.harmonic_statistics.extend(harmonic)
            if frame.segmentation_qc is not None:
                result.segmentation_qc[timepoint] = frame.segmentation_qc
            result.bootstrap_statistics.extend(frame.bootstrap_results)
            summary = _frame_summary(frame)
            result.frames.append(summary)
            if config.outputs.retain_frame_data:
                result.retained_frames[timepoint] = frame
            checkpoint_path = checkpoints.save(
                summary=summary,
                statistics=frame.statistics,
                harmonic=harmonic,
                quality=frame.segmentation_qc,
                bootstrap=frame.bootstrap_results,
                artifacts=tuple(written.values()),
            )
            result.written_paths.append(checkpoint_path)

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

        result.frames.sort(key=lambda item: item.timepoint)
        result.statistics.sort(key=lambda item: item.timepoint)
        result.harmonic_statistics.sort(key=lambda item: (item.basis, item.timepoint))
        adjust_timepoint_statistics(result.statistics)
        adjust_harmonic_statistics(result.harmonic_statistics)
        result.written_paths.extend(
            _write_run_outputs(
                result,
                config,
                output_dir,
                roi_polygons,
                image_lists,
            )
        )
        if config.provenance.enabled:
            result.written_paths.extend(
                provenance.write_complete(
                    processed=result.processed_timepoints,
                    resumed=result.resumed_timepoints,
                    written_paths=result.written_paths,
                    runtime_seconds=time.perf_counter() - started,
                )
            )
        return result
    except Exception as exc:
        if config.provenance.enabled:
            try:
                provenance.write_failed(exc, result.processed_timepoints)
            except Exception:
                LOGGER.exception("Failed to update the run manifest after an analysis error.")
        raise


def _frame_summary(frame: FrameAnalysis) -> FrameSummary:
    return FrameSummary(
        timepoint=frame.timepoint,
        source_path=frame.source_path,
        region_count=len(frame.centroids_rc),
        mean_psi_magnitude=_finite_mean(frame.psi_magnitude),
        mean_smoothed_psi_magnitude=_finite_mean(frame.smoothed_magnitude),
        fourier_metric=frame.fourier.metric if frame.fourier else None,
        raw_region_count=(
            frame.segmentation_qc.raw_component_count if frame.segmentation_qc else None
        ),
        area_weighted_mean_psi_magnitude=_weighted_mean(
            frame.psi_magnitude, frame.component_areas
        ),
        area_weighted_mean_smoothed_psi_magnitude=_weighted_mean(
            frame.smoothed_magnitude, frame.component_areas
        ),
        fourier_v2_metric=frame.fourier_v2.metric if frame.fourier_v2 else None,
        fourier_v2_wavelength_pixels=(
            frame.fourier_v2.dominant_wavelength_pixels if frame.fourier_v2 else None
        ),
        fourier_v2_radial_frequency_cycles_per_pixel=(
            frame.fourier_v2.peak_pairs[0].radial_frequency_cycles_per_pixel
            if frame.fourier_v2 and frame.fourier_v2.peak_pairs
            else None
        ),
        fourier_v2_low_frequency_power_fraction=(
            frame.fourier_v2.low_frequency_power_fraction if frame.fourier_v2 else None
        ),
        fourier_v2_eligible_power_fraction=(
            frame.fourier_v2.eligible_power_fraction if frame.fourier_v2 else None
        ),
        complex_psi_cell=frame.complex_psi_cell,
        complex_psi_area=frame.complex_psi_area,
    )


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
        _write_legacy_fourier_outputs(frame, config, output_dir, minutes, time_label, paths)
    if frame.fourier_v2 is not None:
        _write_fourier_v2_outputs(frame.fourier_v2, config, output_dir, minutes, time_label, paths)

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
            analysis_mask = (
                frame.analysis_segmentation
                if frame.analysis_segmentation is not None
                else frame.segmentation
            )
            masked_result = analyze_polygon_fourier(
                analysis_mask.astype(float),
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


def _write_legacy_fourier_outputs(
    frame: FrameAnalysis,
    config: AnalysisConfig,
    output_dir: Path,
    minutes: float,
    time_label: str,
    paths: Dict[str, Path],
) -> None:
    assert frame.fourier is not None
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
        display_psi = shift_image_rows(frame.psi_map, config.vertical_shift)
        paths["reciprocal_overlay"] = save_reciprocal_overlay(
            display_psi,
            frame.fourier,
            path,
            title=f"{minutes:.1f} min — selected-frequency reconstruction",
        )


def _write_fourier_v2_outputs(
    result: FourierV2Result,
    config: AnalysisConfig,
    output_dir: Path,
    minutes: float,
    time_label: str,
    paths: Dict[str, Path],
) -> None:
    if config.outputs.save_fourier_v2_arrays:
        path = output_dir / "data" / "fourier_v2" / f"fourier_v2_{time_label}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        second_indices = np.asarray(
            [pair.second_index_rc or (-1, -1) for pair in result.peak_pairs], dtype=int
        ).reshape(-1, 2)
        np.savez_compressed(
            path,
            schema_version=np.asarray(result.schema_version),
            rect=np.asarray(result.rect, dtype=int),
            window=np.asarray(result.window),
            removed_mean=result.removed_mean,
            window_power_normalization=result.window_power_normalization,
            spectrum=result.spectrum,
            intensity=result.intensity,
            frequency_y_cycles_per_pixel=result.frequency_y_cycles_per_pixel,
            frequency_x_cycles_per_pixel=result.frequency_x_cycles_per_pixel,
            radial_frequency_cycles_per_pixel=result.radial_frequency_cycles_per_pixel,
            eligible_mask=result.eligible_mask,
            eligible_max_intensity=result.eligible_max_intensity,
            eligible_mean_intensity=result.eligible_mean_intensity,
            metric=result.metric,
            peak_first_indices_rc=np.asarray(
                [pair.first_index_rc for pair in result.peak_pairs], dtype=int
            ).reshape(-1, 2),
            peak_second_indices_rc=second_indices,
            peak_frequency_y_cycles_per_pixel=np.asarray(
                [pair.frequency_y_cycles_per_pixel for pair in result.peak_pairs],
                dtype=float,
            ),
            peak_frequency_x_cycles_per_pixel=np.asarray(
                [pair.frequency_x_cycles_per_pixel for pair in result.peak_pairs],
                dtype=float,
            ),
            peak_radial_frequency_cycles_per_pixel=np.asarray(
                [pair.radial_frequency_cycles_per_pixel for pair in result.peak_pairs],
                dtype=float,
            ),
            peak_wavelength_pixels=np.asarray(
                [pair.wavelength_pixels for pair in result.peak_pairs], dtype=float
            ),
            peak_first_intensity=np.asarray(
                [pair.first_intensity for pair in result.peak_pairs], dtype=float
            ),
            peak_second_intensity=np.asarray(
                [
                    np.nan if pair.second_intensity is None else pair.second_intensity
                    for pair in result.peak_pairs
                ],
                dtype=float,
            ),
            peak_mean_intensity=np.asarray(
                [pair.mean_intensity for pair in result.peak_pairs], dtype=float
            ),
            peak_total_power=np.asarray(
                [pair.total_power for pair in result.peak_pairs], dtype=float
            ),
            peak_member_count=np.asarray(
                [pair.member_count for pair in result.peak_pairs], dtype=int
            ),
            low_frequency_power_fraction=(
                np.nan
                if result.low_frequency_power_fraction is None
                else result.low_frequency_power_fraction
            ),
            eligible_power_fraction=(
                np.nan
                if result.eligible_power_fraction is None
                else result.eligible_power_fraction
            ),
            radial_edges_cycles_per_pixel=result.radial_profile.edges_cycles_per_pixel,
            radial_centers_cycles_per_pixel=result.radial_profile.centers_cycles_per_pixel,
            radial_mean_intensity=result.radial_profile.mean_intensity,
            radial_total_power=result.radial_profile.total_power,
            radial_counts=result.radial_profile.counts,
        )
        paths["fourier_v2_array"] = path
    if config.outputs.save_fourier_v2_tables:
        pair_path = (
            output_dir / "data" / "frequency_pairs_v2" / f"pairs_v2_{time_label}.csv"
        )
        _write_fourier_v2_pairs(result, pair_path)
        paths["fourier_v2_pairs"] = pair_path
        radial_path = (
            output_dir / "data" / "radial_profiles_v2" / f"radial_v2_{time_label}.csv"
        )
        _write_fourier_v2_radial(result, radial_path)
        paths["fourier_v2_radial_table"] = radial_path
    if config.outputs.save_fourier_v2_images:
        path = output_dir / "images" / "fourier_v2" / f"fourier_v2_{time_label}.png"
        paths["fourier_v2_image"] = save_fourier_v2_map(
            result, path, title=f"{minutes:.1f} min — Fourier v2"
        )
        radial_path = (
            output_dir / "images" / "fourier_v2_radial" / f"radial_v2_{time_label}.png"
        )
        paths["fourier_v2_radial_image"] = save_fourier_v2_radial(
            result,
            radial_path,
            title=f"{minutes:.1f} min — radial Fourier v2 power",
        )


def _write_run_outputs(
    result: AnalysisResult,
    config: AnalysisConfig,
    output_dir: Path,
    roi_polygons: Dict[int, Tuple[Tuple[float, float], ...]],
    image_lists: Dict[str, List[Path]],
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

    if config.outputs.save_extended_metrics:
        paths.append(_write_extended_metrics(result, output_dir))
    if config.outputs.save_segmentation_qc:
        paths.append(_write_segmentation_qc(result, config, output_dir))
    if config.outputs.save_harmonic_statistics and result.harmonic_statistics:
        paths.append(_write_harmonic_statistics(result.harmonic_statistics, output_dir))
    if (
        config.outputs.save_bootstrap_statistics
        and config.spatial_bootstrap.enabled
        and result.bootstrap_statistics
    ):
        paths.append(_write_bootstrap_statistics(result, output_dir))
    if config.fourier_v2.enabled:
        paths.append(_write_fourier_v2_metrics(result, output_dir))
        if config.outputs.save_fourier_v2_images and result.frames:
            assert config.fourier_v2.min_frequency_cycles_per_pixel is not None
            assert config.fourier_v2.max_frequency_cycles_per_pixel is not None
            paths.append(
                save_fourier_v2_series(
                    result.frames,
                    output_dir / "images" / "fourier_v2_metrics.png",
                    start_t=config.start_t,
                    frame_interval_minutes=config.frame_interval_minutes,
                    min_frequency_cycles_per_pixel=(
                        config.fourier_v2.min_frequency_cycles_per_pixel
                    ),
                    max_frequency_cycles_per_pixel=(
                        config.fourier_v2.max_frequency_cycles_per_pixel
                    ),
                )
            )

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
            (image_lists["psi"], "order.mp4"),
            (image_lists["smoothed"], "order_smoothed.mp4"),
            (image_lists["fourier"], "fourier.mp4"),
            (image_lists["fourier_v2"], "fourier_v2.mp4"),
        ):
            if images:
                paths.append(
                    create_video(
                        sorted(images), output_dir / "videos" / filename, config.fps
                    )
                )
    return paths


def _write_extended_metrics(result: AnalysisResult, output_dir: Path) -> Path:
    path = output_dir / "data" / "frame_metrics_extended.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = (
        "mean_real",
        "mean_imag",
        "resultant_magnitude",
        "orientation_radians",
        "orientation_degrees",
        "phase_coherence",
    )
    header = [
        "timepoint",
        "source_path",
        "raw_region_count",
        "retained_region_count",
        "cell_mean_psi_magnitude",
        "area_weighted_mean_psi_magnitude",
        "cell_mean_smoothed_psi_magnitude",
        "area_weighted_mean_smoothed_psi_magnitude",
        "legacy_fourier_metric",
        "fourier_v2_metric",
        "fourier_v2_dominant_radial_frequency_cycles_per_pixel",
        "fourier_v2_dominant_wavelength_pixels",
        "fourier_v2_low_frequency_power_fraction",
        "fourier_v2_eligible_power_fraction",
    ]
    header.extend(f"complex_psi_cell_{name}" for name in summary_fields)
    header.extend(f"complex_psi_area_{name}" for name in summary_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for frame in result.frames:
            row = [
                frame.timepoint,
                frame.source_path,
                _csv_value(frame.raw_region_count),
                frame.region_count,
                _csv_value(frame.mean_psi_magnitude),
                _csv_value(frame.area_weighted_mean_psi_magnitude),
                _csv_value(frame.mean_smoothed_psi_magnitude),
                _csv_value(frame.area_weighted_mean_smoothed_psi_magnitude),
                _csv_value(frame.fourier_metric),
                _csv_value(frame.fourier_v2_metric),
                _csv_value(frame.fourier_v2_radial_frequency_cycles_per_pixel),
                _csv_value(frame.fourier_v2_wavelength_pixels),
                _csv_value(frame.fourier_v2_low_frequency_power_fraction),
                _csv_value(frame.fourier_v2_eligible_power_fraction),
            ]
            for summary in (frame.complex_psi_cell, frame.complex_psi_area):
                row.extend(
                    _csv_value(getattr(summary, name)) if summary is not None else ""
                    for name in summary_fields
                )
            writer.writerow(row)
    return path


def _write_segmentation_qc(
    result: AnalysisResult, config: AnalysisConfig, output_dir: Path
) -> Path:
    path = output_dir / "data" / "segmentation_qc.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(SegmentationQC.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timepoint",
                "source_path",
                *fields,
                "raw_foreground_fraction",
                "retained_foreground_fraction",
                "min_area_pixels",
                "max_area_pixels",
                "exclude_border",
            ]
        )
        by_time = {frame.timepoint: frame for frame in result.frames}
        for timepoint, quality in sorted(result.segmentation_qc.items()):
            pixels = quality.image_height * quality.image_width
            writer.writerow(
                [
                    timepoint,
                    by_time[timepoint].source_path,
                    *(getattr(quality, name) for name in fields),
                    quality.raw_foreground_pixels / pixels,
                    quality.retained_foreground_pixels / pixels,
                    _csv_value(config.segmentation_filter.min_area_pixels),
                    _csv_value(config.segmentation_filter.max_area_pixels),
                    config.segmentation_filter.exclude_border,
                ]
            )
    return path


def _write_harmonic_statistics(results: Sequence[HarmonicResult], output_dir: Path) -> Path:
    path = output_dir / "data" / "spatial_statistics_harmonic.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timepoint",
                "basis",
                "cosine_coefficient",
                "sine_coefficient",
                "amplitude",
                "peak_row",
                "r_squared",
                "f_statistic",
                "p_value",
                "q_value_bh_within_basis",
            ]
        )
        for item in sorted(results, key=lambda value: (value.timepoint, value.basis)):
            writer.writerow(
                [
                    item.timepoint,
                    item.basis,
                    _csv_value(item.cosine_coefficient),
                    _csv_value(item.sine_coefficient),
                    _csv_value(item.amplitude),
                    _csv_value(item.peak_row),
                    _csv_value(item.r_squared),
                    _csv_value(item.f_statistic),
                    _csv_value(item.p_value),
                    _csv_value(item.q_value),
                ]
            )
    return path


def _write_bootstrap_statistics(result: AnalysisResult, output_dir: Path) -> Path:
    path = output_dir / "data" / "spatial_block_bootstrap.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timepoint",
                "metric",
                "estimate",
                "confidence_low",
                "confidence_high",
                "occupied_blocks",
                "replicates",
            ]
        )
        ordered = sorted(
            result.bootstrap_statistics,
            key=lambda value: (value.timepoint, value.metric),
        )
        for item in ordered:
            writer.writerow(
                [
                    item.timepoint,
                    item.metric,
                    _csv_value(item.estimate),
                    _csv_value(item.confidence_low),
                    _csv_value(item.confidence_high),
                    item.occupied_blocks,
                    item.replicates,
                ]
            )
    return path


def _write_fourier_v2_metrics(result: AnalysisResult, output_dir: Path) -> Path:
    path = output_dir / "data" / "fourier_v2_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timepoint",
                "metric",
                "dominant_radial_frequency_cycles_per_pixel",
                "dominant_wavelength_pixels",
                "low_frequency_power_fraction",
                "eligible_power_fraction",
            ]
        )
        for frame in result.frames:
            writer.writerow(
                [
                    frame.timepoint,
                    _csv_value(frame.fourier_v2_metric),
                    _csv_value(frame.fourier_v2_radial_frequency_cycles_per_pixel),
                    _csv_value(frame.fourier_v2_wavelength_pixels),
                    _csv_value(frame.fourier_v2_low_frequency_power_fraction),
                    _csv_value(frame.fourier_v2_eligible_power_fraction),
                ]
            )
    return path

def _write_fourier_v2_pairs(result: FourierV2Result, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "first_row",
                "first_column",
                "second_row",
                "second_column",
                "frequency_y_cycles_per_pixel",
                "frequency_x_cycles_per_pixel",
                "radial_frequency_cycles_per_pixel",
                "wavelength_pixels",
                "first_intensity",
                "second_intensity",
                "pair_mean_intensity",
                "pair_total_power",
                "member_count",
            ]
        )
        for pair in result.peak_pairs:
            second = pair.second_index_rc or (None, None)
            writer.writerow(
                [
                    pair.rank,
                    pair.first_index_rc[0],
                    pair.first_index_rc[1],
                    _csv_value(second[0]),
                    _csv_value(second[1]),
                    pair.frequency_y_cycles_per_pixel,
                    pair.frequency_x_cycles_per_pixel,
                    pair.radial_frequency_cycles_per_pixel,
                    pair.wavelength_pixels,
                    pair.first_intensity,
                    _csv_value(pair.second_intensity),
                    pair.mean_intensity,
                    pair.total_power,
                    pair.member_count,
                ]
            )
    return path


def _write_fourier_v2_radial(result: FourierV2Result, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = result.radial_profile
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "lower_frequency_cycles_per_pixel",
                "upper_frequency_cycles_per_pixel",
                "center_frequency_cycles_per_pixel",
                "wavelength_pixels",
                "mean_intensity",
                "total_power",
                "bin_count",
            ]
        )
        for lower, upper, center, mean, total, count in zip(
            profile.edges_cycles_per_pixel[:-1],
            profile.edges_cycles_per_pixel[1:],
            profile.centers_cycles_per_pixel,
            profile.mean_intensity,
            profile.total_power,
            profile.counts,
        ):
            writer.writerow(
                [
                    lower,
                    upper,
                    center,
                    1.0 / center if center > 0 else "",
                    _csv_value(float(mean) if np.isfinite(mean) else None),
                    total,
                    count,
                ]
            )
    return path


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


def _weighted_mean(
    values: Optional[np.ndarray], weights: np.ndarray
) -> Optional[float]:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    sample_weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(array) & np.isfinite(sample_weights) & (sample_weights > 0)
    return float(np.average(array[valid], weights=sample_weights[valid])) if np.any(valid) else None


def _finite_mean(values: Optional[np.ndarray]) -> Optional[float]:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else None


def _csv_value(value: object) -> object:
    return "" if value is None else value
