"""Command-line interface for Blender Fuse."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from . import __version__
from .config import AnalysisConfig
from .fourier import validate_rectangle
from .io import discover_timepoint_files, inspect_segmentation_h5
from .pipeline import run_analysis
from .piv import create_piv_static_frames, load_pivlab_data
from .profiles import load_analysis_profile, merge_analysis_config


def _point(value: str) -> Tuple[float, float]:
    try:
        x_text, y_text = value.split(",", maxsplit=1)
        return float(x_text), float(y_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Expected an ROI vertex formatted as X,Y; got {value!r}."
        ) from exc


def _analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("data_dir", nargs="?", type=Path, help="Directory containing H5 files.")
    parser.add_argument("--profile", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--start", dest="start_t", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--end", dest="end_t", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--dataset", dest="dataset_name", default=argparse.SUPPRESS)
    parser.add_argument("--n-fold", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--smooth-neighbors", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--shift", dest="vertical_shift", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--histogram-bin-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--top-bins", dest="top_frequency_bins", type=int, default=argparse.SUPPRESS
    )
    parser.add_argument("--frame-interval-minutes", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--fps", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--fourier-rect",
        nargs=4,
        type=int,
        metavar=("X1", "X2", "Y1", "Y2"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-fourier",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Disable rectangular legacy and v2 Fourier analysis.",
    )
    periodic = parser.add_mutually_exclusive_group()
    periodic.add_argument("--periodic-y", action="store_true", default=argparse.SUPPRESS)
    periodic.add_argument("--no-periodic-y", action="store_true", default=argparse.SUPPRESS)
    missing = parser.add_mutually_exclusive_group()
    missing.add_argument("--strict-missing", action="store_true", default=argparse.SUPPRESS)
    missing.add_argument("--allow-missing", action="store_true", default=argparse.SUPPRESS)

    parser.add_argument("--min-area", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-area", type=int, default=argparse.SUPPRESS)
    border = parser.add_mutually_exclusive_group()
    border.add_argument("--exclude-border", action="store_true", default=argparse.SUPPRESS)
    border.add_argument("--keep-border", action="store_true", default=argparse.SUPPRESS)

    parser.add_argument("--fourier-v2", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument(
        "--fourier-window",
        choices=("none", "hann", "hamming", "blackman"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fourier-min-frequency-cpp", type=float, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--fourier-max-frequency-cpp", type=float, default=argparse.SUPPRESS
    )
    parser.add_argument("--fourier-peak-pairs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--fourier-radial-bin-cpp", type=float, default=argparse.SUPPRESS)

    bootstrap = parser.add_mutually_exclusive_group()
    bootstrap.add_argument("--bootstrap", action="store_true", default=argparse.SUPPRESS)
    bootstrap.add_argument("--no-bootstrap", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-replicates", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-block-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-confidence", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-seed", type=int, default=argparse.SUPPRESS)

    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true", default=argparse.SUPPRESS)
    resume.add_argument("--no-resume", action="store_true", default=argparse.SUPPRESS)
    input_hash = parser.add_mutually_exclusive_group()
    input_hash.add_argument("--input-checksums", action="store_true", default=argparse.SUPPRESS)
    input_hash.add_argument(
        "--no-input-checksums", action="store_true", default=argparse.SUPPRESS
    )
    output_hash = parser.add_mutually_exclusive_group()
    output_hash.add_argument("--output-checksums", action="store_true", default=argparse.SUPPRESS)
    output_hash.add_argument(
        "--no-output-checksums", action="store_true", default=argparse.SUPPRESS
    )


def _output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-images", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--no-arrays", action="store_true", default=argparse.SUPPRESS)
    video = parser.add_mutually_exclusive_group()
    video.add_argument("--video", action="store_true", default=argparse.SUPPRESS)
    video.add_argument("--no-video", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--retain-frame-data", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--piv-mat", type=Path, default=argparse.SUPPRESS)
    parser.add_argument(
        "--piv-interpolation",
        choices=("bilinear", "bicubic", "inverse_distance", "nearest"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--roi",
        action="append",
        type=_point,
        metavar="X,Y",
        default=argparse.SUPPRESS,
        help="ROI vertex; repeat at least three times when PIV is enabled.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blender-fuse",
        description="Analyze tissue order parameters, Fourier structure, and PIV ROIs.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--verbose", action="store_true", help="Show detailed messages.")
    subparsers = parser.add_subparsers(dest="command")

    analyze = subparsers.add_parser(
        "analyze", help="Analyze a time series of segmentation H5 files."
    )
    _analysis_arguments(analyze)
    _output_arguments(analyze)

    inspect = subparsers.add_parser(
        "inspect", help="Validate and describe inputs without creating outputs."
    )
    _analysis_arguments(inspect)

    piv_frames = subparsers.add_parser(
        "piv-frames", help="Render static vector-field images from a PIVlab MAT file."
    )
    piv_frames.add_argument("mat_file", type=Path)
    piv_frames.add_argument("output_dir", type=Path)
    piv_frames.add_argument("--step", type=int, default=2)
    piv_frames.add_argument("--dpi", type=int, default=200)
    return parser


def _nested_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    supplied = vars(args)
    overrides: Dict[str, Any] = {}
    for name in (
        "output_dir",
        "start_t",
        "end_t",
        "dataset_name",
        "n_fold",
        "smooth_neighbors",
        "vertical_shift",
        "histogram_bin_size",
        "top_frequency_bins",
        "frame_interval_minutes",
        "fps",
    ):
        if name in supplied:
            overrides[name] = supplied[name]
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if "fourier_rect" in supplied:
        overrides["fourier_rect"] = tuple(args.fourier_rect)
    if supplied.get("no_fourier"):
        overrides["fourier_rect"] = None
    if supplied.get("periodic_y"):
        overrides["periodic_y"] = True
    if supplied.get("no_periodic_y"):
        overrides["periodic_y"] = False
    if supplied.get("strict_missing"):
        overrides["strict_missing"] = True
    if supplied.get("allow_missing"):
        overrides["strict_missing"] = False

    filter_values: Dict[str, Any] = {}
    if "min_area" in supplied:
        filter_values["min_area_pixels"] = args.min_area
    if "max_area" in supplied:
        filter_values["max_area_pixels"] = args.max_area
    if supplied.get("exclude_border"):
        filter_values["exclude_border"] = True
    if supplied.get("keep_border"):
        filter_values["exclude_border"] = False
    if filter_values:
        overrides["segmentation_filter"] = filter_values

    v2: Dict[str, Any] = {}
    if supplied.get("fourier_v2"):
        v2["enabled"] = True
    mapping = {
        "fourier_window": "window",
        "fourier_min_frequency_cpp": "min_frequency_cycles_per_pixel",
        "fourier_max_frequency_cpp": "max_frequency_cycles_per_pixel",
        "fourier_peak_pairs": "top_pairs",
        "fourier_radial_bin_cpp": "radial_bin_width_cycles_per_pixel",
    }
    for source, destination in mapping.items():
        if source in supplied:
            v2[destination] = supplied[source]
    if v2:
        overrides["fourier_v2"] = v2

    bootstrap_values: Dict[str, Any] = {}
    if supplied.get("bootstrap"):
        bootstrap_values["enabled"] = True
    if supplied.get("no_bootstrap"):
        bootstrap_values["enabled"] = False
    for source, destination in (
        ("bootstrap_replicates", "replicates"),
        ("bootstrap_block_size", "block_size_pixels"),
        ("bootstrap_confidence", "confidence_level"),
        ("bootstrap_seed", "seed"),
    ):
        if source in supplied:
            bootstrap_values[destination] = supplied[source]
    if bootstrap_values:
        overrides["spatial_bootstrap"] = bootstrap_values

    if supplied.get("resume"):
        overrides["resume"] = {"enabled": True}
    if supplied.get("no_resume"):
        overrides["resume"] = {"enabled": False}
    provenance: Dict[str, Any] = {}
    if supplied.get("input_checksums"):
        provenance["hash_inputs"] = True
    if supplied.get("no_input_checksums"):
        provenance["hash_inputs"] = False
    if supplied.get("output_checksums"):
        provenance["hash_outputs"] = True
    if supplied.get("no_output_checksums"):
        provenance["hash_outputs"] = False
    if provenance:
        overrides["provenance"] = provenance
    return overrides


def _build_config(args: argparse.Namespace) -> AnalysisConfig:
    supplied = vars(args)
    if "profile" in supplied:
        config = load_analysis_profile(args.profile)
    elif args.data_dir is not None:
        config = AnalysisConfig(data_dir=args.data_dir)
    else:
        raise ValueError("Provide DATA_DIR or --profile PROFILE.json.")

    overrides = _nested_overrides(args)
    outputs: Dict[str, Any] = {}
    piv: Dict[str, Any] = {}
    if supplied.get("no_images"):
        outputs.update(
            {
                "save_order_images": False,
                "save_smoothed_images": False,
                "save_histograms": False,
                "save_smoothed_histograms": False,
                "save_fourier_images": False,
                "save_reciprocal_overlay": False,
                "save_fourier_v2_images": False,
            }
        )
        piv.update({"save_roi_images": False, "save_roi_fourier": False})
    if supplied.get("no_arrays"):
        outputs.update(
            {
                "save_order_arrays": False,
                "save_fourier_arrays": False,
                "save_fourier_v2_arrays": False,
            }
        )
    if supplied.get("video"):
        outputs["save_videos"] = True
    if supplied.get("no_video"):
        outputs["save_videos"] = False
    if supplied.get("retain_frame_data"):
        outputs["retain_frame_data"] = True
    if "piv_mat" in supplied:
        piv.update({"enabled": True, "mat_path": args.piv_mat})
    if "piv_interpolation" in supplied:
        piv["interpolation"] = args.piv_interpolation
    if "roi" in supplied:
        piv["roi_polygon"] = tuple(args.roi)
    if outputs:
        overrides["outputs"] = outputs
    if piv:
        overrides["piv"] = piv

    v2_cli_options = (
        "fourier_v2",
        "fourier_window",
        "fourier_min_frequency_cpp",
        "fourier_max_frequency_cpp",
        "fourier_peak_pairs",
        "fourier_radial_bin_cpp",
    )
    if supplied.get("no_fourier") and (
        config.fourier_v2.enabled or any(name in supplied for name in v2_cli_options)
    ):
        raise ValueError("--no-fourier conflicts with Fourier v2 options.")
    merged = merge_analysis_config(config, overrides)
    if "profile" in supplied:
        merged._profile_path = Path(args.profile).resolve()
    return merged


def inspect_analysis(config: AnalysisConfig) -> Dict[str, Any]:
    """Return input/config metadata without creating any filesystem products."""

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
    records = []
    for timepoint, path in sorted(files.items()):
        info = inspect_segmentation_h5(path, config.dataset_name)
        if config.fourier_rect is not None:
            validate_rectangle(config.fourier_rect, info.squeezed_shape)
        records.append(
            {
                "timepoint": timepoint,
                "file": path.name,
                "dataset": info.dataset_name,
                "stored_shape": list(info.stored_shape),
                "squeezed_shape": list(info.squeezed_shape),
                "dtype": info.dtype,
            }
        )
    return {
        "schema_version": "blender-fuse.inspect.v1",
        "data_dir": str(config.data_dir.resolve()),
        "prospective_output_dir": str(config.resolved_output_dir.resolve()),
        "file_count": len(records),
        "timepoints": [record["timepoint"] for record in records],
        "missing_timepoints": missing,
        "dataset_names": sorted({record["dataset"] for record in records}),
        "stored_shapes": sorted({tuple(record["stored_shape"]) for record in records}),
        "squeezed_shapes": sorted({tuple(record["squeezed_shape"]) for record in records}),
        "dtypes": sorted({record["dtype"] for record in records}),
        "dataset_consistent": len({record["dataset"] for record in records}) == 1,
        "shape_consistent": len({tuple(record["squeezed_shape"]) for record in records}) == 1,
        "dtype_consistent": len({record["dtype"] for record in records}) == 1,
        "settings": {
            "periodic_y": config.periodic_y,
            "vertical_shift": config.vertical_shift,
            "fourier_rect": config.fourier_rect,
            "fourier_v2": {
                "enabled": config.fourier_v2.enabled,
                "window": config.fourier_v2.window,
                "min_frequency_cycles_per_pixel": (
                    config.fourier_v2.min_frequency_cycles_per_pixel
                ),
                "max_frequency_cycles_per_pixel": (
                    config.fourier_v2.max_frequency_cycles_per_pixel
                ),
                "top_pairs": config.fourier_v2.top_pairs,
                "radial_bin_width_cycles_per_pixel": (
                    config.fourier_v2.radial_bin_width_cycles_per_pixel
                ),
            },
            "segmentation_filter": {
                "enabled": config.segmentation_filter.enabled,
                "min_area_pixels": config.segmentation_filter.min_area_pixels,
                "max_area_pixels": config.segmentation_filter.max_area_pixels,
                "exclude_border": config.segmentation_filter.exclude_border,
            },
            "spatial_bootstrap_enabled": config.spatial_bootstrap.enabled,
            "resume_enabled": config.resume.enabled,
            "piv_enabled": config.piv.enabled,
        },
        "files": records,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "piv-frames":
            data = load_pivlab_data(args.mat_file)
            paths = create_piv_static_frames(data, args.output_dir, step=args.step, dpi=args.dpi)
            print(f"Created {len(paths)} PIV frame images in {args.output_dir}")
            return 0
        config = _build_config(args)
        if args.command == "inspect":
            print(json.dumps(inspect_analysis(config), indent=2, ensure_ascii=False))
            return 0
        result = run_analysis(config)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    print(
        f"Processed {len(result.frames)} frame(s); "
        f"resumed {len(result.resumed_timepoints)}; "
        f"skipped {len(result.missing_timepoints)}; outputs: {result.output_dir}"
    )
    return 0
