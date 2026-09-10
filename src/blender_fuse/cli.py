"""Command-line interface for Blender Fuse."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

from . import __version__
from .config import AnalysisConfig, OutputOptions, PIVConfig
from .pipeline import run_analysis
from .piv import create_piv_static_frames, load_pivlab_data


def _point(value: str) -> Tuple[float, float]:
    try:
        x_text, y_text = value.split(",", maxsplit=1)
        return float(x_text), float(y_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Expected an ROI vertex formatted as X,Y; got {value!r}."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blender-fuse",
        description="Analyze tissue order parameters, Fourier structure, and PIV ROIs.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--verbose", action="store_true", help="Show detailed processing messages.")
    subparsers = parser.add_subparsers(dest="command")

    analyze = subparsers.add_parser(
        "analyze", help="Analyze a time series of segmentation H5 files."
    )
    analyze.add_argument("data_dir", type=Path, help="Directory containing H5 files.")
    analyze.add_argument("--output-dir", type=Path)
    analyze.add_argument("--start", dest="start_t", type=int, default=1)
    analyze.add_argument("--end", dest="end_t", type=int)
    analyze.add_argument("--dataset", dest="dataset_name")
    analyze.add_argument("--n-fold", type=int, default=6)
    analyze.add_argument("--smooth-neighbors", type=int, default=6)
    analyze.add_argument("--shift", dest="vertical_shift", type=int, default=0)
    analyze.add_argument("--histogram-bin-size", type=int, default=25)
    analyze.add_argument("--top-bins", dest="top_frequency_bins", type=int, default=10)
    analyze.add_argument(
        "--fourier-rect",
        nargs=4,
        type=int,
        metavar=("X1", "X2", "Y1", "Y2"),
    )
    analyze.add_argument("--no-fourier", action="store_true", help="Disable Fourier analysis.")
    analyze.add_argument(
        "--no-periodic-y", action="store_true", help="Disable vertical periodicity."
    )
    analyze.add_argument(
        "--strict-missing", action="store_true", help="Fail if a requested frame is absent."
    )
    analyze.add_argument("--no-images", action="store_true", help="Do not write PNG output.")
    analyze.add_argument(
        "--no-arrays", action="store_true", help="Do not write per-frame NPZ output."
    )
    analyze.add_argument("--video", action="store_true", help="Create MP4 videos.")
    analyze.add_argument(
        "--retain-frame-data",
        action="store_true",
        help="Retain full frame arrays in the returned result (mainly for API parity).",
    )
    analyze.add_argument("--piv-mat", type=Path, help="Enable PIV using this PIVlab MAT file.")
    analyze.add_argument(
        "--piv-interpolation",
        choices=("bilinear", "bicubic", "inverse_distance", "nearest"),
        default="bilinear",
    )
    analyze.add_argument(
        "--roi",
        action="append",
        type=_point,
        metavar="X,Y",
        help="ROI vertex; repeat at least three times when PIV is enabled.",
    )

    piv_frames = subparsers.add_parser(
        "piv-frames", help="Render static vector-field images from a PIVlab MAT file."
    )
    piv_frames.add_argument("mat_file", type=Path)
    piv_frames.add_argument("output_dir", type=Path)
    piv_frames.add_argument("--step", type=int, default=2)
    piv_frames.add_argument("--dpi", type=int, default=200)

    return parser


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

        outputs = OutputOptions(
            save_order_images=not args.no_images,
            save_order_arrays=not args.no_arrays,
            save_smoothed_images=not args.no_images,
            save_histograms=not args.no_images,
            save_smoothed_histograms=not args.no_images,
            save_fourier_images=not args.no_images,
            save_fourier_arrays=not args.no_arrays,
            save_frequency_bins=True,
            save_metrics_csv=True,
            save_reciprocal_overlay=not args.no_images,
            save_statistical_summary=True,
            save_videos=args.video,
            retain_frame_data=args.retain_frame_data,
        )
        roi = tuple(args.roi) if args.roi else PIVConfig().roi_polygon
        piv = PIVConfig(
            enabled=args.piv_mat is not None,
            mat_path=args.piv_mat,
            roi_polygon=roi,
            interpolation=args.piv_interpolation,
            save_roi_images=not args.no_images,
            save_roi_fourier=not args.no_images,
            save_roi_statistics=True,
        )
        if args.no_fourier:
            fourier_rect = None
        elif args.fourier_rect is not None:
            fourier_rect = tuple(args.fourier_rect)
        else:
            fourier_rect = AnalysisConfig(data_dir=args.data_dir).fourier_rect

        config = AnalysisConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            start_t=args.start_t,
            end_t=args.end_t,
            dataset_name=args.dataset_name,
            n_fold=args.n_fold,
            smooth_neighbors=args.smooth_neighbors,
            periodic_y=not args.no_periodic_y,
            vertical_shift=args.vertical_shift,
            fourier_rect=fourier_rect,
            top_frequency_bins=args.top_frequency_bins,
            histogram_bin_size=args.histogram_bin_size,
            strict_missing=args.strict_missing,
            outputs=outputs,
            piv=piv,
        )
        result = run_analysis(config)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    print(
        f"Processed {len(result.frames)} frame(s); "
        f"skipped {len(result.missing_timepoints)}; outputs: {result.output_dir}"
    )
    return 0
