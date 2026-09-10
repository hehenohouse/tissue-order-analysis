#!/usr/bin/env python3
"""Legacy editable launcher for Blender Fuse.

New code should import :mod:`blender_fuse` or use the ``blender-fuse`` command.
This file remains so the historical ``python code_2.py`` workflow still works.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Permit direct execution from a source checkout before an editable install.
_PROJECT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from blender_fuse import AnalysisConfig, OutputOptions, PIVConfig, run_analysis  # noqa: E402

# ---------------------------------------------------------------------------
# Editable legacy defaults
# ---------------------------------------------------------------------------
DATA_DIR = _PROJECT_DIR / "raw_data" / "23C"
OUTPUT_DIR = None  # Defaults to DATA_DIR / "analysis_output"
START_T = 1
END_T = 150
DATASET_NAME = None
N_FOLD = 6
SMOOTH_NEIGHBORS = 6
PERIODIC_Y = True
VERTICAL_SHIFT = 0
FOURIER_RECT = (700, 900, 800, 1000)
TOP_FREQUENCY_BINS = 10
HISTOGRAM_BIN_SIZE = 25
FPS = 2.0

SAVE_IMAGES = True
SAVE_ARRAYS = True
SAVE_VIDEOS = False

PIV_ENABLED = False
PIV_MAT_PATH = _PROJECT_DIR / "raw_data" / "PIV" / "PIVlab_23C.mat"
PIV_INTERPOLATION = "bilinear"
ROI_POLYGON = (
    (800.0, 800.0),
    (800.0, 1000.0),
    (1000.0, 1000.0),
    (1000.0, 800.0),
)


def build_config() -> AnalysisConfig:
    """Translate the editable constants above into the public configuration API."""

    outputs = OutputOptions(
        save_order_images=SAVE_IMAGES,
        save_order_arrays=SAVE_ARRAYS,
        save_smoothed_images=SAVE_IMAGES,
        save_histograms=SAVE_IMAGES,
        save_smoothed_histograms=SAVE_IMAGES,
        save_fourier_images=SAVE_IMAGES,
        save_fourier_arrays=SAVE_ARRAYS,
        save_frequency_bins=SAVE_ARRAYS,
        save_metrics_csv=True,
        save_reciprocal_overlay=SAVE_IMAGES,
        save_statistical_summary=True,
        save_videos=SAVE_VIDEOS,
    )
    piv = PIVConfig(
        enabled=PIV_ENABLED,
        mat_path=PIV_MAT_PATH if PIV_ENABLED else None,
        roi_polygon=ROI_POLYGON,
        interpolation=PIV_INTERPOLATION,
        save_roi_images=SAVE_IMAGES,
        save_roi_fourier=SAVE_IMAGES,
        save_roi_statistics=True,
    )
    return AnalysisConfig(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        start_t=START_T,
        end_t=END_T,
        dataset_name=DATASET_NAME,
        n_fold=N_FOLD,
        smooth_neighbors=SMOOTH_NEIGHBORS,
        periodic_y=PERIODIC_Y,
        vertical_shift=VERTICAL_SHIFT,
        fourier_rect=FOURIER_RECT,
        top_frequency_bins=TOP_FREQUENCY_BINS,
        histogram_bin_size=HISTOGRAM_BIN_SIZE,
        fps=FPS,
        outputs=outputs,
        piv=piv,
    )


def run_pipeline(config: AnalysisConfig | None = None):
    """Compatibility function delegating to :func:`blender_fuse.run_analysis`."""

    return run_analysis(config or build_config())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.warning("code_2.py is a compatibility launcher; prefer the blender_fuse API or CLI.")
    try:
        result = run_pipeline()
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        logging.error("%s", exc)
        return 2
    print(f"Processed {len(result.frames)} frame(s); outputs are in {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
