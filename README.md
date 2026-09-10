# Blender Fuse

Blender Fuse is a Python package for tissue-order analysis from HDF5 segmentation time series. It computes weighted Voronoi order parameters (ψₙ), spatial summaries, Fourier intensity metrics, and optional PIVlab-driven ROI evolution.

> **Research status:** version 0.1 is an alpha research package. Validate settings and methods for your experiment before drawing scientific conclusions.

## Installation

From a cloned repository:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

Video creation is optional:

```bash
python -m pip install ".[video]"
```

After this repository is uploaded, it can also be installed directly from GitHub (replace `hehenohouse` with the repository owner):

```bash
python -m pip install "blender-fuse @ git+https://github.com/hehenohouse/blender-fuse.git"
```

## Python API

```python
from blender_fuse import AnalysisConfig, run_analysis

config = AnalysisConfig(
    data_dir="raw_data/23C",
    output_dir="analysis_results",
    start_t=1,
    end_t=150,
    n_fold=6,
    smooth_neighbors=6,
    fourier_rect=(700, 900, 800, 1000),
)

result = run_analysis(config)
print(result.processed_timepoints)
print(result.output_dir)
```

`run_analysis` streams frames and returns lightweight summaries by default. Set `outputs.retain_frame_data=True` only when all full-resolution arrays are needed in memory. To analyze one file without writing anything:

```python
from blender_fuse import AnalysisConfig, analyze_frame

config = AnalysisConfig(data_dir="raw_data/23C", fourier_rect=None)
frame = analyze_frame("raw_data/23C/sample_T0001_Simple Segmentation.h5", 1, config)
print(frame.psi_magnitude)
print(frame.psi_map)
```

The array products are deliberately distinct:

- `frame.psi`: one complex ψₙ value per segmented region;
- `frame.psi_magnitude`: one scalar |ψₙ| per region;
- `frame.psi_map`: a two-dimensional scalar map with NaN background;
- PNG files: visual renderings only, never reused as numerical ψ data.

## Command line

```bash
blender-fuse analyze raw_data/23C --start 1 --end 150 \
  --fourier-rect 700 900 800 1000
```

Useful options:

```text
--no-fourier                 Skip Fourier analysis
--no-images                  Do not write PNG files
--no-arrays                  Do not write per-frame NPZ files
--no-periodic-y              Disable vertical periodic boundaries
--shift PIXELS               Roll analysis/display coordinates vertically
--strict-missing             Fail when any requested timepoint is absent
--piv-mat FILE.mat           Enable PIVlab ROI evolution
--roi X,Y                    Add an ROI vertex (repeat at least three times)
--video                      Build MP4 output (requires the video extra)
```

Inspect every option with:

```bash
blender-fuse analyze --help
```

PIV vector-field frames can be rendered separately:

```bash
blender-fuse piv-frames raw_data/PIV/PIVlab_23C.mat piv_frames
```

The historical commands still work after installation:

```bash
python code_2.py
python create_piv_static_frames.py INPUT.mat OUTPUT_DIR
```

They are compatibility launchers; new integrations should use the package API or CLI.

## Input contract

### Segmentation H5 files

The default filename pattern is:

```text
*T0001_Simple Segmentation.h5
*T0002_Simple Segmentation.h5
...
```

By default, Blender Fuse recursively selects the first H5 dataset. If a file contains multiple datasets, set `dataset_name` explicitly. The selected dataset must:

- squeeze to exactly two dimensions;
- have a numeric or boolean dtype;
- contain no NaN or infinite values;
- use values `0` and `2` as background by default.

Change `filename_pattern`, `dataset_name`, or `background_values` in `AnalysisConfig` when your format differs. If no matching files exist, analysis fails before creating output directories.

### PIVlab MAT files

PIV is optional and disabled by default. A PIVlab file must contain:

- `u_original` and `v_original`: per-frame displacement fields;
- `x` and `y`: physical image-coordinate grids.

Velocity values are interpreted as pixels per frame. ROI vertices and coordinate grids must therefore use the same pixel coordinate system.

## Coordinate conventions

- Public rectangles and polygon vertices use `(x, y)`.
- Internal centroid arrays use `(row, column)`, equivalent to `(y, x)`.
- Conversion to `skimage` masks is centralized and explicit.
- A positive `vertical_shift` rolls images and centroid rows downward.
- With `periodic_y=True`, vertical order-parameter neighbors and ROI movement wrap at the top/bottom boundary. Horizontal boundaries remain open/clamped.

ROI means are finite-pixel (cell-area-weighted) means of the scalar |ψₙ| map; NaN background is excluded.

PIV ROI time alignment is:

1. the configured initial polygon belongs to `start_t`;
2. the PIV field for `t → t+1` advances that polygon;
3. the evolved polygon is measured and drawn on frame `t+1`.

## Numerical methods

### Weighted Voronoi ψₙ

Foreground pixels are grouped into regions with 8-neighbor connectivity, matching the original 2-D `skimage.measure.label` behavior. For each region centroid `i`, Blender Fuse uses finite Voronoi ridges to calculate:

```text
ψₙ(i) = Σⱼ wᵢⱼ exp(i n θᵢⱼ) / Σⱼ wᵢⱼ
wᵢⱼ = (Voronoi ridge length)²
```

`θᵢⱼ` is measured from the positive x/column axis. Complex ψ phase therefore follows the standard image-coordinate orientation; magnitude is invariant to this orientation choice.

The squared-ridge weighting is retained from the original research implementation. This differs from equal-weight and edge-length-weight definitions used elsewhere, so report the definition when comparing results.

Vertical periodicity is implemented by augmenting centroids at `y-H`, `y`, and `y+H`, then retaining values only for the original copy. Segmentation objects and statistical samples are not triplicated.

### Smoothing

`smooth_neighbors` is independent from `n_fold`. By default, each region receives the mean |ψₙ| of its nearest neighboring regions, excluding itself. Set `smooth_include_self=True` to include the focal value as well.

### Fourier metric

The configured rectangle is extracted from the binary segmentation image (after any configured vertical shift). The package computes a shifted 2-D FFT, sets the central DC **intensity** to zero, and reports:

```text
maximum non-DC intensity / mean non-DC intensity
```

`top_frequency_bins` selects the highest-valued frequency pixels. These are frequency bins, not guaranteed distinct local peaks. Reciprocal visualizations use the original complex coefficients at those bins rather than treating intensity as a complex amplitude.

### Statistics

For each frame, Blender Fuse reports:

- Pearson correlation between y row and ψ magnitude: a test of linear association only;
- one-way ANOVA across y bins: a test of equality of bin means only.

A value above the threshold is described as **failing to reject** the null hypothesis. It does not prove independence or a uniform distribution. Summary CSV/plots include raw p-values and Benjamini–Hochberg adjusted q-values across timepoints. Spatial autocorrelation and experimental design may require a more specialized statistical model.

## Output layout

Requested outputs are written beneath `output_dir` (default: `data_dir/analysis_output`):

```text
analysis_output/
├── data/
│   ├── order/*.npz
│   ├── fourier/*.npz
│   ├── frequency_bins/*.csv
│   ├── frame_metrics.csv
│   ├── spatial_statistics.csv
│   └── roi_*.csv
├── images/
│   ├── order/
│   ├── order_smoothed/
│   ├── histograms*/
│   ├── fourier/
│   ├── reciprocal/
│   └── roi*/
└── videos/*.mp4
```

NPZ order files contain the numerical labels, centroids, complex ψ values, magnitude arrays, and scalar maps. CSV files provide table equivalents for statistical plots.

## Testing without original data

The test suite builds temporary H5 and MAT fixtures and synthetic lattices, so the original experiment files are not required:

```bash
python -m pytest
```

It checks periodic boundaries, scalar/RGB separation, asymmetric ROI coordinates, Fourier bins, statistical edge cases, PIV time alignment, CLI behavior, and a multi-frame end-to-end pipeline.

## Repository layout

```text
src/blender_fuse/        Installable package
examples/                Small API/statistical examples
tests/                   Synthetic automated tests
legacy/code_2_original.py  Unsupported archival implementation
code_2.py                Legacy compatibility launcher
```

The archive is retained because the initial repository had no Git history. It is not installed or imported by the package and contains known defects.

## License

MIT — see [LICENSE](LICENSE).
