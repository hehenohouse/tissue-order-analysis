# Blender Fuse

Blender Fuse is an installable Python package and command-line tool for tissue-order analysis from HDF5 segmentation time series. It computes weighted Voronoi order parameters (ψₙ), segmentation quality control, spatial statistics, rectangular Fourier summaries, and optional PIVlab-driven ROI evolution.

> **Research status:** version 0.2 is an alpha research package. Its outputs support quantitative exploration, but experimental settings and statistical assumptions must be validated for each dataset before biological conclusions are drawn.

## Installation

From a cloned repository:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

MP4 creation uses an optional dependency group:

```bash
python -m pip install ".[video]"
```

The package can also be installed directly from GitHub:

```bash
python -m pip install "blender-fuse @ git+https://github.com/hehenohouse/blender-fuse.git"
```

Blender Fuse supports Python 3.9 and newer.

## Quick start

### Python API

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

`run_analysis` streams frames and returns lightweight summaries by default. Set `outputs.retain_frame_data=True` only when full-resolution arrays from every frame must remain in memory.

To analyze one file without writing anything:

```python
from blender_fuse import AnalysisConfig, analyze_frame

config = AnalysisConfig(data_dir="raw_data/23C", fourier_rect=None)
frame = analyze_frame("raw_data/23C/sample_T0001_Simple Segmentation.h5", 1, config)
print(frame.psi_magnitude)
print(frame.psi_map)
```

The numerical products are deliberately distinct:

- `frame.psi`: one complex ψₙ value per retained region;
- `frame.psi_magnitude`: one scalar |ψₙ| per retained region;
- `frame.psi_map`: a two-dimensional scalar map with NaN background;
- PNG files: visual renderings only, never reused as numerical input.

### Command line

```bash
blender-fuse analyze raw_data/23C --start 1 --end 150 \
  --fourier-rect 700 900 800 1000
```

Useful options include:

```text
--profile PROFILE.json         Load a strict, reproducible analysis profile
--no-fourier                  Disable rectangular legacy and v2 Fourier analysis
--fourier-v2                  Enable band-limited Fourier v2
--min-area PIXELS             Reject components smaller than this inclusive bound
--max-area PIXELS             Reject components larger than this inclusive bound
--exclude-border              Reject components touching configured image borders
--bootstrap                   Enable exploratory spatial block bootstrap intervals
--resume                      Validate checkpoints and reuse completed frames
--no-images                   Do not write any frame PNGs, including v2 PNGs
--no-arrays                   Do not write legacy or v2 per-frame NPZ files
--no-periodic-y               Disable vertical periodic boundaries
--shift PIXELS                Roll analysis/display coordinates vertically
--strict-missing              Fail when any requested timepoint is absent
--piv-mat FILE.mat            Enable PIVlab ROI evolution
--roi X,Y                     Add an ROI vertex; repeat at least three times
--video                       Build MP4 output using enabled image sequences
```

Inspect all options with:

```bash
blender-fuse analyze --help
```

### Strict JSON profiles

A profile captures the effective scientific configuration without shell-specific quoting. Paths inside a profile are resolved relative to the profile file. Explicit CLI arguments override profile values; omitted CLI defaults do not leak into the profile.

```bash
blender-fuse inspect --profile examples/profiles/analysis_v2.json
blender-fuse analyze --profile examples/profiles/analysis_v2.json
```

The profile format is versioned and rejects unknown fields and invalid JSON types. See `examples/profiles/analysis_v2.json` for a portable template. The Python helpers `load_analysis_profile` and `write_analysis_profile` provide the same strict conversion.

### Side-effect-free inspection

`inspect` discovers files and reads H5 metadata without loading full pixel arrays or creating output directories, caches, manifests, or logs:

```bash
blender-fuse inspect raw_data/23C --start 1 --end 150 \
  --fourier-rect 700 900 800 1000
```

Its JSON output reports selected timepoints and gaps, dataset names, stored/squeezed shapes, dtypes, consistency flags, prospective output path, and relevant analysis settings. Every configured Fourier rectangle is validated against every observed shape.

### Compatibility commands

PIV vector-field frames can be rendered separately:

```bash
blender-fuse piv-frames raw_data/PIV/PIVlab_23C.mat piv_frames
```

Historical launchers still work after installation:

```bash
python code_2.py
python create_piv_static_frames.py INPUT.mat OUTPUT_DIR
```

They remain compatibility entry points; new integrations should use the package API, profile format, or CLI.

## Input contract

### Segmentation H5 files

The default filename pattern is:

```text
*T0001_Simple Segmentation.h5
*T0002_Simple Segmentation.h5
...
```

Blender Fuse recursively selects the first H5 dataset unless `dataset_name` is explicit. The selected dataset must:

- squeeze to exactly two dimensions;
- have a numeric or Boolean dtype;
- contain no NaN or infinite values;
- use values `0` and `2` as background by default.

Change `filename_pattern`, `dataset_name`, or `background_values` when the source format differs. If no matching files exist, analysis fails before an output directory is created.

### PIVlab MAT files

PIV is optional and disabled by default. A PIVlab file must contain:

- `u_original` and `v_original`: per-frame displacement fields;
- `x` and `y`: physical image-coordinate grids.

Velocity is interpreted as pixels per frame. ROI vertices and grids must use the same pixel coordinate system.

## Coordinate conventions

- Public rectangles and polygons use `(x, y)`.
- Internal centroids use `(row, column)`, equivalent to `(y, x)`.
- A positive `vertical_shift` rolls images and centroid rows downward.
- With `periodic_y=True`, vertical order-parameter neighbors and ROI movement wrap at the top/bottom boundary. Horizontal boundaries remain open or clamped.

PIV ROI alignment is:

1. the initial polygon belongs to `start_t`;
2. the PIV field for `t → t+1` advances it;
3. the evolved polygon is measured and drawn on frame `t+1`.

## Numerical methods

### Connected components and filtering

Foreground pixels use 8-neighbor connectivity, matching the original two-dimensional `skimage.measure.label` behavior. Optional component filters are applied before ψ, smoothing, rectangular Fourier analysis, and ROI summaries:

```python
from blender_fuse import AnalysisConfig, SegmentationFilterConfig

config = AnalysisConfig(
    data_dir="raw_data/23C",
    segmentation_filter=SegmentationFilterConfig(
        min_area_pixels=20,
        max_area_pixels=5000,
        exclude_border=True,
    ),
)
```

Area bounds are inclusive: areas equal to the minimum or maximum are retained. Left and right always count as borders. Top and bottom count as borders only when `periodic_y=False`; version 0.2 does not merge components across the periodic seam. With every filter disabled, original labels and legacy numerical inputs are preserved without relabeling.

`segmentation_qc.csv` reports raw and retained foreground/component counts, overlapping rejection-reason counts, area summaries, thresholds, and source paths.

### Weighted Voronoi ψₙ

For each component centroid `i`, finite Voronoi ridges define:

```text
ψₙ(i) = Σⱼ wᵢⱼ exp(i n θᵢⱼ) / Σⱼ wᵢⱼ
wᵢⱼ = (Voronoi ridge length)²
```

`θᵢⱼ` is measured from the positive x/column axis. Complex phase therefore follows image-coordinate orientation; magnitude is invariant to this orientation choice. The squared-ridge weighting is retained from the original research implementation and differs from equal-weight or edge-length-weight definitions.

Vertical periodicity augments centroids at `y-H`, `y`, and `y+H`, then retains values only for the original copy. Statistical samples are not triplicated.

### Smoothing

`smooth_neighbors` is independent of `n_fold`. By default, each region receives the mean |ψₙ| of its nearest neighboring regions, excluding itself. Set `smooth_include_self=True` to include the focal value.

### Legacy Fourier analysis

The legacy method is frozen for 0.1 compatibility. It extracts the post-shift binary segmentation rectangle, computes `fftshift(fft2(...))`, forms `|FT|²`, and sets the exact central DC intensity to zero. Its concentration metric is:

```text
max(intensity) / mean(intensity)
```

The mean is over the **entire array after the DC slot is zeroed**, so that zeroed slot remains in the denominator. `top_frequency_bins` selects the highest individual bins; conjugate members count separately and are not required to be distinct local peaks. Legacy NPZ/CSV paths, fields, and numerical semantics are unchanged.

### Fourier v2

Fourier v2 is separate and opt-in:

```python
from blender_fuse import AnalysisConfig, FourierV2Config

config = AnalysisConfig(
    data_dir="raw_data/23C",
    fourier_rect=(700, 900, 800, 1000),
    fourier_v2=FourierV2Config(
        enabled=True,
        window="hann",
        min_frequency_cycles_per_pixel=0.025,
        max_frequency_cycles_per_pixel=0.1,
        top_pairs=5,
    ),
)
```

V2 subtracts the rectangle mean, applies a separable `none`, Hann, Hamming, or Blackman window, and normalizes power by `sum(window²)`. Frequencies come independently from each rectangle dimension through `fftshift(fftfreq(...))` and are reported in cycles/pixel. The inclusive eligible radial band excludes exact DC. V2 concentration is eligible maximum divided by eligible mean, and wavelength is `1 / radial_frequency` in pixels.

Real-input conjugates are grouped deterministically in modular unshifted DFT index space. Nyquist self-conjugate bins are singletons. Separate products preserve pair membership, physical frequencies, wavelength, power, eligibility masks, and annular radial profiles. V2 is defined only for rectangular segmentation Fourier in this release; polygon/PIV Fourier remains legacy.

The `0.025–0.1` cycles/pixel band in the example profile corresponds to wavelengths of `10–40` pixels. It is an experiment-specific example, not a universal biological default.

### Extended summaries and statistics

Legacy `frame_metrics.csv` and `spatial_statistics.csv` remain unchanged. New files add:

- equal-cell and component-area-weighted means of raw/smoothed |ψₙ|;
- complex ψ mean, resultant magnitude, n-fold orientation modulo `2π/n`, and phase coherence;
- Fourier v2 concentration, dominant physical frequency/wavelength, and power fractions;
- segmentation QC.

Legacy Pearson and one-way ANOVA tests retain their narrow meanings. Pearson tests linear association with image row; ANOVA tests equality of configured row-bin means. Their Benjamini–Hochberg corrections remain separate families across timepoints. Legacy spatial tests use smoothed magnitudes when smoothing is enabled.

When vertical periodicity is enabled, `spatial_statistics_harmonic.csv` fits:

```text
value = β₀ + βc cos(2πy/H) + βs sin(2πy/H)
```

It reports coefficients, amplitude, peak row, R², nested-model F/p, and BH q-values. Raw and smoothed harmonic tests are corrected as independent families.

The optional deterministic two-dimensional block bootstrap resamples occupied fixed blocks in shifted display coordinates and reports exploratory confidence intervals. It does not create biological replicates, replace a spatial generative model, or provide corrected hypothesis-test p-values. Version 0.2 does not join bootstrap blocks across the periodic seam.

## Provenance and resume

Unless disabled, each run maintains an atomic `run_manifest.json` with running/complete/failed state, canonical effective configuration, source inventory, package/dependency/platform/Git metadata, limitations, runtime, and processed/resumed/missing frames. Optional SHA-256 ledgers cover selected inputs and written outputs.

Per-frame checkpoints are committed only after every requested artifact for that frame succeeds. `--resume` requires the same checkpoint schema, effective scientific/output configuration, and input inventory. Every recorded artifact is validated by size and SHA-256 before reuse; missing or modified files are rejected rather than mixed into a run. Complete-series BH corrections, aggregate tables/plots, and videos are rebuilt from combined checkpoint and new-frame state.

With `retain_frame_data=True`, checkpointed frames are recomputed in memory to preserve the documented return contract. PIV plus resume is explicitly unsupported in version 0.2 because ROI trajectory state requires a separate checkpoint definition.

## Robust video creation

`create_video(image_paths, output_path, fps)` accepts grayscale, RGB, and RGBA frames. It alpha-composites RGBA over the plotting surface, scans the maximum dimensions, rounds the canvas to multiples of 16, and center-pads without stretching scientific content. Encoding targets a sibling temporary MP4; the destination is atomically replaced only after the writer closes successfully, so a failed encode preserves an existing video.

Order, smoothed-order, legacy Fourier, and Fourier-v2 sequences all use this path.

## Output layout

Requested outputs are written beneath `output_dir` (default: `data_dir/analysis_output`):

```text
analysis_output/
├── checkpoints/
│   ├── run.json
│   └── frames/frame_T####.json
├── data/
│   ├── order/*.npz
│   ├── fourier/*.npz
│   ├── frequency_bins/*.csv
│   ├── fourier_v2/*.npz
│   ├── frequency_pairs_v2/*.csv
│   ├── radial_profiles_v2/*.csv
│   ├── frame_metrics.csv
│   ├── frame_metrics_extended.csv
│   ├── segmentation_qc.csv
│   ├── spatial_statistics.csv
│   ├── spatial_statistics_harmonic.csv
│   ├── spatial_block_bootstrap.csv
│   ├── fourier_v2_metrics.csv
│   └── roi_*.csv
├── images/
│   ├── order/
│   ├── order_smoothed/
│   ├── histograms*/
│   ├── fourier/
│   ├── reciprocal/
│   ├── fourier_v2/
│   ├── fourier_v2_radial/
│   └── roi*/
├── provenance/
│   ├── input_checksums.sha256
│   └── output_checksums.sha256
├── videos/*.mp4
└── run_manifest.json
```

Raw H5/MAT/TIF data and generated `analysis_output*` directories are ignored by Git and must remain outside published source history.

## Testing and packaging

The test suite builds temporary H5/MAT fixtures and synthetic lattices, so original experiment files are not required:

```bash
python -m ruff check .
python -m pytest --cov=blender_fuse
python -m build
```

Tests cover periodic boundaries, legacy schemas, filtering/QC, physical-frequency Fourier v2, conjugate/Nyquist pairing, profile precedence, side-effect-free inspection, spatial summaries, checkpoint validation, PIV alignment, and atomic variable-size video encoding.

## Repository layout

```text
src/blender_fuse/          Installable package
examples/                  API examples and portable profiles
tests/                     Synthetic automated tests
legacy/code_2_original.py  Unsupported archival implementation
code_2.py                  Legacy compatibility launcher
```

The archive is retained because the initial repository had no Git history. It is not installed or imported and contains known defects.

## License

MIT — see [LICENSE](LICENSE).
