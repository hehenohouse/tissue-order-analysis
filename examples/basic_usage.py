"""Minimal programmatic Blender Fuse example."""

from blender_fuse import AnalysisConfig, OutputOptions, run_analysis

config = AnalysisConfig(
    data_dir="raw_data/23C",
    output_dir="analysis_results",
    start_t=1,
    end_t=150,
    n_fold=6,
    smooth_neighbors=6,
    outputs=OutputOptions(save_videos=False),
)

result = run_analysis(config)
for frame in result.frames:
    print(
        frame.timepoint,
        frame.region_count,
        frame.mean_psi_magnitude,
        frame.fourier_metric,
    )
