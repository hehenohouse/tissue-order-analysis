"""Demonstrate conservative ANOVA reporting on synthetic spatial data."""

import numpy as np

from blender_fuse.statistics import anova_conclusion, spatial_statistics

rng = np.random.default_rng(7)
rows = np.repeat([25.0, 75.0, 125.0], 20)
columns = rng.uniform(0, 200, len(rows))
centroids_rc = np.column_stack((rows, columns))

# The three bins deliberately have different means.
values = np.concatenate(
    (
        rng.normal(0.75, 0.03, 20),
        rng.normal(0.50, 0.03, 20),
        rng.normal(0.25, 0.03, 20),
    )
)
result = spatial_statistics(
    timepoint=1,
    centroids_rc=centroids_rc,
    values=values,
    image_height=150,
    bin_size=50,
)

print(f"ANOVA p-value: {result.anova_p_value:.6g}")
print(anova_conclusion(result))
