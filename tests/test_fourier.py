import numpy as np
import pytest

from blender_fuse.fourier import (
    analyze_fourier,
    polygon_mask,
    reconstruct_selected_frequencies,
)


def test_known_sinusoid_has_expected_frequency_bins() -> None:
    size = 64
    frequency = 5
    x = np.arange(size)
    image = np.tile(np.cos(2 * np.pi * frequency * x / size), (size, 1))

    result = analyze_fourier(image, (0, size, 0, size), top_bins=2)

    offsets = {tuple(offset.astype(int)) for offset in result.offsets_rc}
    assert offsets == {(0, -frequency), (0, frequency)}
    assert result.metric > 1
    reconstruction = reconstruct_selected_frequencies(result)
    assert reconstruction.shape == image.shape
    assert np.max(reconstruction) == pytest.approx(1.0)


def test_fourier_rejects_out_of_bounds_rectangle() -> None:
    with pytest.raises(ValueError, match="exceeds image bounds"):
        analyze_fourier(np.zeros((10, 10)), (0, 11, 0, 10))


def test_dc_bin_is_never_reintroduced_when_non_dc_values_are_zero() -> None:
    image = np.ones((8, 8), dtype=float)
    result = analyze_fourier(image, (0, 8, 0, 8), top_bins=63)
    assert not np.any(np.all(result.bin_indices_rc == [4, 4], axis=1))
    assert np.max(reconstruct_selected_frequencies(result)) == 0.0


def test_public_polygon_xy_is_converted_to_row_column() -> None:
    polygon = ((20, 2), (26, 2), (26, 6), (20, 6))
    mask = polygon_mask((15, 30), polygon)

    assert mask[4, 23]
    assert not mask[10, 4]
