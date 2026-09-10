import numpy as np
import pytest

from blender_fuse.fourier import analyze_fourier, analyze_fourier_v2


def sinusoid(height: int, width: int, x_frequency: int, offset: float = 0.0) -> np.ndarray:
    x = np.arange(width)
    row = np.cos(2 * np.pi * x_frequency * x / width) + offset
    return np.tile(row, (height, 1))


def test_legacy_metric_keeps_zeroed_dc_in_full_array_mean() -> None:
    image = sinusoid(16, 20, 3, offset=4.0)
    result = analyze_fourier(image, (0, 20, 0, 16), top_bins=2)
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    intensity = np.abs(spectrum) ** 2
    intensity[8, 10] = 0.0
    assert result.metric == pytest.approx(np.max(intensity) / np.mean(intensity))


def test_v2_is_invariant_to_constant_offset() -> None:
    first = analyze_fourier_v2(
        sinusoid(32, 40, 4),
        (0, 40, 0, 32),
        min_frequency_cycles_per_pixel=0.05,
        max_frequency_cycles_per_pixel=0.2,
        window="hann",
    )
    second = analyze_fourier_v2(
        sinusoid(32, 40, 4, offset=20.0),
        (0, 40, 0, 32),
        min_frequency_cycles_per_pixel=0.05,
        max_frequency_cycles_per_pixel=0.2,
        window="hann",
    )
    np.testing.assert_allclose(first.intensity, second.intensity, atol=1e-24)
    assert first.metric == pytest.approx(second.metric)


def test_v2_reports_rectangular_wavelength_in_pixels() -> None:
    result = analyze_fourier_v2(
        sinusoid(48, 80, 8),
        (0, 80, 0, 48),
        min_frequency_cycles_per_pixel=0.08,
        max_frequency_cycles_per_pixel=0.12,
        window="none",
        top_pairs=1,
    )
    assert len(result.peak_pairs) == 1
    assert result.peak_pairs[0].wavelength_pixels == pytest.approx(10.0)
    assert result.peak_pairs[0].member_count == 2


def test_v2_band_excludes_stronger_low_frequency_signal() -> None:
    width = 100
    x = np.arange(width)
    row = 10 * np.cos(2 * np.pi * 1 * x / width)
    row += np.cos(2 * np.pi * 10 * x / width)
    image = np.tile(row, (40, 1))
    result = analyze_fourier_v2(
        image,
        (0, width, 0, 40),
        min_frequency_cycles_per_pixel=0.08,
        max_frequency_cycles_per_pixel=0.12,
        window="none",
        top_pairs=1,
    )
    assert result.peak_pairs[0].wavelength_pixels == pytest.approx(10.0)
    assert result.low_frequency_power_fraction is not None
    assert result.low_frequency_power_fraction > 0.9


@pytest.mark.parametrize("shape", [(8, 8), (8, 9), (9, 8), (9, 9)])
def test_v2_conjugate_pairing_handles_odd_and_even_shapes(shape) -> None:
    height, width = shape
    image = sinusoid(height, width, 1)
    result = analyze_fourier_v2(
        image,
        (0, width, 0, height),
        min_frequency_cycles_per_pixel=0.05,
        max_frequency_cycles_per_pixel=0.4,
        window="none",
        top_pairs=1,
    )
    pair = result.peak_pairs[0]
    assert pair.second_index_rc is not None
    assert pair.member_count == 2
    assert pair.first_intensity == pytest.approx(pair.second_intensity)
    assert result.radial_profile.counts.sum() == height * width - 1


def test_v2_nyquist_bin_is_a_singleton() -> None:
    width = 8
    image = np.tile((-1.0) ** np.arange(width), (8, 1))
    result = analyze_fourier_v2(
        image,
        (0, width, 0, 8),
        min_frequency_cycles_per_pixel=0.49,
        max_frequency_cycles_per_pixel=0.51,
        window="none",
        top_pairs=1,
    )
    assert result.peak_pairs[0].member_count == 1
    assert result.peak_pairs[0].second_index_rc is None
    assert result.peak_pairs[0].wavelength_pixels == pytest.approx(2.0)


def test_v2_rejects_invalid_or_empty_settings() -> None:
    image = np.zeros((8, 8))
    with pytest.raises(ValueError, match="increasing"):
        analyze_fourier_v2(
            image,
            (0, 8, 0, 8),
            min_frequency_cycles_per_pixel=0.2,
            max_frequency_cycles_per_pixel=0.1,
        )
