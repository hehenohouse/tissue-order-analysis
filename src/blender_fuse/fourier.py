"""Fourier and polygon-mask utilities."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from skimage.draw import polygon2mask

from .models import FourierPeakPair, FourierRadialProfile, FourierResult, FourierV2Result

PointXY = Tuple[float, float]


def validate_rectangle(rect: Tuple[int, int, int, int], image_shape: Sequence[int]) -> None:
    """Validate a public ``(x1, x2, y1, y2)`` rectangle against an image."""

    if len(image_shape) < 2:
        raise ValueError(f"image_shape must have at least two dimensions: {image_shape}")
    x1, x2, y1, y2 = rect
    height, width = int(image_shape[0]), int(image_shape[1])
    if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("Fourier rectangle must have non-negative origins and increasing bounds.")
    if x2 > width or y2 > height:
        raise ValueError(
            f"Fourier rectangle {rect} exceeds image bounds (width={width}, height={height})."
        )


def analyze_fourier(
    image: NDArray[np.generic],
    rect: Tuple[int, int, int, int],
    top_bins: int = 10,
) -> FourierResult:
    """Analyze a rectangular image region using the legacy max/mean metric.

    Selected bins are the highest-valued frequency pixels after the DC intensity
    is set to zero. They are not guaranteed to be distinct local maxima.
    """

    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError(f"Fourier input must be a 2-D scalar image; got {source.shape}.")
    if top_bins <= 0:
        raise ValueError("top_bins must be positive.")
    validate_rectangle(rect, source.shape)

    x1, x2, y1, y2 = rect
    region = np.asarray(source[y1:y2, x1:x2], dtype=float)
    if not np.all(np.isfinite(region)):
        raise ValueError("Fourier input region contains NaN or infinite values.")

    spectrum = np.fft.fftshift(np.fft.fft2(region))
    intensity = np.asarray(np.abs(spectrum) ** 2, dtype=float)
    center = (intensity.shape[0] // 2, intensity.shape[1] // 2)
    intensity[center] = 0.0

    mean_intensity = float(np.mean(intensity))
    metric = float(np.max(intensity) / mean_intensity) if mean_intensity > 0 else 0.0

    flattened = intensity.ravel()
    center_flat = int(np.ravel_multi_index(center, intensity.shape))
    candidates = np.flatnonzero(np.arange(intensity.size) != center_flat)
    count = min(top_bins, len(candidates))
    if count:
        candidate_values = flattened[candidates]
        selected_in_candidates = np.argpartition(candidate_values, -count)[-count:]
        selected = candidates[selected_in_candidates]
        selected = selected[np.argsort(flattened[selected])[::-1]]
        indices = np.column_stack(np.unravel_index(selected, intensity.shape)).astype(int)
        values = flattened[selected].astype(float)
        offsets = indices.astype(float) - np.asarray(center, dtype=float)
        distances = np.linalg.norm(offsets, axis=1)
    else:
        indices = np.empty((0, 2), dtype=int)
        values = np.empty(0, dtype=float)
        offsets = np.empty((0, 2), dtype=float)
        distances = np.empty(0, dtype=float)

    return FourierResult(
        rect=tuple(int(value) for value in rect),
        spectrum=np.asarray(spectrum, dtype=np.complex128),
        intensity=intensity,
        metric=metric,
        bin_indices_rc=indices,
        offsets_rc=offsets,
        bin_values=values,
        distances=distances,
    )


def _window_vector(length: int, name: str) -> NDArray[np.floating]:
    windows = {
        "none": lambda size: np.ones(size, dtype=float),
        "hann": np.hanning,
        "hamming": np.hamming,
        "blackman": np.blackman,
    }
    try:
        values = windows[name](length)
    except KeyError as exc:
        choices = ", ".join(sorted(windows))
        raise ValueError(f"Unsupported Fourier v2 window {name!r}; choose {choices}.") from exc
    return np.asarray(values, dtype=float)


def _conjugate_shifted_index(
    index_rc: Tuple[int, int], shape: Tuple[int, int]
) -> Tuple[int, int]:
    """Return the shifted-array index of a DFT bin's conjugate partner."""

    partner = []
    for shifted_index, length in zip(index_rc, shape):
        unshifted = (int(shifted_index) - length // 2) % length
        conjugate_unshifted = (-unshifted) % length
        partner.append((conjugate_unshifted + length // 2) % length)
    return int(partner[0]), int(partner[1])


def _radial_profile(
    intensity: NDArray[np.floating],
    radial_frequency: NDArray[np.floating],
    bin_width: float,
) -> FourierRadialProfile:
    maximum = float(np.max(radial_frequency, initial=0.0))
    edges = np.arange(0.0, maximum + bin_width, bin_width, dtype=float)
    if len(edges) < 2:
        edges = np.asarray([0.0, bin_width], dtype=float)
    if edges[-1] <= maximum:
        edges = np.append(edges, edges[-1] + bin_width)
    centers = (edges[:-1] + edges[1:]) / 2.0
    means = np.full(len(centers), np.nan, dtype=float)
    totals = np.zeros(len(centers), dtype=float)
    counts = np.zeros(len(centers), dtype=int)
    non_dc = radial_frequency > 0
    assignments = np.digitize(radial_frequency[non_dc], edges, right=False) - 1
    values = intensity[non_dc]
    for index in range(len(centers)):
        selected = assignments == index
        counts[index] = int(np.count_nonzero(selected))
        if counts[index]:
            totals[index] = float(np.sum(values[selected]))
            means[index] = float(np.mean(values[selected]))
    return FourierRadialProfile(
        edges_cycles_per_pixel=edges,
        centers_cycles_per_pixel=centers,
        mean_intensity=means,
        total_power=totals,
        counts=counts,
    )


def analyze_fourier_v2(
    image: NDArray[np.generic],
    rect: Tuple[int, int, int, int],
    *,
    min_frequency_cycles_per_pixel: float,
    max_frequency_cycles_per_pixel: float,
    window: str = "hann",
    top_pairs: int = 5,
    radial_bin_width_cycles_per_pixel: Optional[float] = None,
) -> FourierV2Result:
    """Analyze a rectangle with windowing, a physical-frequency band, and pairs."""

    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError(f"Fourier input must be a 2-D scalar image; got {source.shape}.")
    if not np.all(np.isfinite(source)):
        raise ValueError("Fourier input contains NaN or infinite values.")
    validate_rectangle(rect, source.shape)
    if top_pairs <= 0:
        raise ValueError("top_pairs must be positive.")
    bounds = (float(min_frequency_cycles_per_pixel), float(max_frequency_cycles_per_pixel))
    if not np.all(np.isfinite(bounds)) or bounds[0] < 0 or bounds[1] <= bounds[0]:
        raise ValueError(
            "Fourier v2 frequency bounds must be finite, non-negative, and increasing."
        )

    x1, x2, y1, y2 = rect
    region = np.asarray(source[y1:y2, x1:x2], dtype=float)
    height, width = region.shape
    if window != "none" and min(height, width) < 2:
        raise ValueError("Windowed Fourier v2 rectangles must be at least 2 by 2 pixels.")
    row_window = _window_vector(height, window)
    column_window = _window_vector(width, window)
    window_2d = row_window[:, None] * column_window[None, :]
    normalization = float(np.sum(window_2d**2))
    if normalization <= 0:
        raise ValueError("Fourier v2 window has zero power for this rectangle.")

    removed_mean = float(np.mean(region))
    processed = (region - removed_mean) * window_2d
    spectrum = np.fft.fftshift(np.fft.fft2(processed))
    intensity = np.asarray(np.abs(spectrum) ** 2 / normalization, dtype=float)
    frequency_y = np.asarray(np.fft.fftshift(np.fft.fftfreq(height)), dtype=float)
    frequency_x = np.asarray(np.fft.fftshift(np.fft.fftfreq(width)), dtype=float)
    radial = np.hypot(frequency_y[:, None], frequency_x[None, :])
    eligible = (
        (radial >= bounds[0])
        & (radial <= bounds[1])
        & (radial > 0)
    )
    eligible_values = intensity[eligible]
    if eligible_values.size:
        eligible_max = float(np.max(eligible_values))
        eligible_mean = float(np.mean(eligible_values))
        metric = eligible_max / eligible_mean if eligible_mean > 0 else 0.0
    else:
        eligible_max = 0.0
        eligible_mean = 0.0
        metric = 0.0

    shape = (height, width)
    pair_records = []
    visited = set()
    for row, column in np.column_stack(np.nonzero(eligible)):
        first = (int(row), int(column))
        partner = _conjugate_shifted_index(first, shape)
        canonical = tuple(sorted((first, partner)))
        if canonical in visited:
            continue
        visited.add(canonical)
        members = (first,) if first == partner else canonical
        member_values = [float(intensity[index]) for index in members]
        total = float(np.sum(member_values))
        if not np.isfinite(total) or total <= 0:
            continue
        pair_records.append((total, canonical[0], canonical, members, member_values))
    pair_records.sort(key=lambda record: (-record[0], record[1][0], record[1][1]))

    peak_pairs = []
    for rank, (_, _, canonical, members, member_values) in enumerate(
        pair_records[:top_pairs], start=1
    ):
        first = canonical[0]
        second = canonical[1] if canonical[1] != canonical[0] else None
        fy = float(frequency_y[first[0]])
        fx = float(frequency_x[first[1]])
        radial_value = float(np.hypot(fy, fx))
        peak_pairs.append(
            FourierPeakPair(
                rank=rank,
                first_index_rc=first,
                second_index_rc=second,
                frequency_y_cycles_per_pixel=fy,
                frequency_x_cycles_per_pixel=fx,
                radial_frequency_cycles_per_pixel=radial_value,
                wavelength_pixels=float(1.0 / radial_value),
                first_intensity=member_values[0],
                second_intensity=(member_values[1] if len(member_values) == 2 else None),
                mean_intensity=float(np.mean(member_values)),
                total_power=float(np.sum(member_values)),
                member_count=len(members),
            )
        )

    bin_width = (
        float(radial_bin_width_cycles_per_pixel)
        if radial_bin_width_cycles_per_pixel is not None
        else min(1.0 / height, 1.0 / width)
    )
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("radial_bin_width_cycles_per_pixel must be positive and finite.")
    radial_profile = _radial_profile(intensity, radial, bin_width)
    non_dc_power = float(np.sum(intensity[radial > 0]))
    low_power = float(np.sum(intensity[(radial > 0) & (radial < bounds[0])]))
    band_power = float(np.sum(eligible_values))
    low_fraction = low_power / non_dc_power if non_dc_power > 0 else None
    band_fraction = band_power / non_dc_power if non_dc_power > 0 else None

    return FourierV2Result(
        rect=tuple(int(value) for value in rect),
        schema_version="blender-fuse.fourier-v2.v1",
        window=window,
        removed_mean=removed_mean,
        window_power_normalization=normalization,
        spectrum=np.asarray(spectrum, dtype=np.complex128),
        intensity=intensity,
        frequency_y_cycles_per_pixel=frequency_y,
        frequency_x_cycles_per_pixel=frequency_x,
        radial_frequency_cycles_per_pixel=np.asarray(radial, dtype=float),
        eligible_mask=np.asarray(eligible, dtype=bool),
        eligible_max_intensity=eligible_max,
        eligible_mean_intensity=eligible_mean,
        metric=float(metric),
        peak_pairs=tuple(peak_pairs),
        radial_profile=radial_profile,
        low_frequency_power_fraction=low_fraction,
        eligible_power_fraction=band_fraction,
    )


def reconstruct_selected_frequencies(result: FourierResult) -> NDArray[np.floating]:
    """Inverse-transform selected bins using their original complex coefficients."""

    selected = np.zeros_like(result.spectrum, dtype=np.complex128)
    if result.bin_indices_rc.size:
        rows = result.bin_indices_rc[:, 0]
        columns = result.bin_indices_rc[:, 1]
        selected[rows, columns] = result.spectrum[rows, columns]
    reconstructed = np.fft.ifft2(np.fft.ifftshift(selected))
    magnitude = np.abs(reconstructed)
    maximum = float(np.max(magnitude, initial=0.0))
    if maximum > 0:
        magnitude = magnitude / maximum
    return np.asarray(magnitude, dtype=float)


def polygon_mask(image_shape: Sequence[int], polygon_xy: Iterable[PointXY]) -> NDArray[np.bool_]:
    """Create a mask from public ``(x, y)`` vertices.

    ``skimage.draw.polygon2mask`` expects ``(row, column)``, so conversion is
    deliberately centralized here.
    """

    shape = tuple(int(value) for value in image_shape[:2])
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"Invalid image shape: {image_shape}")
    points = tuple((float(x), float(y)) for x, y in polygon_xy)
    if len(points) < 3:
        raise ValueError("A polygon requires at least three vertices.")
    vertices_rc = np.asarray([(y, x) for x, y in points], dtype=float)
    return np.asarray(polygon2mask(shape, vertices_rc), dtype=bool)


def polygon_bounds(mask: NDArray[np.bool_]) -> Optional[Tuple[int, int, int, int]]:
    """Return a non-empty mask's ``(row1, row2, col1, col2)`` bounds."""

    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    return (
        int(rows.min()),
        int(rows.max()) + 1,
        int(columns.min()),
        int(columns.max()) + 1,
    )


def masked_polygon_crop(
    image: NDArray[np.generic], polygon_xy: Iterable[PointXY]
) -> Tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Mask a 2-D image and crop it to the polygon's clipped bounding box."""

    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError(f"Polygon crop requires a 2-D image; got {source.shape}.")
    mask = polygon_mask(source.shape, polygon_xy)
    bounds = polygon_bounds(mask)
    if bounds is None:
        raise ValueError("Polygon does not overlap the image.")
    row1, row2, col1, col2 = bounds
    cropped_mask = mask[row1:row2, col1:col2]
    cropped = np.where(cropped_mask, source[row1:row2, col1:col2], 0.0)
    return np.asarray(cropped, dtype=float), cropped_mask


def analyze_polygon_fourier(
    image: NDArray[np.generic],
    polygon_xy: Iterable[PointXY],
    top_bins: int = 10,
) -> FourierResult:
    """Mask and tightly crop a polygon before Fourier analysis."""

    cropped, _ = masked_polygon_crop(image, polygon_xy)
    rect = (0, cropped.shape[1], 0, cropped.shape[0])
    return analyze_fourier(cropped, rect, top_bins=top_bins)


def analyze_centroid_polygon_fourier(
    centroids_rc: NDArray[np.floating],
    polygon_xy: Iterable[PointXY],
    image_shape: Sequence[int],
    top_bins: int = 10,
) -> FourierResult:
    """Fourier-analyze impulses at centroids lying inside a public ROI polygon."""

    shape = (int(image_shape[0]), int(image_shape[1]))
    mask = polygon_mask(shape, polygon_xy)
    impulses = np.zeros(shape, dtype=float)
    for row, column in np.asarray(centroids_rc, dtype=float):
        row_index, column_index = int(round(row)), int(round(column))
        if (
            0 <= row_index < shape[0]
            and 0 <= column_index < shape[1]
            and mask[row_index, column_index]
        ):
            impulses[row_index, column_index] = 1.0
    return analyze_polygon_fourier(impulses, polygon_xy, top_bins=top_bins)


def pad_center(
    image: NDArray[np.generic], target_height: int, target_width: int
) -> NDArray[np.generic]:
    """Center-pad a 2-D array with zeros without cropping it."""

    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError("pad_center requires a 2-D array.")
    height, width = source.shape
    if target_height < height or target_width < width:
        raise ValueError(
            f"Target {target_height}x{target_width} is smaller than source {height}x{width}."
        )
    pad_height, pad_width = target_height - height, target_width - width
    return np.pad(
        source,
        (
            (pad_height // 2, pad_height - pad_height // 2),
            (pad_width // 2, pad_width - pad_width // 2),
        ),
        mode="constant",
    )


def center_crop(image: NDArray[np.generic], size: int) -> NDArray[np.generic]:
    """Return an at-most ``size`` square crop centered on a 2-D array."""

    source = np.asarray(image)
    if source.ndim != 2 or size <= 0:
        raise ValueError("center_crop requires a 2-D array and a positive size.")
    height, width = source.shape
    crop_height, crop_width = min(size, height), min(size, width)
    row1 = (height - crop_height) // 2
    col1 = (width - crop_width) // 2
    return source[row1 : row1 + crop_height, col1 : col1 + crop_width]
