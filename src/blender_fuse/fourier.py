"""Fourier and polygon-mask utilities."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from skimage.draw import polygon2mask

from .models import FourierResult

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
