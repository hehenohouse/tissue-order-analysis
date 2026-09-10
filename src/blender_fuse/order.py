"""Connected-component and weighted Voronoi order-parameter analysis."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import QhullError, Voronoi, cKDTree
from skimage import measure

LOGGER = logging.getLogger(__name__)


def label_segmentation(
    segmentation: NDArray[np.bool_],
) -> Tuple[NDArray[np.integer], NDArray[np.floating]]:
    """Label foreground components and return centroids as ``(row, column)``."""

    if segmentation.ndim != 2:
        raise ValueError(f"segmentation must be 2-D; got shape {segmentation.shape}.")
    # Match skimage's historical 2-D default used by the original analysis: 8-connectivity.
    labels = measure.label(segmentation, connectivity=segmentation.ndim)
    regions = measure.regionprops(labels)
    if regions:
        centroids = np.asarray([region.centroid for region in regions], dtype=float)
    else:
        centroids = np.empty((0, 2), dtype=float)
    return np.asarray(labels, dtype=np.int32), centroids


def compute_order_parameter(
    points_rc: NDArray[np.floating], n_fold: int
) -> NDArray[np.complexfloating]:
    """Compute weighted Voronoi ψₙ values for two-dimensional points.

    Each finite Voronoi ridge contributes ``ridge_length**2 * exp(i*n*theta)``
    to both incident points. Points without finite contributing ridges receive zero.
    """

    points = np.asarray(points_rc, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError(f"points_rc must have shape (N, 2); got {points.shape}.")
    if n_fold <= 0:
        raise ValueError("n_fold must be positive.")

    count = len(points)
    if count < 4:
        return np.zeros(count, dtype=np.complex128)
    if not np.all(np.isfinite(points)):
        raise ValueError("points_rc contains NaN or infinite coordinates.")
    if np.unique(points, axis=0).shape[0] != count:
        LOGGER.warning(
            "Duplicate centroids prevent a unique Voronoi tessellation; returning zeros."
        )
        return np.zeros(count, dtype=np.complex128)

    try:
        voronoi = Voronoi(points, qhull_options="QJ")
    except QhullError as exc:
        LOGGER.warning("Voronoi tessellation failed (%s); returning zero ψ values.", exc)
        return np.zeros(count, dtype=np.complex128)

    contributions = np.zeros(count, dtype=np.complex128)
    weights = np.zeros(count, dtype=float)

    for (first, second), vertices in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        if len(vertices) != 2 or -1 in vertices:
            continue
        edge = voronoi.vertices[vertices[1]] - voronoi.vertices[vertices[0]]
        weight = float(np.dot(edge, edge))
        if not np.isfinite(weight) or weight <= 0:
            continue

        delta_row, delta_col = points[second] - points[first]
        theta = np.arctan2(delta_row, delta_col)
        contribution = weight * np.exp(1j * n_fold * theta)
        contributions[first] += contribution
        weights[first] += weight

        reverse_contribution = weight * np.exp(1j * n_fold * (theta + np.pi))
        contributions[second] += reverse_contribution
        weights[second] += weight

    psi = np.zeros(count, dtype=np.complex128)
    np.divide(contributions, weights, out=psi, where=weights > 0)
    return psi


def compute_periodic_order_parameter(
    points_rc: NDArray[np.floating],
    n_fold: int,
    image_height: int,
    periodic_y: bool = True,
) -> NDArray[np.complexfloating]:
    """Compute ψₙ, augmenting centroid rows for vertical periodicity when requested."""

    points = np.asarray(points_rc, dtype=float)
    if not periodic_y or len(points) == 0:
        return compute_order_parameter(points, n_fold)
    if image_height <= 0:
        raise ValueError("image_height must be positive for periodic-y analysis.")

    upper = points + np.array([-float(image_height), 0.0])
    lower = points + np.array([float(image_height), 0.0])
    augmented = np.vstack((upper, points, lower))
    augmented_psi = compute_order_parameter(augmented, n_fold)
    count = len(points)
    return augmented_psi[count : 2 * count]


def values_to_label_map(
    labels: NDArray[np.integer], values: NDArray[np.generic]
) -> NDArray[np.floating]:
    """Expand one scalar per labeled component into a float image with NaN background."""

    label_image = np.asarray(labels)
    region_values = np.asarray(values)
    max_label = int(label_image.max(initial=0))
    if region_values.ndim != 1 or len(region_values) != max_label:
        raise ValueError(
            f"Expected one value for each of {max_label} labels; got shape {region_values.shape}."
        )

    output = np.full(label_image.shape, np.nan, dtype=float)
    if max_label == 0:
        return output
    lookup = np.empty(max_label + 1, dtype=float)
    lookup[0] = np.nan
    lookup[1:] = np.asarray(region_values, dtype=float)
    foreground = label_image > 0
    output[foreground] = lookup[label_image[foreground]]
    return output


def smooth_order_magnitudes(
    points_rc: NDArray[np.floating],
    psi: NDArray[np.complexfloating],
    n_neighbors: int,
    *,
    image_height: Optional[int] = None,
    periodic_y: bool = False,
    include_self: bool = False,
) -> NDArray[np.floating]:
    """Average each point's nearest-neighbor ψ magnitudes.

    ``n_neighbors`` counts other regions. If ``include_self`` is true, the focal
    region is included in addition to those neighbors.
    """

    points = np.asarray(points_rc, dtype=float)
    magnitudes = np.abs(np.asarray(psi, dtype=np.complex128))
    if points.shape != (len(magnitudes), 2):
        raise ValueError("points_rc and psi must describe the same number of regions.")
    if n_neighbors < 0:
        raise ValueError("n_neighbors cannot be negative.")
    if len(points) <= 1 or n_neighbors == 0:
        return np.asarray(magnitudes, dtype=float)

    count = len(points)
    if periodic_y:
        if image_height is None or image_height <= 0:
            raise ValueError("A positive image_height is required for periodic smoothing.")
        search_points = np.vstack(
            (
                points + np.array([-float(image_height), 0.0]),
                points,
                points + np.array([float(image_height), 0.0]),
            )
        )
        source_ids = np.tile(np.arange(count), 3)
    else:
        search_points = points
        source_ids = np.arange(count)

    tree = cKDTree(search_points)
    wanted = min(n_neighbors, count - 1)
    smoothed = np.empty(count, dtype=float)

    for point_index, point in enumerate(points):
        query_count = min(len(search_points), max(2, wanted + 1))
        selected: list[int] = []
        while True:
            _, indices = tree.query(point, k=query_count)
            for search_index in np.atleast_1d(indices):
                source_index = int(source_ids[int(search_index)])
                if source_index == point_index or source_index in selected:
                    continue
                selected.append(source_index)
                if len(selected) == wanted:
                    break
            if len(selected) == wanted or query_count == len(search_points):
                break
            query_count = min(len(search_points), query_count * 2)

        sample_indices = selected
        if include_self:
            sample_indices = [point_index, *sample_indices]
        if sample_indices:
            smoothed[point_index] = float(np.mean(magnitudes[sample_indices]))
        else:
            smoothed[point_index] = float(magnitudes[point_index])

    return smoothed


def analyze_order(
    segmentation: NDArray[np.bool_],
    n_fold: int,
    *,
    periodic_y: bool = True,
    smooth_neighbors: Optional[int] = None,
    smooth_include_self: bool = False,
) -> Tuple[
    NDArray[np.integer],
    NDArray[np.floating],
    NDArray[np.complexfloating],
    NDArray[np.floating],
    Optional[NDArray[np.floating]],
]:
    """Analyze one foreground mask and return labels, centroids, ψ, and maps."""

    labels, centroids = label_segmentation(segmentation)
    psi = compute_periodic_order_parameter(
        centroids,
        n_fold,
        image_height=segmentation.shape[0],
        periodic_y=periodic_y,
    )
    psi_map = values_to_label_map(labels, np.abs(psi))

    smoothed_map: Optional[NDArray[np.floating]] = None
    if smooth_neighbors is not None:
        smoothed = smooth_order_magnitudes(
            centroids,
            psi,
            smooth_neighbors,
            image_height=segmentation.shape[0],
            periodic_y=periodic_y,
            include_self=smooth_include_self,
        )
        smoothed_map = values_to_label_map(labels, smoothed)

    return labels, centroids, psi, psi_map, smoothed_map


def shift_image_rows(image: NDArray[np.generic], pixels: int) -> NDArray[np.generic]:
    """Shift an image down for positive values, wrapping around vertically."""

    if image.ndim < 2:
        raise ValueError("image must have at least two dimensions.")
    return np.roll(image, int(pixels), axis=0)


def shift_points_for_display(
    points_rc: NDArray[np.floating], image_height: int, pixels: int
) -> NDArray[np.floating]:
    """Shift ``(row, column)`` points exactly as :func:`shift_image_rows`."""

    shifted = np.asarray(points_rc, dtype=float).copy()
    if len(shifted):
        shifted[:, 0] = (shifted[:, 0] + pixels) % image_height
    return shifted
