"""Topological defects and spatial correlation of bond-orientational order."""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, QhullError, Voronoi, cKDTree

from .models import StructureResult

LOGGER = logging.getLogger(__name__)


def spacing_qc(
    points_rc: NDArray[np.floating],
    image_shape: Tuple[int, int],
    *,
    hole_radius_factor: float = 1.0,
) -> dict:
    """Detect missing detections from the geometry of retained centroids.

    When nuclei are missed, the survivors keep their true local spacing while
    the global density falls. So the nearest-neighbor distance ``d`` stays
    put while the mean spacing ``a = sqrt(H·W/N)`` grows, and ``d / a`` drops
    (≈1.07 for a perfect triangular lattice). A missing point leaves a
    Delaunay triangle whose circumradius is about ``d`` instead of ``d/√3``,
    so ``hole_area_fraction`` is the share of triangle area whose circumradius
    exceeds ``hole_radius_factor · median(d)``. Positional disorder alone also
    produces some such triangles, so read both metrics relative to frames
    known to be well segmented rather than as absolute counts.

    Statistics use an interior window three median spacings from every edge.
    """

    empty = {
        "median_nn_distance_pixels": None,
        "nn_distance_cv": None,
        "nn_distance_ratio": None,
        "hole_area_fraction": None,
    }
    points = np.asarray(points_rc, dtype=float)
    height, width = int(image_shape[0]), int(image_shape[1])
    if len(points) < 10:
        return empty
    distances = cKDTree(points).query(points, k=2)[0][:, 1]
    median = float(np.median(distances))
    margin = 3.0 * median
    inner = (
        (points[:, 0] > margin)
        & (points[:, 0] < height - margin)
        & (points[:, 1] > margin)
        & (points[:, 1] < width - margin)
    )
    if np.count_nonzero(inner) < 10:
        return empty
    inner_distances = distances[inner]
    spacing = math.sqrt(height * width / len(points))
    result = {
        "median_nn_distance_pixels": float(np.median(inner_distances)),
        "nn_distance_cv": float(np.std(inner_distances) / np.mean(inner_distances)),
        "nn_distance_ratio": float(np.median(inner_distances) / spacing),
        "hole_area_fraction": None,
    }
    try:
        triangles = Delaunay(points).simplices
    except QhullError:
        return result
    first, second, third = (points[triangles[:, k]] for k in range(3))
    side_a = np.linalg.norm(first - second, axis=1)
    side_b = np.linalg.norm(second - third, axis=1)
    side_c = np.linalg.norm(third - first, axis=1)
    area = 0.5 * np.abs(
        (second[:, 0] - first[:, 0]) * (third[:, 1] - first[:, 1])
        - (second[:, 1] - first[:, 1]) * (third[:, 0] - first[:, 0])
    )
    radius = side_a * side_b * side_c / np.maximum(4.0 * area, 1e-12)
    center = (first + second + third) / 3.0
    keep = (
        (center[:, 0] > margin)
        & (center[:, 0] < height - margin)
        & (center[:, 1] > margin)
        & (center[:, 1] < width - margin)
    )
    total = float(np.sum(area[keep]))
    if total > 0:
        holes = keep & (radius > hole_radius_factor * median)
        result["hole_area_fraction"] = float(np.sum(area[holes]) / total)
    return result


def voronoi_neighbors(
    points_rc: NDArray[np.floating],
    image_shape: Tuple[int, int],
    periodic_y: bool = True,
) -> Tuple[List[NDArray[np.integer]], NDArray[np.bool_]]:
    """Return each region's Voronoi neighbors and a boundary flag.

    Neighbors are indices into ``points_rc``. With vertical periodicity, a
    neighbor reached across the seam maps back to its original region. The
    boundary flag uses the same rule as :func:`blender_fuse.order.voronoi_boundary_mask`:
    an unbounded cell or a vertex outside the image (rows only when not periodic).
    """

    points = np.asarray(points_rc, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError(f"points_rc must have shape (N, 2); got {points.shape}.")
    height, width = int(image_shape[0]), int(image_shape[1])
    count = len(points)
    empty = [np.empty(0, dtype=np.int64) for _ in range(count)]
    if count < 4:
        return empty, np.ones(count, dtype=bool)

    if periodic_y:
        search = np.vstack(
            (points - [float(height), 0.0], points, points + [float(height), 0.0])
        )
        offset = count
    else:
        search = points
        offset = 0
    try:
        voronoi = Voronoi(search, qhull_options="QJ")
    except QhullError as exc:
        LOGGER.warning("Voronoi tessellation failed (%s); no neighbors computed.", exc)
        return empty, np.ones(count, dtype=bool)

    vertices = voronoi.vertices
    outside = (vertices[:, 1] < -0.5) | (vertices[:, 1] > width - 0.5)
    if not periodic_y:
        outside |= (vertices[:, 0] < -0.5) | (vertices[:, 0] > height - 0.5)
    boundary = np.zeros(count, dtype=bool)
    for index in range(count):
        region = voronoi.regions[voronoi.point_region[offset + index]]
        if not region or -1 in region or np.any(outside[region]):
            boundary[index] = True

    ridges = np.asarray(voronoi.ridge_points, dtype=np.int64)
    first, second = ridges[:, 0], ridges[:, 1]
    keep_first = (first >= offset) & (first < offset + count)
    keep_second = (second >= offset) & (second < offset + count)
    sources = np.concatenate((first[keep_first], second[keep_second])) - offset
    targets = np.concatenate((second[keep_first], first[keep_second])) % count
    order = np.lexsort((targets, sources))
    sources, targets = sources[order], targets[order]
    pairs = np.unique(np.column_stack((sources, targets)), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    splits = np.searchsorted(pairs[:, 0], np.arange(count + 1))
    neighbors = [pairs[splits[i] : splits[i + 1], 1] for i in range(count)]
    return neighbors, boundary


def defect_summary(
    neighbors: List[NDArray[np.integer]],
    boundary: NDArray[np.bool_],
) -> dict:
    """Summarize coordination numbers and defect clustering for interior regions.

    Topological charge is ``q = 6 - z`` for coordination number ``z``. A defect
    is any interior region with ``z != 6``. ``dislocation_fraction`` is the
    share of defects adjacent to an interior defect of opposite charge sign
    (5–7 pairs are the elementary dislocation in a triangular lattice).

    Interior means neither the region nor any of its neighbors is a boundary
    region: a region next to the free edge can have a closed Voronoi cell yet
    still miss the neighbor that lies beyond the edge, which reads as a
    spurious z = 5.
    """

    coordination = np.asarray([len(item) for item in neighbors], dtype=int)
    flagged = np.asarray(boundary, dtype=bool)
    interior = ~flagged & np.asarray(
        [not np.any(flagged[item]) for item in neighbors], dtype=bool
    )
    z = coordination[interior]
    total = int(z.size)
    result = {
        "interior_regions": total,
        "mean_coordination": None,
        "fraction_z4_or_less": None,
        "fraction_z5": None,
        "fraction_z6": None,
        "fraction_z7": None,
        "fraction_z8_or_more": None,
        "defect_fraction": None,
        "mean_abs_charge": None,
        "net_charge_per_region": None,
        "defect_neighbor_fraction": None,
        "dislocation_fraction": None,
    }
    if total == 0:
        return result
    charge = 6 - coordination
    is_defect = interior & (coordination != 6)
    defect_ids = np.flatnonzero(is_defect)
    touching_any = 0
    touching_opposite = 0
    for index in defect_ids:
        others = neighbors[index]
        others = others[is_defect[others]]
        if others.size:
            touching_any += 1
            if np.any(np.sign(charge[others]) == -np.sign(charge[index])):
                touching_opposite += 1
    defects = int(defect_ids.size)
    result.update(
        mean_coordination=float(np.mean(z)),
        fraction_z4_or_less=float(np.mean(z <= 4)),
        fraction_z5=float(np.mean(z == 5)),
        fraction_z6=float(np.mean(z == 6)),
        fraction_z7=float(np.mean(z == 7)),
        fraction_z8_or_more=float(np.mean(z >= 8)),
        defect_fraction=defects / total,
        mean_abs_charge=float(np.mean(np.abs(6 - z))),
        net_charge_per_region=float(np.mean(6 - z)),
        defect_neighbor_fraction=(touching_any / defects) if defects else None,
        dislocation_fraction=(touching_opposite / defects) if defects else None,
    )
    return result


def orientational_correlation(
    points_rc: NDArray[np.floating],
    psi: NDArray[np.complexfloating],
    image_shape: Tuple[int, int],
    *,
    periodic_y: bool = True,
    max_distance: float,
    bin_width: float,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.integer]]:
    """Compute g₆(r) = ⟨Re ψ(rᵢ) ψ*(rⱼ)⟩ over region pairs at separation r.

    Only regions with finite ψ contribute. Separations wrap vertically when
    ``periodic_y`` is true; ``max_distance`` must then not exceed half the
    image height. Returns bin centers, the mean product per bin (NaN for empty
    bins), and pair counts.
    """

    points = np.asarray(points_rc, dtype=float)
    values = np.asarray(psi, dtype=np.complex128)
    if points.shape != (len(values), 2):
        raise ValueError("points_rc and psi must describe the same regions.")
    if not (math.isfinite(max_distance) and max_distance > 0):
        raise ValueError("max_distance must be positive and finite.")
    if not (math.isfinite(bin_width) and 0 < bin_width <= max_distance):
        raise ValueError("bin_width must be positive and no larger than max_distance.")
    height = float(image_shape[0])
    if periodic_y and max_distance > height / 2.0:
        raise ValueError("max_distance cannot exceed half the image height when periodic.")

    edges = np.arange(0.0, max_distance + bin_width * 0.5, bin_width)
    if edges[-1] < max_distance:
        edges = np.append(edges, max_distance)
    centers = (edges[:-1] + edges[1:]) / 2.0
    sums = np.zeros(len(centers), dtype=float)
    counts = np.zeros(len(centers), dtype=np.int64)

    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    points, values = points[finite], values[finite]
    if len(points) >= 2:
        if periodic_y:
            rows = np.mod(points[:, 0], height)
            rows[rows >= height] = 0.0
            columns = points[:, 1] - float(np.min(points[:, 1]))
            # A wide horizontal box keeps columns effectively non-periodic.
            box = (height, float(np.max(columns)) + 2.0 * max_distance + 1.0)
            data = np.column_stack((rows, columns))
            tree = cKDTree(data, boxsize=box)
        else:
            data = points
            tree = cKDTree(data)
        pairs = tree.query_pairs(max_distance, output_type="ndarray")
        if len(pairs):
            delta = np.abs(data[pairs[:, 0]] - data[pairs[:, 1]])
            if periodic_y:
                delta[:, 0] = np.minimum(delta[:, 0], height - delta[:, 0])
            distance = np.hypot(delta[:, 0], delta[:, 1])
            product = np.real(values[pairs[:, 0]] * np.conj(values[pairs[:, 1]]))
            index = np.clip(np.digitize(distance, edges) - 1, 0, len(centers) - 1)
            sums = np.bincount(index, weights=product, minlength=len(centers))
            counts = np.bincount(index, minlength=len(centers)).astype(np.int64)

    mean = np.full(len(centers), np.nan, dtype=float)
    np.divide(sums, counts, out=mean, where=counts > 0)
    return centers, mean, counts


def _log_fit(x: NDArray[np.floating], y: NDArray[np.floating]) -> Tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - float(np.sum((y - fitted) ** 2)) / total if total > 0 else float("nan")
    return float(slope), r_squared


def correlation_lengths(
    centers: NDArray[np.floating],
    normalized: NDArray[np.floating],
    spacing: float,
    *,
    fit_floor: float = 0.05,
) -> dict:
    """Estimate decay scales of a normalized g₆(r).

    Bins closer than ~one mean spacing hold few, often spurious pairs (split or
    tiny objects), so every estimate starts at the first bin with r ≥ 0.75·a.

    ``xi_1e`` is where g₆/⟨|ψ|²⟩ first falls below 1/e (linear interpolation),
    a model-free scale. If the first usable bin is already below 1/e, its
    center is returned with ``xi_1e_is_upper_bound`` set. Exponential
    (liquid-like) and power-law (hexatic-like)
    fits use r ≥ a up to the first bin below ``fit_floor``, beyond which the
    curve is dominated by noise; R² in log space indicates which form
    describes the decay better over that range.
    """

    out = {
        "xi_1e_pixels": None,
        "xi_1e_is_upper_bound": False,
        "exp_xi_pixels": None,
        "exp_r_squared": None,
        "power_eta": None,
        "power_r_squared": None,
    }
    valid = np.isfinite(normalized) & (centers >= 0.75 * spacing)
    r, g = centers[valid], normalized[valid]
    if r.size == 0:
        return out
    threshold = 1.0 / math.e
    below = np.flatnonzero(g < threshold)
    if below.size and below[0] > 0:
        k = below[0]
        r0, r1, g0, g1 = r[k - 1], r[k], g[k - 1], g[k]
        out["xi_1e_pixels"] = float(r0 + (threshold - g0) * (r1 - r0) / (g1 - g0))
    elif below.size:
        # Already below 1/e in the first usable bin: only an upper bound.
        out["xi_1e_pixels"] = float(r[0])
        out["xi_1e_is_upper_bound"] = True

    in_range = r >= spacing
    stop = np.flatnonzero(in_range & (g < fit_floor))
    if stop.size:
        in_range &= np.arange(r.size) < stop[0]
    fit = in_range & (g > 0)
    if np.count_nonzero(fit) >= 3:
        slope, r2 = _log_fit(r[fit], np.log(g[fit]))
        if slope < 0:
            out["exp_xi_pixels"] = float(-1.0 / slope)
        out["exp_r_squared"] = r2
        slope, r2 = _log_fit(np.log(r[fit]), np.log(g[fit]))
        out["power_eta"] = float(-slope)
        out["power_r_squared"] = r2
    return out


def analyze_structure(
    timepoint: int,
    points_rc: NDArray[np.floating],
    psi: NDArray[np.complexfloating],
    image_shape: Tuple[int, int],
    *,
    periodic_y: bool = True,
    max_distance_spacings: float = 15.0,
    bin_width_spacings: float = 0.25,
) -> StructureResult:
    """Compute defect statistics and g₆(r) for one frame.

    Distances are binned in units of the mean spacing ``a = sqrt(H·W / N)``,
    which makes curves comparable across frames with different densities.
    The maximum distance is capped at half the image width (and half the
    height when periodic).
    """

    points = np.asarray(points_rc, dtype=float)
    height, width = int(image_shape[0]), int(image_shape[1])
    count = len(points)
    spacing = math.sqrt(height * width / count) if count else float("nan")

    neighbors, boundary = voronoi_neighbors(points, (height, width), periodic_y=periodic_y)
    defects = defect_summary(neighbors, boundary)

    centers = np.empty(0, dtype=float)
    mean = np.empty(0, dtype=float)
    counts = np.empty(0, dtype=np.int64)
    normalized = np.empty(0, dtype=float)
    lengths = correlation_lengths(centers, normalized, spacing)
    values = np.asarray(psi, dtype=np.complex128)
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    psi_squared = float(np.mean(np.abs(values[finite]) ** 2)) if np.any(finite) else None
    if count >= 2 and math.isfinite(spacing) and psi_squared:
        limit = width / 2.0
        if periodic_y:
            limit = min(limit, height / 2.0)
        max_distance = min(max_distance_spacings * spacing, limit)
        bin_width = min(bin_width_spacings * spacing, max_distance)
        centers, mean, counts = orientational_correlation(
            points,
            values,
            (height, width),
            periodic_y=periodic_y,
            max_distance=max_distance,
            bin_width=bin_width,
        )
        normalized = mean / psi_squared
        lengths = correlation_lengths(centers, normalized, spacing)

    return StructureResult(
        timepoint=int(timepoint),
        region_count=count,
        mean_spacing_pixels=float(spacing) if math.isfinite(spacing) else None,
        mean_psi_squared=psi_squared,
        **defects,
        **lengths,
        g6_r_pixels=tuple(float(value) for value in centers),
        g6=tuple(None if not np.isfinite(value) else float(value) for value in mean),
        g6_normalized=tuple(
            None if not np.isfinite(value) else float(value) for value in normalized
        ),
        g6_pair_counts=tuple(int(value) for value in counts),
    )
