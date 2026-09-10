import numpy as np

from blender_fuse.order import (
    compute_order_parameter,
    compute_periodic_order_parameter,
    label_segmentation,
    shift_image_rows,
    shift_points_for_display,
    smooth_order_magnitudes,
    values_to_label_map,
)


def triangular_lattice(rows: int = 7, columns: int = 7) -> np.ndarray:
    points = []
    for row in range(rows):
        y = row * np.sqrt(3.0) / 2.0
        for column in range(columns):
            x = column + 0.5 * (row % 2)
            points.append((y, x))
    return np.asarray(points, dtype=float)


def test_label_segmentation_preserves_diagonal_connectivity() -> None:
    segmentation = np.asarray([[True, False], [False, True]], dtype=bool)

    labels, centroids = label_segmentation(segmentation)

    assert labels.max() == 1
    np.testing.assert_allclose(centroids, [[0.5, 0.5]])


def test_hexagonal_lattice_has_high_interior_psi6() -> None:
    points = triangular_lattice()
    psi = compute_order_parameter(points, 6)
    center = 3 * 7 + 3
    assert abs(psi[center]) > 0.95


def test_complex_psi_phase_tracks_rotation_from_positive_x_axis() -> None:
    points = triangular_lattice()
    angle = 0.1
    xy = points[:, [1, 0]]
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_rc = (xy @ rotation.T)[:, [1, 0]]

    center = 3 * 7 + 3
    base = compute_order_parameter(points, 6)[center]
    rotated = compute_order_parameter(rotated_rc, 6)[center]
    phase_ratio = (rotated / base) / abs(rotated / base)

    np.testing.assert_allclose(phase_ratio, np.exp(1j * 6 * angle), atol=1e-5)


def test_small_point_sets_return_stable_zeros() -> None:
    for count in range(4):
        points = np.column_stack((np.arange(count), np.arange(count))).astype(float)
        result = compute_order_parameter(points, 6)
        np.testing.assert_array_equal(result, np.zeros(count, dtype=complex))


def test_periodic_result_retains_only_original_regions() -> None:
    points = triangular_lattice(5, 5)
    result = compute_periodic_order_parameter(points, 6, image_height=10, periodic_y=True)
    assert result.shape == (len(points),)


def test_scalar_map_has_nan_background_not_rgb() -> None:
    labels = np.asarray([[0, 1, 1], [0, 2, 0]], dtype=int)
    result = values_to_label_map(labels, np.asarray([0.25, 0.75]))
    assert result.shape == labels.shape
    assert result.ndim == 2
    assert np.isnan(result[0, 0])
    assert result[0, 1] == 0.25
    assert result[1, 1] == 0.75


def test_smoothing_uses_neighbor_count_independent_of_n_fold() -> None:
    points = np.asarray([[0, 0], [0, 2], [0, 5]], dtype=float)
    psi = np.asarray([0.0, 0.5, 1.0], dtype=complex)
    result = smooth_order_magnitudes(points, psi, 1)
    np.testing.assert_allclose(result, [0.5, 0.0, 0.5])


def test_image_and_point_shift_use_same_direction() -> None:
    image = np.zeros((5, 4), dtype=float)
    image[1, 2] = 1.0
    shifted_image = shift_image_rows(image, 2)
    shifted_points = shift_points_for_display(np.asarray([[1.0, 2.0]]), 5, 2)
    assert shifted_image[3, 2] == 1.0
    np.testing.assert_allclose(shifted_points, [[3.0, 2.0]])
