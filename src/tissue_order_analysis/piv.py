"""PIVlab loading, interpolation, and ROI evolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.io import loadmat
from scipy.spatial import cKDTree

LOGGER = logging.getLogger(__name__)
PointXY = Tuple[float, float]


@dataclass
class PIVData:
    """Validated PIV velocity frames sharing one physical coordinate grid."""

    u_frames: List[NDArray[np.floating]]
    v_frames: List[NDArray[np.floating]]
    x_coords: NDArray[np.floating]
    y_coords: NDArray[np.floating]

    def __post_init__(self) -> None:
        if len(self.u_frames) != len(self.v_frames):
            raise ValueError("PIV u and v series contain different numbers of frames.")
        if not self.u_frames:
            raise ValueError("PIV series contains no velocity frames.")
        grid_shape = self.x_coords.shape
        if self.y_coords.shape != grid_shape:
            raise ValueError("PIV x and y coordinate grids have different shapes.")
        for index, (u_frame, v_frame) in enumerate(zip(self.u_frames, self.v_frames)):
            if u_frame.shape != grid_shape or v_frame.shape != grid_shape:
                raise ValueError(
                    f"PIV frame {index} has velocity shape {u_frame.shape}/{v_frame.shape}, "
                    f"but coordinate grid shape is {grid_shape}."
                )

    @property
    def frame_count(self) -> int:
        return len(self.u_frames)


def _mat_frames(value: NDArray[np.generic], key: str) -> List[NDArray[np.floating]]:
    array = np.asarray(value)
    if array.dtype == object:
        frames = [np.squeeze(np.asarray(cell, dtype=float)) for cell in array.ravel()]
    elif array.ndim == 2:
        frames = [np.asarray(array, dtype=float)]
    elif array.ndim == 3:
        frames = [np.asarray(array[index], dtype=float) for index in range(array.shape[0])]
    else:
        raise ValueError(
            f"Unsupported PIVlab array layout for {key!r}: shape={array.shape}, "
            f"dtype={array.dtype}."
        )
    for index, frame in enumerate(frames):
        if frame.ndim != 2:
            raise ValueError(f"{key} frame {index} is not 2-D after squeezing: {frame.shape}")
    return frames


def _coordinate_grid(value: NDArray[np.generic], key: str) -> NDArray[np.floating]:
    frames = _mat_frames(value, key)
    first = frames[0]
    for index, frame in enumerate(frames[1:], start=1):
        if frame.shape != first.shape or not np.allclose(frame, first, equal_nan=True):
            raise ValueError(
                f"PIV coordinate grid {key!r} changes at frame {index}; "
                "per-frame grids are not currently supported."
            )
    return np.asarray(first, dtype=float)


def load_pivlab_data(path: Path) -> PIVData:
    """Load a PIVlab MAT file with explicit physical coordinate grids."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PIVlab MAT file does not exist: {path}")
    raw = loadmat(path)
    required = ("u_original", "v_original", "x", "y")
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(
            f"PIVlab MAT file is missing required keys {missing}; "
            f"available keys: {sorted(key for key in raw if not key.startswith('__'))}"
        )

    return PIVData(
        u_frames=_mat_frames(raw["u_original"], "u_original"),
        v_frames=_mat_frames(raw["v_original"], "v_original"),
        x_coords=_coordinate_grid(raw["x"], "x"),
        y_coords=_coordinate_grid(raw["y"], "y"),
    )


class VelocityInterpolator:
    """Reusable interpolator for one PIV velocity frame."""

    def __init__(
        self,
        u: NDArray[np.floating],
        v: NDArray[np.floating],
        x_coords: NDArray[np.floating],
        y_coords: NDArray[np.floating],
        method: str = "bilinear",
    ) -> None:
        methods = {"bilinear", "bicubic", "inverse_distance", "nearest"}
        if method not in methods:
            raise ValueError(f"Unsupported interpolation method: {method!r}")

        u_array = np.asarray(u, dtype=float)
        v_array = np.asarray(v, dtype=float)
        x_array = np.asarray(x_coords, dtype=float)
        y_array = np.asarray(y_coords, dtype=float)
        if not (
            u_array.shape == v_array.shape == x_array.shape == y_array.shape and u_array.ndim == 2
        ):
            raise ValueError("PIV velocities and coordinate grids must share one 2-D shape.")

        valid = (
            np.isfinite(u_array)
            & np.isfinite(v_array)
            & np.isfinite(x_array)
            & np.isfinite(y_array)
        )
        if not np.any(valid):
            raise ValueError("PIV velocity frame contains no finite samples.")

        self.method = method
        self.points = np.column_stack((x_array[valid], y_array[valid]))
        self.velocities = np.column_stack((u_array[valid], v_array[valid]))
        self._tree: Optional[cKDTree] = None
        self._interpolator: object = None

        if method == "bilinear":
            self._interpolator = LinearNDInterpolator(
                self.points, self.velocities, fill_value=np.nan
            )
        elif method == "bicubic":
            self._interpolator = CloughTocher2DInterpolator(
                self.points, self.velocities, fill_value=np.nan
            )
        elif method == "nearest":
            self._tree = cKDTree(self.points)

    def velocity_at(self, x: float, y: float) -> PointXY:
        """Return interpolated ``(u, v)``; outside-grid interpolation is zero."""

        query = np.asarray([float(x), float(y)], dtype=float)
        if self.method in {"bilinear", "bicubic"}:
            value = np.asarray(self._interpolator(query), dtype=float).reshape(-1)
            if value.size != 2 or not np.all(np.isfinite(value)):
                LOGGER.warning(
                    "PIV query (%.3f, %.3f) lies outside the interpolation domain; "
                    "using zero velocity.",
                    x,
                    y,
                )
                return 0.0, 0.0
            return float(value[0]), float(value[1])

        if self.method == "nearest":
            assert self._tree is not None
            _, index = self._tree.query(query, k=1)
            value = self.velocities[int(index)]
            return float(value[0]), float(value[1])

        distances = np.linalg.norm(self.points - query, axis=1)
        exact = np.flatnonzero(distances <= 1e-12)
        if exact.size:
            value = self.velocities[int(exact[0])]
            return float(value[0]), float(value[1])
        weights = 1.0 / np.square(distances)
        weights /= np.sum(weights)
        value = np.sum(self.velocities * weights[:, None], axis=0)
        return float(value[0]), float(value[1])


def evolve_polygon(
    polygon_xy: Iterable[PointXY],
    interpolator: VelocityInterpolator,
    image_shape: Sequence[int],
    *,
    periodic_y: bool = False,
) -> Tuple[PointXY, ...]:
    """Advance an image-space polygon by one PIV displacement field."""

    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image shape: {image_shape}")

    evolved: List[PointXY] = []
    for x, y in polygon_xy:
        u, v = interpolator.velocity_at(float(x), float(y))
        new_x = float(np.clip(float(x) + u, 0.0, width - 1.0))
        unbounded_y = float(y) + v
        if periodic_y:
            new_y = unbounded_y % height
        else:
            new_y = float(np.clip(unbounded_y, 0.0, height - 1.0))
        evolved.append((new_x, new_y))
    return tuple(evolved)


def polygon_to_display(
    polygon_xy: Iterable[PointXY], image_height: int, vertical_shift: int
) -> Tuple[PointXY, ...]:
    """Transform original coordinates to a vertically rolled display image."""

    if image_height <= 0:
        raise ValueError("image_height must be positive.")
    return tuple((float(x), (float(y) + vertical_shift) % image_height) for x, y in polygon_xy)


def polygon_from_display(
    polygon_xy: Iterable[PointXY], image_height: int, vertical_shift: int
) -> Tuple[PointXY, ...]:
    """Transform coordinates on a rolled display back to the original image."""

    if image_height <= 0:
        raise ValueError("image_height must be positive.")
    return tuple((float(x), (float(y) - vertical_shift) % image_height) for x, y in polygon_xy)


def create_piv_static_frames(
    data: PIVData,
    output_dir: Path,
    *,
    step: int = 2,
    dpi: int = 200,
) -> List[Path]:
    """Render one consistently scaled quiver image per PIV frame."""

    if step <= 0 or dpi <= 0:
        raise ValueError("step and dpi must be positive.")
    import matplotlib.pyplot as plt

    from .plotting import sequential_blue_colormap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    magnitudes = [
        np.hypot(u_frame, v_frame) for u_frame, v_frame in zip(data.u_frames, data.v_frames)
    ]
    finite = np.concatenate([magnitude[np.isfinite(magnitude)] for magnitude in magnitudes])
    if finite.size:
        color_min, color_max = np.percentile(finite, [2, 98])
        if color_max <= color_min:
            color_max = color_min + 1.0
    else:
        color_min, color_max = 0.0, 1.0

    paths: List[Path] = []
    for index, (u_frame, v_frame, magnitude) in enumerate(
        zip(data.u_frames, data.v_frames, magnitudes), start=1
    ):
        u_clean = np.nan_to_num(u_frame, nan=0.0, posinf=0.0, neginf=0.0)
        v_clean = np.nan_to_num(v_frame, nan=0.0, posinf=0.0, neginf=0.0)
        magnitude_clean = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
        fig, axis = plt.subplots(figsize=(10, 8))
        quiver = axis.quiver(
            data.x_coords[::step, ::step],
            data.y_coords[::step, ::step],
            u_clean[::step, ::step],
            v_clean[::step, ::step],
            magnitude_clean[::step, ::step],
            scale_units="xy",
            cmap=sequential_blue_colormap(),
        )
        quiver.set_clim(float(color_min), float(color_max))
        axis.set_title(f"PIV vector field — frame {index}/{data.frame_count}")
        axis.set_xlabel("X position (pixels)")
        axis.set_ylabel("Y position (pixels)")
        axis.set_aspect("equal")
        colorbar = fig.colorbar(quiver, ax=axis)
        colorbar.set_label("Velocity magnitude (pixels/frame)")
        path = output_dir / f"piv_frame_{index:04d}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
