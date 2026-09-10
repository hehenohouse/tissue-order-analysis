"""Input discovery and validation for segmentation HDF5 files."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)
_TIMEPOINT_FIELD = re.compile(r"\{timepoint(?::([^}]+))?\}")


def _pattern_regex(filename_pattern: str) -> re.Pattern[str]:
    """Translate the supported glob/format pattern into a filename regex."""

    match = _TIMEPOINT_FIELD.search(filename_pattern)
    if match is None:
        raise ValueError("filename_pattern must contain a {timepoint...} field.")

    format_spec = match.group(1) or "d"
    width_match = re.fullmatch(r"0?(\d*)d", format_spec)
    if width_match is None:
        raise ValueError("The timepoint format must be an integer format such as {timepoint:04d}.")
    width = width_match.group(1)
    digits = rf"\d{{{int(width)}}}" if width else r"\d+"

    before = re.escape(filename_pattern[: match.start()])
    after = re.escape(filename_pattern[match.end() :])
    before = before.replace(r"\*", ".*").replace(r"\?", ".")
    after = after.replace(r"\*", ".*").replace(r"\?", ".")
    return re.compile(rf"^{before}(?P<timepoint>{digits}){after}$")


def _discovery_glob(filename_pattern: str) -> str:
    return _TIMEPOINT_FIELD.sub("*", filename_pattern)


def discover_timepoint_files(
    data_dir: Path,
    filename_pattern: str,
    start_t: int = 0,
    end_t: Optional[int] = None,
) -> Tuple[Dict[int, Path], List[int]]:
    """Discover unambiguous files and report missing points in an explicit range."""

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Segmentation data directory does not exist: {data_dir}")

    filename_regex = _pattern_regex(filename_pattern)
    discovered: Dict[int, Path] = {}
    duplicates: Dict[int, List[Path]] = {}

    for path in sorted(data_dir.glob(_discovery_glob(filename_pattern))):
        if not path.is_file():
            continue
        match = filename_regex.match(path.name)
        if match is None:
            continue
        timepoint = int(match.group("timepoint"))
        if timepoint < start_t or (end_t is not None and timepoint > end_t):
            continue
        if timepoint in discovered:
            duplicates.setdefault(timepoint, [discovered[timepoint]]).append(path)
        else:
            discovered[timepoint] = path

    if duplicates:
        detail = "; ".join(
            f"T{timepoint:04d}: {', '.join(str(path) for path in paths)}"
            for timepoint, paths in sorted(duplicates.items())
        )
        raise ValueError(f"Multiple segmentation files match the same timepoint: {detail}")

    if not discovered:
        expected = data_dir / filename_pattern.format(timepoint=start_t)
        raise FileNotFoundError(
            f"No segmentation H5 files were found. The first expected match resembles: {expected}"
        )

    missing: List[int] = []
    if end_t is not None:
        missing = [
            timepoint for timepoint in range(start_t, end_t + 1) if timepoint not in discovered
        ]

    return discovered, missing


def _dataset_paths(handle: h5py.File) -> List[str]:
    paths: List[str] = []

    def collect(name: str, item: object) -> None:
        if isinstance(item, h5py.Dataset):
            paths.append(name)

    handle.visititems(collect)
    return paths


def load_segmentation_h5(
    path: Path,
    dataset_name: Optional[str] = None,
    background_values: Iterable[float] = (0.0, 2.0),
) -> NDArray[np.bool_]:
    """Load one numeric 2-D segmentation and return a foreground mask."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Segmentation file does not exist: {path}")

    with h5py.File(path, "r") as handle:
        if dataset_name is None:
            datasets = _dataset_paths(handle)
            if not datasets:
                raise ValueError(f"H5 file contains no datasets: {path}")
            selected = datasets[0]
            if len(datasets) > 1:
                LOGGER.warning(
                    "%s contains multiple datasets; using the first one, %s. "
                    "Set dataset_name to choose explicitly.",
                    path,
                    selected,
                )
        else:
            selected = dataset_name.lstrip("/")
            if selected not in handle:
                available = _dataset_paths(handle)
                raise KeyError(
                    f"Dataset {dataset_name!r} is not present in {path}; "
                    f"available datasets: {available}"
                )
            if not isinstance(handle[selected], h5py.Dataset):
                raise ValueError(f"H5 object {dataset_name!r} is not a dataset.")

        data = np.asarray(handle[selected])

    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(
            f"Segmentation dataset must squeeze to 2 dimensions; got shape {data.shape} "
            f"from {path}."
        )
    if not np.issubdtype(data.dtype, np.number) and data.dtype != np.bool_:
        raise TypeError(
            f"Segmentation dataset must be numeric or boolean; got {data.dtype} from {path}."
        )
    if np.issubdtype(data.dtype, np.floating) and not np.all(np.isfinite(data)):
        raise ValueError(f"Segmentation dataset contains NaN or infinite values: {path}")

    background = np.asarray(tuple(background_values))
    return np.asarray(~np.isin(data, background), dtype=bool)
