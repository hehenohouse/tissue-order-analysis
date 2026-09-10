from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import h5py
import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture
def write_segmentation():
    def write(
        directory: Path,
        timepoint: int,
        foreground: Iterable[Tuple[int, int]],
        shape: Tuple[int, int] = (32, 32),
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        data = np.full(shape, 2, dtype=np.uint8)
        for row, column in foreground:
            data[row, column] = 1
        path = directory / f"sample_T{timepoint:04d}_Simple Segmentation.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("segmentation", data=data)
        return path

    return write
