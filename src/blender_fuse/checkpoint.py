"""Validated per-frame checkpoints for resumable analyses."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .models import (
    BootstrapResult,
    ComplexPsiSummary,
    FrameSummary,
    HarmonicResult,
    SegmentationQC,
    StatisticalResult,
)
from .provenance import CHECKPOINT_SCHEMA_VERSION, atomic_write_json, sha256_file


def _optional_array(values: Sequence[Any], dtype: Any) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def _summary_payload(summary: FrameSummary) -> Dict[str, Any]:
    payload = asdict(summary)
    payload["source_path"] = str(summary.source_path)
    return payload


def _statistical_payload(result: Optional[StatisticalResult]) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    payload = asdict(result)
    payload["bin_centers"] = result.bin_centers.tolist()
    payload["bin_means"] = [
        float(value) if np.isfinite(value) else None for value in result.bin_means
    ]
    payload["bin_counts"] = result.bin_counts.tolist()
    return payload


def _artifact_records(paths: Sequence[Path], output_dir: Path) -> list[Dict[str, Any]]:
    records = []
    for path in sorted({Path(item).resolve() for item in paths}, key=str):
        relative = path.relative_to(output_dir.resolve()).as_posix()
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


class CheckpointManager:
    """Write and validate private run/frame checkpoint records."""

    def __init__(
        self,
        output_dir: Path,
        signature: str,
        expected_timepoints: Sequence[int],
        *,
        resume: bool,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.signature = signature
        self.expected_timepoints = tuple(int(value) for value in expected_timepoints)
        self.resume = bool(resume)
        self.run_path = self.output_dir / "checkpoints" / "run.json"
        self.frames_dir = self.output_dir / "checkpoints" / "frames"

    def initialize(self) -> None:
        expected = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_signature_sha256": self.signature,
            "expected_timepoints": list(self.expected_timepoints),
        }
        if self.resume:
            if self.run_path.is_file():
                current = json.loads(self.run_path.read_text(encoding="utf-8"))
                if current != expected:
                    raise ValueError(
                        "Cannot resume: configuration, input files, or checkpoint schema "
                        "changed."
                    )
                return
            if self.output_dir.is_dir() and any(self.output_dir.iterdir()):
                raise ValueError(
                    "Cannot resume: the output directory is nonempty but has no compatible "
                    "run checkpoint. Choose a new output directory or run without resume."
                )
        atomic_write_json(self.run_path, expected)

    def frame_path(self, timepoint: int) -> Path:
        return self.frames_dir / f"frame_T{int(timepoint):04d}.json"

    def load(self, timepoint: int) -> Optional[Dict[str, Any]]:
        path = self.frame_path(timepoint)
        if not self.resume or not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"Cannot resume: invalid checkpoint schema in {path}.")
        if record.get("run_signature_sha256") != self.signature:
            raise ValueError(f"Cannot resume: checkpoint signature mismatch in {path}.")
        if int(record.get("timepoint", -1)) != int(timepoint):
            raise ValueError(f"Cannot resume: checkpoint timepoint mismatch in {path}.")
        for artifact in record.get("artifacts", []):
            artifact_path = self.output_dir / artifact["path"]
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"Cannot resume: checkpoint artifact is missing: {artifact_path}"
                )
            if artifact_path.stat().st_size != int(artifact["size_bytes"]):
                raise ValueError(f"Cannot resume: artifact size changed: {artifact_path}")
            if sha256_file(artifact_path) != artifact["sha256"]:
                raise ValueError(f"Cannot resume: artifact checksum changed: {artifact_path}")
        return record

    def save(
        self,
        *,
        summary: FrameSummary,
        statistics: Optional[StatisticalResult],
        harmonic: Sequence[HarmonicResult],
        quality: Optional[SegmentationQC],
        bootstrap: Sequence[BootstrapResult],
        artifacts: Sequence[Path],
    ) -> Path:
        record = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_signature_sha256": self.signature,
            "timepoint": summary.timepoint,
            "summary": _summary_payload(summary),
            "statistics": _statistical_payload(statistics),
            "harmonic": [asdict(item) for item in harmonic],
            "segmentation_qc": asdict(quality) if quality is not None else None,
            "bootstrap": [asdict(item) for item in bootstrap],
            "artifacts": _artifact_records(artifacts, self.output_dir),
        }
        return atomic_write_json(self.frame_path(summary.timepoint), record)


def restore_checkpoint(
    record: Dict[str, Any],
) -> Tuple[
    FrameSummary,
    Optional[StatisticalResult],
    Tuple[HarmonicResult, ...],
    Optional[SegmentationQC],
    Tuple[BootstrapResult, ...],
    Tuple[str, ...],
]:
    """Rehydrate lightweight run state from a validated frame checkpoint."""

    summary_values = dict(record["summary"])
    summary_values["source_path"] = Path(summary_values["source_path"])
    for name in ("complex_psi_cell", "complex_psi_area"):
        if summary_values.get(name) is not None:
            summary_values[name] = ComplexPsiSummary(**summary_values[name])
    summary = FrameSummary(**summary_values)

    statistics = None
    if record.get("statistics") is not None:
        values = dict(record["statistics"])
        values["bin_centers"] = _optional_array(values["bin_centers"], float)
        values["bin_means"] = _optional_array(
            [np.nan if value is None else value for value in values["bin_means"]], float
        )
        values["bin_counts"] = _optional_array(values["bin_counts"], int)
        statistics = StatisticalResult(**values)
    harmonic = tuple(HarmonicResult(**item) for item in record.get("harmonic", []))
    quality = (
        SegmentationQC(**record["segmentation_qc"])
        if record.get("segmentation_qc") is not None
        else None
    )
    bootstrap = tuple(BootstrapResult(**item) for item in record.get("bootstrap", []))
    artifacts = tuple(item["path"] for item in record.get("artifacts", []))
    return summary, statistics, harmonic, quality, bootstrap, artifacts
