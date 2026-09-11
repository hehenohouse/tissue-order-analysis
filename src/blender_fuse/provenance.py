"""Atomic run manifests and cryptographic provenance."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ._version import __version__
from .config import AnalysisConfig
from .io import inspect_segmentation_h5
from .profiles import analysis_config_to_dict, canonical_json

MANIFEST_SCHEMA_VERSION = "blender-fuse.run-manifest.v1"
CHECKPOINT_SCHEMA_VERSION = "blender-fuse.checkpoint.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    return atomic_write_text(path, text)


def _dependency_versions() -> Dict[str, Optional[str]]:
    names = (
        "blender-fuse",
        "h5py",
        "matplotlib",
        "numpy",
        "scikit-image",
        "scipy",
        "imageio",
        "imageio-ffmpeg",
    )
    versions: Dict[str, Optional[str]] = {"blender-fuse": __version__}
    for name in names[1:]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_state() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _relative(path: Path, root: Path) -> str:
    try:
        relative = Path(path).resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(os.path.relpath(Path(path).resolve(), root.resolve()))
    return relative.as_posix()


def _manifest_config(config: AnalysisConfig) -> Dict[str, Any]:
    values = analysis_config_to_dict(config)
    values["data_dir"] = "."
    values["output_dir"] = _relative(config.resolved_output_dir, config.data_dir)
    if config.piv.mat_path is not None:
        values["piv"]["mat_path"] = _relative(config.piv.mat_path, config.data_dir)
    return values


@dataclass
class RunProvenance:
    """Prepared source inventory and mutable manifest lifecycle state."""

    config: AnalysisConfig
    input_records: Sequence[Dict[str, Any]]
    missing_timepoints: Sequence[int]
    signature: str
    started_at_utc: str
    profile_record: Optional[Dict[str, Any]]

    @property
    def output_dir(self) -> Path:
        return self.config.resolved_output_dir

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "run_manifest.json"

    def base_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "status": "running",
            "started_at_utc": self.started_at_utc,
            "software": {
                "dependencies": _dependency_versions(),
                "python": sys.version,
                "platform": platform.platform(),
                "git": _git_state(),
            },
            "configuration": _manifest_config(self.config),
            "profile": self.profile_record,
            "run_signature_sha256": self.signature,
            "inputs": list(self.input_records),
            "missing_timepoints": list(self.missing_timepoints),
            "limitations": [
                "Legacy Fourier semantics are retained exactly for compatibility.",
                "Fourier v2 is defined only for rectangular segmentation analysis.",
                "Per-cell tests do not create independent biological replicates.",
            ],
        }

    def write_running(self) -> Path:
        self._write_input_ledger()
        return atomic_write_json(self.manifest_path, self.base_payload())

    def write_failed(self, exc: BaseException, completed: Sequence[int]) -> Path:
        payload = self.base_payload()
        payload.update(
            {
                "status": "failed",
                "ended_at_utc": utc_now(),
                "completed_timepoints": list(completed),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        return atomic_write_json(self.manifest_path, payload)

    def write_complete(
        self,
        *,
        processed: Sequence[int],
        resumed: Sequence[int],
        written_paths: Sequence[Path],
        runtime_seconds: float,
    ) -> Sequence[Path]:
        output_records, ledger_path = self._output_records(written_paths)
        payload = self.base_payload()
        payload.update(
            {
                "status": "complete",
                "ended_at_utc": utc_now(),
                "runtime_seconds": float(runtime_seconds),
                "processed_timepoints": list(processed),
                "resumed_timepoints": list(resumed),
                "outputs": output_records,
            }
        )
        atomic_write_json(self.manifest_path, payload)
        paths = [self.manifest_path]
        input_ledger = self.output_dir / "provenance" / "input_checksums.sha256"
        if input_ledger.exists():
            paths.append(input_ledger)
        if ledger_path is not None:
            paths.append(ledger_path)
        return paths

    def _write_input_ledger(self) -> None:
        if not self.config.provenance.hash_inputs:
            return
        lines = [
            f"{record['sha256']}  {record['path']}"
            for record in self.input_records
            if record.get("sha256") is not None
        ]
        atomic_write_text(
            self.output_dir / "provenance" / "input_checksums.sha256",
            "\n".join(lines) + ("\n" if lines else ""),
        )

    def _output_records(
        self, written_paths: Sequence[Path]
    ) -> tuple[list[Dict[str, Any]], Optional[Path]]:
        excluded = {
            self.manifest_path.resolve(),
            (self.output_dir / "provenance" / "input_checksums.sha256").resolve(),
            (self.output_dir / "provenance" / "output_checksums.sha256").resolve(),
        }
        unique = sorted(
            {
                Path(path).resolve()
                for path in written_paths
                if Path(path).is_file() and Path(path).resolve() not in excluded
            },
            key=lambda path: _relative(path, self.output_dir),
        )
        records = []
        lines = []
        for path in unique:
            digest = sha256_file(path) if self.config.provenance.hash_outputs else None
            relative = _relative(path, self.output_dir)
            records.append(
                {"path": relative, "size_bytes": path.stat().st_size, "sha256": digest}
            )
            if digest is not None:
                lines.append(f"{digest}  {relative}")
        ledger_path: Optional[Path] = None
        if self.config.provenance.hash_outputs:
            ledger_path = self.output_dir / "provenance" / "output_checksums.sha256"
            atomic_write_text(ledger_path, "\n".join(lines) + ("\n" if lines else ""))
        return records, ledger_path


def prepare_run_provenance(
    config: AnalysisConfig,
    files: Mapping[int, Path],
    missing_timepoints: Sequence[int],
) -> RunProvenance:
    """Inspect and optionally hash all selected inputs before a run starts."""

    records = []
    for timepoint, path in sorted(files.items()):
        info = inspect_segmentation_h5(path, config.dataset_name)
        records.append(
            {
                "timepoint": int(timepoint),
                "path": _relative(path, config.data_dir),
                "size_bytes": Path(path).stat().st_size,
                "dataset": info.dataset_name,
                "stored_shape": list(info.stored_shape),
                "squeezed_shape": list(info.squeezed_shape),
                "dtype": info.dtype,
                "sha256": sha256_file(path) if config.provenance.hash_inputs else None,
            }
        )
    profile_path = getattr(config, "_profile_path", None)
    profile_record = None
    if profile_path is not None:
        profile_path = Path(profile_path)
        profile_record = {
            "name": profile_path.name,
            "sha256": sha256_file(profile_path),
        }
    signature_config = analysis_config_to_dict(config)
    signature_config["resume"] = {"enabled": False}
    signature_payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "configuration": signature_config,
        "inputs": records,
        "missing_timepoints": list(missing_timepoints),
    }
    signature = hashlib.sha256(canonical_json(signature_payload).encode("utf-8")).hexdigest()
    return RunProvenance(
        config=config,
        input_records=records,
        missing_timepoints=missing_timepoints,
        signature=signature,
        started_at_utc=utc_now(),
        profile_record=profile_record,
    )
