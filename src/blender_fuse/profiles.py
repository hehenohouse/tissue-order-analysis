"""Strict JSON profiles for reproducible Blender Fuse analyses."""

from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type

from .config import (
    AnalysisConfig,
    FourierV2Config,
    OutputOptions,
    PIVConfig,
    ProvenanceConfig,
    ResumeConfig,
    SegmentationFilterConfig,
    SpatialBootstrapConfig,
)

PROFILE_SCHEMA_VERSION = "blender-fuse.analysis-profile.v1"
_NESTED_TYPES = {
    "outputs": OutputOptions,
    "piv": PIVConfig,
    "segmentation_filter": SegmentationFilterConfig,
    "fourier_v2": FourierV2Config,
    "spatial_bootstrap": SpatialBootstrapConfig,
    "resume": ResumeConfig,
    "provenance": ProvenanceConfig,
}


def _reject_unknown(data: Mapping[str, Any], cls: Type[Any], context: str) -> None:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown {context} field(s): {', '.join(unknown)}")


def _resolve_path(value: Any, base_dir: Optional[Path], context: str) -> Optional[Path]:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{context} must be a string path or null.")
    path = Path(value).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)



def analysis_config_from_dict(
    data: Mapping[str, Any], *, base_dir: Optional[Path] = None
) -> AnalysisConfig:
    """Construct and validate a configuration from a strict analysis mapping."""

    if not isinstance(data, Mapping):
        raise TypeError("analysis must be a JSON object.")
    _reject_unknown(data, AnalysisConfig, "analysis")
    values: Dict[str, Any] = dict(data)
    if "data_dir" not in values:
        raise ValueError("analysis.data_dir is required.")
    values["data_dir"] = _resolve_path(values["data_dir"], base_dir, "analysis.data_dir")
    if "output_dir" in values:
        values["output_dir"] = _resolve_path(
            values["output_dir"], base_dir, "analysis.output_dir"
        )
    if "background_values" in values:
        if not isinstance(values["background_values"], list):
            raise TypeError("analysis.background_values must be a JSON array.")
        values["background_values"] = tuple(values["background_values"])
    if "fourier_rect" in values and values["fourier_rect"] is not None:
        if not isinstance(values["fourier_rect"], list):
            raise TypeError("analysis.fourier_rect must be a JSON array or null.")
        values["fourier_rect"] = tuple(values["fourier_rect"])

    for name, nested_cls in _NESTED_TYPES.items():
        if name not in values:
            continue
        nested_data = values[name]
        if not isinstance(nested_data, Mapping):
            raise TypeError(f"analysis.{name} must be a JSON object.")
        nested_values: Dict[str, Any] = dict(nested_data)
        _reject_unknown(nested_values, nested_cls, f"analysis.{name}")
        if name == "piv":
            if "mat_path" in nested_values:
                nested_values["mat_path"] = _resolve_path(
                    nested_values["mat_path"], base_dir, "analysis.piv.mat_path"
                )
            if "roi_polygon" in nested_values:
                polygon = nested_values["roi_polygon"]
                if not isinstance(polygon, list):
                    raise TypeError("analysis.piv.roi_polygon must be a JSON array.")
                nested_values["roi_polygon"] = tuple(tuple(point) for point in polygon)
        values[name] = nested_cls(**nested_values)

    config = AnalysisConfig(**values)
    config.validate()
    return config


def load_analysis_profile(path: Path) -> AnalysisConfig:
    """Load a strict versioned JSON profile with profile-relative paths."""

    profile_path = Path(path)
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON profile {profile_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise TypeError("Analysis profile root must be a JSON object.")
    allowed = {"schema_version", "analysis"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown profile field(s): {', '.join(unknown)}")
    if raw.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported profile schema_version {raw.get('schema_version')!r}; "
            f"expected {PROFILE_SCHEMA_VERSION!r}."
        )
    if "analysis" not in raw:
        raise ValueError("Profile must contain an analysis object.")
    config = analysis_config_from_dict(
        raw["analysis"], base_dir=profile_path.parent.resolve()
    )
    config._profile_path = profile_path.resolve()
    return config


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def analysis_config_to_dict(config: AnalysisConfig) -> Dict[str, Any]:
    """Return a JSON-safe effective configuration mapping."""

    return _json_value(config)


def profile_document(config: AnalysisConfig) -> Dict[str, Any]:
    """Return a complete versioned profile document."""

    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "analysis": analysis_config_to_dict(config),
    }


def merge_analysis_config(
    config: AnalysisConfig, overrides: Mapping[str, Any]
) -> AnalysisConfig:
    """Deep-merge explicit overrides onto an existing validated configuration."""

    merged = analysis_config_to_dict(config)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return analysis_config_from_dict(merged)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically and reject non-standard floating values."""

    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_analysis_profile(config: AnalysisConfig, path: Path) -> Path:
    """Write a versioned profile document atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                profile_document(config),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
