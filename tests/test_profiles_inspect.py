import json
from pathlib import Path

import pytest

from blender_fuse.cli import _build_config, build_parser, inspect_analysis
from blender_fuse.profiles import PROFILE_SCHEMA_VERSION, load_analysis_profile


def write_profile(path: Path, analysis: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": PROFILE_SCHEMA_VERSION, "analysis": analysis}),
        encoding="utf-8",
    )
    return path


def test_profile_paths_resolve_relative_to_profile(tmp_path: Path) -> None:
    profile = write_profile(
        tmp_path / "profiles" / "analysis.json",
        {
            "data_dir": "../data",
            "output_dir": "../output",
            "fourier_rect": None,
        },
    )
    config = load_analysis_profile(profile)
    assert config.data_dir == (tmp_path / "data").resolve()
    assert config.output_dir == (tmp_path / "output").resolve()


def test_profile_rejects_unknown_nested_field(tmp_path: Path) -> None:
    profile = write_profile(
        tmp_path / "analysis.json",
        {
            "data_dir": "data",
            "fourier_v2": {"enabled": False, "mystery": 4},
        },
    )
    with pytest.raises(ValueError, match="mystery"):
        load_analysis_profile(profile)


def test_profile_rejects_boolean_integer(tmp_path: Path) -> None:
    profile = write_profile(
        tmp_path / "analysis.json",
        {"data_dir": "data", "n_fold": True},
    )
    with pytest.raises(TypeError, match="n_fold"):
        load_analysis_profile(profile)


def test_explicit_cli_values_override_profile_without_default_leakage(tmp_path: Path) -> None:
    profile = write_profile(
        tmp_path / "analysis.json",
        {
            "data_dir": "data",
            "start_t": 7,
            "n_fold": 4,
            "periodic_y": False,
            "fourier_rect": None,
        },
    )
    parser = build_parser()
    args = parser.parse_args(
        ["analyze", "cli-data", "--profile", str(profile), "--n-fold", "8"]
    )
    config = _build_config(args)
    assert config.data_dir == Path("cli-data").resolve()
    assert config.start_t == 7
    assert config.n_fold == 8
    assert config.periodic_y is False
    assert config.fourier_rect is None


def test_inspect_is_side_effect_free(tmp_path: Path, write_segmentation) -> None:
    data_dir = tmp_path / "data"
    write_segmentation(data_dir, 1, [(4, 4), (8, 8)], shape=(16, 16))
    output_dir = tmp_path / "never-created"
    profile = write_profile(
        tmp_path / "profile.json",
        {
            "data_dir": "data",
            "output_dir": "never-created",
            "start_t": 1,
            "end_t": 1,
            "fourier_rect": [0, 16, 0, 16],
        },
    )
    config = load_analysis_profile(profile)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = inspect_analysis(config)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert before == after
    assert not output_dir.exists()
    assert result["file_count"] == 1
    assert result["squeezed_shapes"] == [(16, 16)]
    assert result["dataset_names"] == ["segmentation"]


def test_inspect_cli_emits_json_without_outputs(tmp_path: Path, write_segmentation, capsys) -> None:
    from blender_fuse.cli import main

    data_dir = tmp_path / "data"
    write_segmentation(data_dir, 2, [(4, 4)], shape=(12, 14))
    profile = write_profile(
        tmp_path / "profile.json",
        {
            "data_dir": "data",
            "output_dir": "not-created",
            "start_t": 2,
            "end_t": 2,
            "fourier_rect": None,
        },
    )
    assert main(["inspect", "--profile", str(profile)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "blender-fuse.inspect.v1"
    assert payload["timepoints"] == [2]
    assert payload["squeezed_shapes"] == [[12, 14]]
    assert not (tmp_path / "not-created").exists()


def test_complete_profile_round_trip_preserves_nulls(tmp_path: Path) -> None:
    from blender_fuse import AnalysisConfig
    from blender_fuse.profiles import write_analysis_profile

    config = AnalysisConfig(
        data_dir=tmp_path / "data",
        output_dir=None,
        dataset_name=None,
        fourier_rect=None,
    )
    path = write_analysis_profile(config, tmp_path / "saved.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["analysis"]["output_dir"] is None
    assert raw["analysis"]["dataset_name"] is None
    loaded = load_analysis_profile(path)
    assert loaded.output_dir is None
    assert loaded.fourier_rect is None


def test_no_fourier_rejects_any_v2_cli_option() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["analyze", "data", "--no-fourier", "--fourier-window", "hann"]
    )
    with pytest.raises(ValueError, match="conflicts"):
        _build_config(args)
