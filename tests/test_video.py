from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from blender_fuse.plotting import _as_uint8_rgb, create_video


def test_video_frame_normalization_handles_grayscale_and_rgba() -> None:
    gray = np.asarray([[0, 255]], dtype=np.uint8)
    normalized = _as_uint8_rgb(gray)
    assert normalized.shape == (1, 2, 3)
    np.testing.assert_array_equal(normalized[0, 1], [255, 255, 255])

    rgba = np.asarray([[[255, 0, 0, 0], [255, 0, 0, 255]]], dtype=np.uint8)
    normalized = _as_uint8_rgb(rgba)
    np.testing.assert_array_equal(normalized[0, 0], [252, 252, 251])
    np.testing.assert_array_equal(normalized[0, 1], [255, 0, 0])


def test_create_video_center_pads_variable_frames_and_publishes_atomically(
    tmp_path: Path, monkeypatch
) -> None:
    frames = {
        "first.png": np.full((3, 5, 4), 255, dtype=np.uint8),
        "second.png": np.full((5, 7, 3), 17, dtype=np.uint8),
    }
    reads = []
    appended = []

    def fake_read(path):
        reads.append(Path(path).name)
        return frames[Path(path).name]

    class Writer:
        def __init__(self, path):
            self.path = Path(path)

        def __enter__(self):
            return self

        def append_data(self, frame):
            appended.append(frame.copy())

        def __exit__(self, exc_type, exc, traceback):
            if exc_type is None:
                self.path.write_bytes(b"video")

    monkeypatch.setattr(imageio, "imread", fake_read)
    monkeypatch.setattr(imageio, "get_writer", lambda path, **kwargs: Writer(path))
    destination = tmp_path / "movie.mp4"
    result = create_video(
        [tmp_path / "first.png", tmp_path / "second.png"], destination, fps=2
    )
    assert result == destination
    assert destination.read_bytes() == b"video"
    assert reads == ["first.png", "second.png", "first.png", "second.png"]
    assert [frame.shape for frame in appended] == [(16, 16, 3), (16, 16, 3)]
    np.testing.assert_array_equal(appended[1][5:10, 4:11], 17)


def test_video_failure_preserves_existing_destination(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "movie.mp4"
    destination.write_bytes(b"existing")
    monkeypatch.setattr(
        imageio, "imread", lambda _: np.zeros((4, 4, 3), dtype=np.uint8)
    )

    class FailingWriter:
        def __init__(self, path):
            self.path = Path(path)

        def __enter__(self):
            return self

        def append_data(self, frame):
            raise RuntimeError("encoder failed")

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(
        imageio, "get_writer", lambda path, **kwargs: FailingWriter(path)
    )
    with pytest.raises(RuntimeError, match="encoder failed"):
        create_video([tmp_path / "frame.png"], destination, fps=2)
    assert destination.read_bytes() == b"existing"
    assert not (tmp_path / ".movie.tmp.mp4").exists()


@pytest.mark.parametrize("fps", [0, -1, float("nan")])
def test_video_rejects_invalid_fps(tmp_path: Path, fps: float) -> None:
    with pytest.raises(ValueError, match="fps"):
        create_video([tmp_path / "frame.png"], tmp_path / "movie.mp4", fps)
