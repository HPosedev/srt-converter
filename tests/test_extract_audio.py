"""Tests de extractor.extract_audio (ffmpeg mockeado, sin binarios reales)."""

import subprocess
from pathlib import Path

import pytest

import extractor
from extractor import extract_audio


def _completed(returncode: int = 0, stderr: str = "") -> object:
    return subprocess.CompletedProcess(args=["x"], returncode=returncode, stdout="", stderr=stderr)


def test_extract_audio_builds_mp3_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Comprueba mapa 0:a:0 y flags mono/16k/64k."""
    mkv = tmp_path / "v.mkv"
    mkv.write_bytes(b"x")
    out = tmp_path / "audio.mp3"
    captured: dict = {}
    monkeypatch.setattr(extractor, "check_dependencies", lambda *a, **k: None)

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        captured["cmd"] = cmd
        Path(out).write_bytes(b"mp3")
        return _completed()

    monkeypatch.setattr(extractor.subprocess, "run", fake_run)
    assert extract_audio(mkv, out) == out
    cmd = captured["cmd"]
    assert cmd[0] == "ffmpeg" and "-n" in cmd
    assert "0:a:0" in cmd
    for flag in ("-ac", "1", "-ar", "16000", "-b:a", "64k", "libmp3lame"):
        assert flag in cmd
    assert cmd[-1].endswith(".mp3")


def test_extract_audio_second_track_and_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """audio_index=1 mapea 0:a:1; overwrite usa -y."""
    mkv = tmp_path / "v.mkv"
    mkv.write_bytes(b"x")
    out = tmp_path / "a.mp3"
    out.write_bytes(b"old")
    captured: dict = {}
    monkeypatch.setattr(extractor, "check_dependencies", lambda *a, **k: None)

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        captured["cmd"] = cmd
        return _completed()

    monkeypatch.setattr(extractor.subprocess, "run", fake_run)
    extract_audio(mkv, out, audio_index=1, overwrite=True)
    assert "0:a:1" in captured["cmd"] and "-y" in captured["cmd"]


def test_extract_audio_guards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Valida extensión, índice, existencia y origen."""
    monkeypatch.setattr(extractor, "check_dependencies", lambda *a, **k: None)
    mkv = tmp_path / "v.mkv"
    mkv.write_bytes(b"x")
    with pytest.raises(ValueError, match=r"\.mp3"):
        extract_audio(mkv, tmp_path / "a.wav")
    with pytest.raises(ValueError, match="audio_index"):
        extract_audio(mkv, tmp_path / "a.mp3", audio_index=-1)
    with pytest.raises(FileNotFoundError):
        extract_audio(tmp_path / "no.mkv", tmp_path / "a.mp3")
    out = tmp_path / "e.mp3"
    out.write_bytes(b"x")
    with pytest.raises(FileExistsError):
        extract_audio(mkv, out)


def test_extract_audio_ffmpeg_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Error de ffmpeg (p. ej. sin pista audio) → RuntimeError."""
    monkeypatch.setattr(extractor, "check_dependencies", lambda *a, **k: None)
    mkv = tmp_path / "v.mkv"
    mkv.write_bytes(b"x")
    monkeypatch.setattr(
        extractor.subprocess, "run", lambda *a, **k: _completed(1, "Stream map matches no streams")
    )
    with pytest.raises(RuntimeError, match="ffmpeg falló"):
        extract_audio(mkv, tmp_path / "a.mp3")
