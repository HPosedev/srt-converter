"""Tests unitarios de extractor (con subprocess mockeado, sin ffmpeg real)."""

import json
import subprocess
from pathlib import Path

import pytest

import extractor
from extractor import (
    FFmpegNotFoundError,
    SubtitleTrack,
    extract_subtitle,
    format_tracks_table,
    is_text_based,
    list_subtitle_tracks,
    select_track,
)

FFPROBE_JSON = {
    "streams": [
        {
            "index": 2,
            "codec_name": "subrip",
            "disposition": {"default": 1, "forced": 0},
            "tags": {"language": "eng", "title": "Full"},
        },
        {
            "index": 3,
            "codec_name": "subrip",
            "disposition": {"default": 0, "forced": 1},
            "tags": {"language": "eng", "title": "Forced"},
        },
        {
            "index": 4,
            "codec_name": "hdmv_pgs_subtitle",
            "disposition": {"default": 0, "forced": 0},
            "tags": {"language": "spa"},
        },
    ]
}


def _completed(stdout: str = "", returncode: int = 0, stderr: str = "") -> object:
    return subprocess.CompletedProcess(args=["x"], returncode=returncode, stdout=stdout, stderr=stderr)


def _tracks() -> list[SubtitleTrack]:
    return [
        SubtitleTrack(2, 0, "eng", "Full", "subrip", False, True),
        SubtitleTrack(3, 1, "eng", "Forced", "subrip", True, False),
        SubtitleTrack(4, 2, "spa", "", "hdmv_pgs_subtitle", False, False),
    ]


def test_track_kind_and_describe() -> None:
    """kind distingue forzados/completos y describe() es informativo."""
    full, forced = _tracks()[:2]
    assert full.kind == "completos"
    assert forced.kind == "forzados"
    assert "forzados" in forced.describe()
    assert "s:1" in forced.describe()


def test_list_subtitle_tracks_parses_ffprobe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """El JSON de ffprobe se mapea a SubtitleTrack con sub_index 0-based."""
    mkv = tmp_path / "video.mkv"
    mkv.write_bytes(b"fake")
    monkeypatch.setattr(extractor, "check_dependencies", lambda *a, **k: None)
    monkeypatch.setattr(
        extractor.subprocess, "run", lambda *a, **k: _completed(json.dumps(FFPROBE_JSON))
    )
    tracks = list_subtitle_tracks(mkv)
    assert len(tracks) == 3
    assert tracks[0].language == "eng" and tracks[0].sub_index == 0
    assert tracks[1].forced is True
    assert tracks[2].codec_name == "hdmv_pgs_subtitle"


def test_list_missing_file_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Archivo inexistente → FileNotFoundError antes de llamar a ffprobe."""
    monkeypatch.setattr(extractor, "check_dependencies", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError):
        list_subtitle_tracks(tmp_path / "noexiste.mkv")


def test_select_track_rules() -> None:
    """Selección: única, por índice, por idioma, ambigüedad y sin match."""
    tracks = _tracks()
    assert select_track([tracks[0]]).sub_index == 0  # única → directa
    assert select_track(tracks, sub_index=2).language == "spa"
    assert select_track(tracks, lang="spa").sub_index == 2
    with pytest.raises(ValueError, match="varias pistas"):
        select_track(tracks)  # sin filtro y varias → pedir criterio
    with pytest.raises(ValueError, match="ambiguo"):
        select_track(tracks, lang="eng")  # dos eng → pedir --track
    with pytest.raises(ValueError, match="Ninguna pista"):
        select_track(tracks, lang="fra")
    with pytest.raises(ValueError, match="No existe"):
        select_track(tracks, sub_index=9)


def test_is_text_based_rejects_pgs() -> None:
    """PGS/DVD son imagen → no convertibles a SRT sin OCR."""
    tracks = _tracks()
    assert is_text_based(tracks[0]) is True
    assert is_text_based(tracks[2]) is False


def test_extract_builds_ffmpeg_map(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """El map usa el índice global (0:<index>) y crea el .srt."""
    mkv = tmp_path / "v.mkv"
    mkv.write_bytes(b"x")
    out = tmp_path / "out.srt"
    captured: dict = {}
    monkeypatch.setattr(extractor, "check_dependencies", lambda *a, **k: None)

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        captured["cmd"] = cmd
        Path(out).write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n", encoding="utf-8")
        return _completed()

    monkeypatch.setattr(extractor.subprocess, "run", fake_run)
    result = extract_subtitle(mkv, out, _tracks()[0])
    assert result == out
    cmd = captured["cmd"]
    assert cmd[:2] == ["ffmpeg", "-n"]
    assert "-map" in cmd and "0:2" in cmd  # índice global, no sub_index
    assert cmd[-1].endswith(".srt")


def test_extract_refuses_pgs_without_ocr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pista de imagen → ValueError explicativo, sin invocar ffmpeg."""
    monkeypatch.setattr(extractor, "check_dependencies", lambda *a, **k: None)
    mkv = tmp_path / "v.mkv"
    mkv.write_bytes(b"x")
    with pytest.raises(ValueError, match="sin OCR"):
        extract_subtitle(mkv, tmp_path / "o.srt", _tracks()[2])


def test_extract_refuses_existing_without_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No sobrescribe por defecto (la traducción tampoco lo hará)."""
    monkeypatch.setattr(extractor, "check_dependencies", lambda *a, **k: None)
    mkv = tmp_path / "v.mkv"
    mkv.write_bytes(b"x")
    out = tmp_path / "o.srt"
    out.write_text("x")
    with pytest.raises(FileExistsError):
        extract_subtitle(mkv, out, 2)


def test_extract_rejects_non_srt_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """El destino debe ser .srt."""
    monkeypatch.setattr(extractor, "check_dependencies", lambda *a, **k: None)
    mkv = tmp_path / "v.mkv"
    mkv.write_bytes(b"x")
    with pytest.raises(ValueError, match=r"\.srt"):
        extract_subtitle(mkv, tmp_path / "o.ass", 2)


def test_check_dependencies_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin ffmpeg/ffprobe → error claro con la orden de instalación."""
    monkeypatch.setattr(extractor.shutil, "which", lambda name: None)
    with pytest.raises(FFmpegNotFoundError, match="pacman -S ffmpeg"):
        extractor.check_dependencies()


def test_format_tracks_table_empty() -> None:
    """Sin pistas, mensaje claro en vez de tabla vacía."""
    assert "no contiene" in format_tracks_table([]).lower()
    assert "[0]" in format_tracks_table(_tracks()[:1])
