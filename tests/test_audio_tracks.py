"""Tests de pistas de audio (ffprobe mockeado) y CLI de audio."""

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

import app
import extractor
from extractor import (
    AudioTrack,
    extract_audio,
    format_audio_tracks_table,
    list_audio_tracks,
    select_audio_track,
)

runner = CliRunner()

FFPROBE_AUDIO_JSON = {
    "streams": [
        {
            "index": 1,
            "codec_name": "aac",
            "channels": 2,
            "disposition": {"default": 1},
            "tags": {"language": "eng", "title": "Stereo"},
        },
        {
            "index": 2,
            "codec_name": "ac3",
            "channels": 6,
            "disposition": {"default": 0},
            "tags": {"language": "spa"},
        },
        {
            "index": 3,
            "codec_name": "opus",
            "disposition": {"default": 0},
            "tags": {},
        },
    ]
}


def _completed(stdout: str = "", returncode: int = 0) -> object:
    return subprocess.CompletedProcess(args=["x"], returncode=returncode, stdout=stdout, stderr="")


def _tracks() -> list[AudioTrack]:
    return [
        AudioTrack(1, 0, "eng", "Stereo", "aac", 2, True),
        AudioTrack(2, 1, "spa", "", "ac3", 6, False),
        AudioTrack(3, 2, "und", "", "opus", 0, False),
    ]


def test_audio_track_describe() -> None:
    """describe() incluye índices, codec, canales e idioma."""
    text = _tracks()[1].describe()
    assert "a:1" in text and "0:2" in text
    assert "ac3" in text and "6ch" in text and "spa" in text


def test_list_audio_tracks_parses_ffprobe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """El JSON de audio se mapea con audio_index 0-based y defaults."""
    mkv = tmp_path / "v.mkv"
    mkv.write_bytes(b"x")
    monkeypatch.setattr(extractor, "check_dependencies", lambda *a, **k: None)
    seen: dict = {}

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        seen["cmd"] = cmd
        return _completed(json.dumps(FFPROBE_AUDIO_JSON))

    monkeypatch.setattr(extractor.subprocess, "run", fake_run)
    tracks = list_audio_tracks(mkv)
    assert [t.audio_index for t in tracks] == [0, 1, 2]
    assert [t.index for t in tracks] == [1, 2, 3]
    assert tracks[0].lang == "eng" and tracks[0].is_default is True
    assert tracks[2].lang == "und" and tracks[2].channels == 0
    assert "-select_streams" in seen["cmd"] and "a" in seen["cmd"]


def test_list_audio_tracks_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Archivo inexistente → FileNotFoundError antes de ffprobe."""
    monkeypatch.setattr(extractor, "check_dependencies", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError):
        list_audio_tracks(tmp_path / "no.mkv")


def test_select_audio_track_priority() -> None:
    """Índice explícito > idioma > default > primera."""
    tracks = _tracks()
    assert select_audio_track(tracks, track_index=1).lang == "spa"
    assert select_audio_track(tracks, lang="spa").audio_index == 1
    assert select_audio_track(tracks, lang="en").lang == "eng"  # tolerante en/eng
    assert select_audio_track(tracks, lang="eng").is_default is True
    assert select_audio_track(tracks).is_default is True  # default gana
    plain = [AudioTrack(5, 0, "fra", "", "aac", 2), AudioTrack(6, 1, "deu", "", "aac", 2)]
    assert select_audio_track(plain).audio_index == 0  # sin default → primera


def test_select_audio_track_errors() -> None:
    """Sin pistas, índice malo o idioma ausente → ValueError claro."""
    with pytest.raises(ValueError, match="no contiene pistas de audio"):
        select_audio_track([])
    with pytest.raises(ValueError, match="No existe la pista"):
        select_audio_track(_tracks(), track_index=9)
    with pytest.raises(ValueError, match="Ninguna pista"):
        select_audio_track(_tracks(), lang="jpn")


def test_extract_audio_with_track_uses_global_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Con AudioTrack se mapea 0:<global>, con prioridad sobre audio_index."""
    mkv = tmp_path / "v.mkv"
    mkv.write_bytes(b"x")
    out = tmp_path / "a.mp3"
    captured: dict = {}
    monkeypatch.setattr(extractor, "check_dependencies", lambda *a, **k: None)

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        captured["cmd"] = cmd
        Path(out).write_bytes(b"mp3")
        return _completed()

    monkeypatch.setattr(extractor.subprocess, "run", fake_run)
    extract_audio(mkv, out, audio_index=0, track=_tracks()[1])
    assert "0:2" in captured["cmd"]  # global, no 0:a:0


def test_extract_audio_with_stream_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Con stream_index global se mapea 0:<index>."""
    mkv = tmp_path / "v.mkv"
    mkv.write_bytes(b"x")
    out = tmp_path / "a.mp3"
    captured: dict = {}
    monkeypatch.setattr(extractor, "check_dependencies", lambda *a, **k: None)

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        captured["cmd"] = cmd
        return _completed()

    monkeypatch.setattr(extractor.subprocess, "run", fake_run)
    extract_audio(mkv, out, stream_index=5)
    assert "0:5" in captured["cmd"]


def test_format_audio_tracks_table() -> None:
    """Tabla vacía con mensaje; con pistas muestra índices a:N."""
    assert "no contiene" in format_audio_tracks_table([]).lower()
    assert "[1]" in format_audio_tracks_table(_tracks())


def test_extract_list_audio_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """extract --list-audio muestra la tabla de sonido."""
    monkeypatch.setattr(app, "list_audio_tracks", lambda m: _tracks())
    mkv = tmp_path / "v.mkv"
    mkv.write_bytes(b"x")
    result = runner.invoke(app.app, ["extract", str(mkv), "--list-audio"])
    assert result.exit_code == 0, result.output
    assert "a:0" in result.output and "aac" in result.output


def test_extract_audio_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """extract --audio elige por idioma y genera el .mp3."""
    monkeypatch.setattr(app, "list_audio_tracks", lambda m: _tracks())
    seen: dict = {}

    def fake_extract(
        mkv: Path, out: Path, track: AudioTrack | None = None, overwrite: bool = False
    ) -> Path:
        seen["track"] = track
        Path(out).write_bytes(b"mp3")
        return Path(out)

    monkeypatch.setattr(app, "extract_audio", fake_extract)
    mkv = tmp_path / "v.mkv"
    mkv.write_bytes(b"x")
    result = runner.invoke(app.app, ["extract", str(mkv), "--audio", "--audio-lang", "spa"])
    assert result.exit_code == 0, result.output
    assert seen["track"].lang == "spa"
    assert "lang=spa" in result.output


def test_auto_audio_lang_selects_track(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """auto --from-audio --audio-lang spa usa esa pista para el MP3."""
    from config import AppConfig, GeminiConfig

    monkeypatch.setattr(app, "list_audio_tracks", lambda m: _tracks())
    monkeypatch.setattr(
        app, "load_config", lambda path=None: AppConfig(gemini=GeminiConfig(api_key="x"))
    )
    seen: dict = {}

    def fake_extract(
        mkv: Path, out: Path, track: AudioTrack | None = None, overwrite: bool = True
    ) -> Path:
        seen["track"] = track
        Path(out).write_bytes(b"mp3")
        return Path(out)

    monkeypatch.setattr(app, "extract_audio", fake_extract)

    class DummySub:
        """Doble mínimo con transcripción fija."""

        def transcribe_and_translate_audio(self, audio: Path) -> list:
            """Devuelve un bloque traducido."""
            from srt_utils import parse_srt_content

            return parse_srt_content("1\n00:00:01,000 --> 00:00:02,000\nHola\n")

    monkeypatch.setattr(app, "build_subtitler", lambda cfg, window_size=None: DummySub())
    mkv = tmp_path / "epi.mkv"
    mkv.write_bytes(b"x")
    result = runner.invoke(
        app.app, ["auto", str(mkv), "--from-audio", "--audio-lang", "spa"]
    )
    assert result.exit_code == 0, result.output
    assert seen["track"].lang == "spa"
    assert (tmp_path / "epi_es.srt").is_file()
