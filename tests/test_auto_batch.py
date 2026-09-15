"""Tests de auto en modo directorio, SIGINT y punto de entrada CLI."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import app
from app import cleanup_temps, expected_final, iter_auto_sources
from extractor import AudioTrack, SubtitleTrack

runner = CliRunner()

SAMPLE = """1
00:00:01,000 --> 00:00:03,000
Hello world
"""


class DummySubtitler:
    """Traductor falso sin red."""

    window_size = 50
    fallback_notices: list = []

    def _translate_window(self, window: list) -> list:
        """Antepone 'ES:' a cada bloque."""
        from srt_utils import replace_contents

        return replace_contents(window, [f"ES:{s.content}" for s in window])


def _patch_common(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config y subtitulador falsos; pistas de subtítulo de texto."""
    from config import AppConfig, GeminiConfig

    monkeypatch.setattr(app, "build_subtitler", lambda cfg, window_size=None: DummySubtitler())
    monkeypatch.setattr(
        app, "load_config", lambda path=None: AppConfig(gemini=GeminiConfig(api_key="x"))
    )
    tracks = [SubtitleTrack(2, 0, "eng", "", "subrip", False, True)]
    monkeypatch.setattr(app, "list_subtitle_tracks", lambda m: tracks)


def _video(path: Path, name: str) -> Path:
    """Crea un vídeo falso."""
    video = path / name
    video.write_bytes(b"x")
    return video


def test_auto_dir_processes_videos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Carpeta con .mkv/.mp4 → un *_es.srt por vídeo (ignora otros)."""
    _patch_common(monkeypatch)

    def fake_extract(mkv: Path, out: Path, track: object, overwrite: bool) -> Path:
        Path(out).write_text(SAMPLE, encoding="utf-8")
        return Path(out)

    monkeypatch.setattr(app, "extract_subtitle", fake_extract)
    _video(tmp_path, "cap01.mkv")
    _video(tmp_path, "cap02.mp4")
    (tmp_path / "nota.txt").write_text("no es vídeo")
    result = runner.invoke(app.app, ["auto", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "cap01_es.srt").is_file()
    assert (tmp_path / "cap02_es.srt").is_file()
    assert "2/2" in result.output


def test_auto_dir_skips_already_translated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Vídeos con *_es.srt existente se omiten sin llamar a ffmpeg."""
    _patch_common(monkeypatch)
    calls: list = []

    def fake_extract(mkv: Path, out: Path, track: object, overwrite: bool) -> Path:
        calls.append(mkv)
        Path(out).write_text(SAMPLE, encoding="utf-8")
        return Path(out)

    monkeypatch.setattr(app, "extract_subtitle", fake_extract)
    _video(tmp_path, "hecho.mkv")
    (tmp_path / "hecho_es.srt").write_text("ya estaba", encoding="utf-8")
    _video(tmp_path, "nuevo.mkv")
    result = runner.invoke(app.app, ["auto", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and calls[0].name == "nuevo.mkv"
    assert "Omitido hecho.mkv" in result.output
    assert (tmp_path / "hecho_es.srt").read_text(encoding="utf-8") == "ya estaba"


def test_auto_dir_continues_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Un vídeo roto no bloquea el resto; salida 1 al final."""
    _patch_common(monkeypatch)

    def fake_extract(mkv: Path, out: Path, track: object, overwrite: bool) -> Path:
        if mkv.name == "roto.mkv":
            raise RuntimeError("ffmpeg falló")
        Path(out).write_text(SAMPLE, encoding="utf-8")
        return Path(out)

    monkeypatch.setattr(app, "extract_subtitle", fake_extract)
    _video(tmp_path, "roto.mkv")
    _video(tmp_path, "bueno.mkv")
    result = runner.invoke(app.app, ["auto", str(tmp_path)])
    assert result.exit_code == 1
    assert "Error en roto.mkv" in result.output
    assert (tmp_path / "bueno_es.srt").is_file()


def test_auto_dir_empty_and_output_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sin vídeos → mensaje; --output con carpeta → error."""
    _patch_common(monkeypatch)
    result = runner.invoke(app.app, ["auto", str(tmp_path)])
    assert result.exit_code == 0
    assert "Sin vídeos" in result.output
    _video(tmp_path, "a.mkv")
    result = runner.invoke(app.app, ["auto", str(tmp_path), "--output", "x.srt"])
    assert result.exit_code != 0


def test_auto_interrupt_cleans_audio_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ctrl+C en vía audio borra el .mp3 y sale 130."""
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        app, "list_audio_tracks", lambda m: [AudioTrack(1, 0, "eng", "", "aac", 2, True)]
    )

    def fake_extract_audio(mkv: Path, out: Path, **kwargs: object) -> Path:
        Path(out).write_bytes(b"mp3")
        return Path(out)

    monkeypatch.setattr(app, "extract_audio", fake_extract_audio)

    class InterruptingSub:
        """Corta a mitad de la transcripción."""

        def transcribe_and_translate_audio(self, audio: Path) -> list:
            """Simula Ctrl+C del usuario."""
            raise KeyboardInterrupt

    monkeypatch.setattr(app, "build_subtitler", lambda cfg, window_size=None: InterruptingSub())
    mkv = _video(tmp_path, "epi.mkv")
    result = runner.invoke(app.app, ["auto", str(mkv), "--from-audio"])
    assert result.exit_code == 130
    assert not tmp_path.joinpath("epi_es.auto-audio.mp3").is_file()
    assert not tmp_path.joinpath("epi_es.srt").is_file()


def test_auto_interrupt_cleans_subtitle_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ctrl+C en vía subtítulo borra el .extracted.srt y sale 130."""
    _patch_common(monkeypatch)

    def fake_extract(mkv: Path, out: Path, track: object, overwrite: bool) -> Path:
        Path(out).write_text(SAMPLE, encoding="utf-8")
        return Path(out)

    monkeypatch.setattr(app, "extract_subtitle", fake_extract)

    def boom(*args: object, **kwargs: object) -> Path:
        raise KeyboardInterrupt

    monkeypatch.setattr(app, "run_translate_single", boom)
    mkv = _video(tmp_path, "epi.mkv")
    result = runner.invoke(app.app, ["auto", str(mkv)])
    assert result.exit_code == 130
    assert not tmp_path.joinpath("epi_es.extracted.srt").is_file()


def test_cleanup_temps_keeps_cache(tmp_path: Path) -> None:
    """cleanup_temps borra intermedios pero nunca la caché."""
    mp3 = tmp_path / "a.auto-audio.mp3"
    srt = tmp_path / "a.extracted.srt"
    cache = tmp_path / "a.subtrans-cache.json"
    mp3.write_bytes(b"x")
    srt.write_bytes(b"x")
    cache.write_text("{}", encoding="utf-8")
    cleanup_temps(mp3, srt, tmp_path / "noexiste.tmp")
    assert not mp3.is_file() and not srt.is_file()
    assert cache.is_file()


def test_iter_auto_sources_and_expected_final(tmp_path: Path) -> None:
    """Solo .mkv/.mp4 ordenados; destino *_es.srt junto al vídeo."""
    _video(tmp_path, "b.mkv")
    _video(tmp_path, "a.mp4")
    (tmp_path / "c.avi").write_bytes(b"x")
    sources = iter_auto_sources(tmp_path)
    assert [p.name for p in sources] == ["a.mp4", "b.mkv"]
    assert expected_final(tmp_path / "b.mkv").name == "b_es.srt"


def test_cli_entrypoint() -> None:
    """main() existe y --help funciona; script subtrans registrado."""
    from app import app as typer_app
    from app import main

    assert callable(main)
    result = runner.invoke(typer_app, ["--help"])
    assert result.exit_code == 0
    assert "auto" in result.output and "translate" in result.output
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    assert 'subtrans = "app:app"' in pyproject.read_text(encoding="utf-8")
