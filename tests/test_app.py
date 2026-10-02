"""Tests de la CLI Typer (extract/translate/auto) con traducción mockeada."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import app
from app import build_subtitler
from extractor import SubtitleTrack

runner = CliRunner()

SAMPLE = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:05,000
Bye
"""


class DummySubtitler:
    """Traductor falso: antepone 'ES:' sin red."""

    window_size = 50

    def __init__(self) -> None:
        """Sin estado."""
        self.calls = 0

    def _translate_window(self, window: list) -> list:
        """Traduce la ventana entera de una vez."""
        from srt_utils import replace_contents

        self.calls += 1
        return replace_contents(window, [f"ES:{s.content}" for s in window])


@pytest.fixture()
def srt_file(tmp_path: Path) -> Path:
    """Crea un SRT inglés de ejemplo."""
    src = tmp_path / "cap01.srt"
    src.write_text(SAMPLE, encoding="utf-8")
    return src


def _patch_subtitler(monkeypatch: pytest.MonkeyPatch) -> DummySubtitler:
    """Evita leer config real ni llamar a Gemini."""
    from config import AppConfig, GeminiConfig

    dummy = DummySubtitler()
    monkeypatch.setattr(app, "build_subtitler", lambda cfg, window_size=None: dummy)
    monkeypatch.setattr(
        app, "load_config", lambda path=None: AppConfig(gemini=GeminiConfig(api_key="x"))
    )
    return dummy


def test_build_subtitler_requires_api_key() -> None:
    """Sin API key → BadParameter con pista de cómo configurarla."""
    import typer

    from config import AppConfig

    with pytest.raises(typer.BadParameter, match="API key"):
        build_subtitler(AppConfig())


def test_translate_single_writes_es_and_audits(
    monkeypatch: pytest.MonkeyPatch, srt_file: Path
) -> None:
    """translate genera *_es.srt y muestra avisos de legibilidad."""
    _patch_subtitler(monkeypatch)
    result = runner.invoke(app.app, ["translate", str(srt_file)])
    assert result.exit_code == 0, result.output
    out = srt_file.with_name("cap01_es.srt")
    assert out.is_file()
    assert "ES:Hello world" in out.read_text(encoding="utf-8")
    assert "Traducido" in result.output


def test_translate_reports_long_line_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Una traducción >42 chars aparece como aviso (sin modificar texto)."""
    long_src = tmp_path / "long.srt"
    long_src.write_text(
        "1\n00:00:01,000 --> 00:00:05,000\nHi\n", encoding="utf-8"
    )
    dummy = _patch_subtitler(monkeypatch)

    def _long_window(window: list) -> list:
        from srt_utils import replace_contents

        return replace_contents(window, ["x" * 50])

    monkeypatch.setattr(dummy, "_translate_window", _long_window)
    result = runner.invoke(app.app, ["translate", str(long_src)])
    assert result.exit_code == 0, result.output
    assert "aviso" in result.output.lower()
    assert "x" * 50 in long_src.with_name("long_es.srt").read_text(encoding="utf-8")


def test_translate_never_overwrites_original(
    monkeypatch: pytest.MonkeyPatch, srt_file: Path
) -> None:
    """Destino == origen → error, original intacto."""
    _patch_subtitler(monkeypatch)
    result = runner.invoke(
        app.app, ["translate", str(srt_file), "--output", str(srt_file)]
    )
    assert result.exit_code != 0
    assert srt_file.read_text(encoding="utf-8") == SAMPLE


def test_translate_batch_skips_es_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--batch traduce cada SRT salvo los *_es.srt ya generados."""
    _patch_subtitler(monkeypatch)
    (tmp_path / "a.srt").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "b.srt").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "a_es.srt").write_text("ya traducido", encoding="utf-8")
    result = runner.invoke(app.app, ["translate", str(tmp_path), "--batch"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "b_es.srt").is_file()
    assert (tmp_path / "a_es.srt").read_text(encoding="utf-8") != "ya traducido"


def test_translate_batch_empty_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Carpeta sin SRT → mensaje y salida 0."""
    _patch_subtitler(monkeypatch)
    result = runner.invoke(app.app, ["translate", str(tmp_path), "--batch"])
    assert result.exit_code == 0
    assert "Sin archivos" in result.output


def test_extract_list_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """extract --list muestra pistas sin extraer nada."""
    tracks = [SubtitleTrack(2, 0, "eng", "Full", "subrip", False, True)]
    monkeypatch.setattr(app, "list_subtitle_tracks", lambda mkv: tracks)
    mkv = tmp_path / "v.mkv"
    mkv.write_bytes(b"x")
    result = runner.invoke(app.app, ["extract", str(mkv), "--list"])
    assert result.exit_code == 0, result.output
    assert "completos" in result.output


def test_auto_from_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """auto --from-audio: extrae MP3, transcribe y audita."""
    import srt as srt_lib

    from srt_utils import parse_srt_content

    _patch_subtitler(monkeypatch)
    mkv = tmp_path / "epi.mkv"
    mkv.write_bytes(b"x")
    mp3_seen: list[Path] = []

    from extractor import AudioTrack

    fake_audio = [AudioTrack(1, 0, "eng", "", "aac", 2, True)]
    monkeypatch.setattr(app, "list_audio_tracks", lambda m: fake_audio)

    def fake_extract_audio(
        mkv_path: Path, out: Path, overwrite: bool = True, track: object = None
    ) -> Path:
        mp3_seen.append(Path(out))
        Path(out).write_bytes(b"mp3")
        return Path(out)

    monkeypatch.setattr(app, "extract_audio", fake_extract_audio)
    dummy_transcribed = parse_srt_content(SAMPLE)
    got: dict = {}

    def fake_transcribe(audio: Path) -> list:
        got["audio"] = Path(audio)
        return [srt_lib.Subtitle(s.index, s.start, s.end, f"ES:{s.content}") for s in dummy_transcribed]

    def fake_build(cfg: object, window_size: int | None = None) -> object:
        from types import SimpleNamespace

        return SimpleNamespace(transcribe_and_translate_audio=fake_transcribe)

    monkeypatch.setattr(app, "build_subtitler", fake_build)
    result = runner.invoke(app.app, ["auto", str(mkv), "--from-audio"])
    assert result.exit_code == 0, result.output
    final = tmp_path / "epi_es.srt"
    assert final.is_file()
    assert got["audio"] == final.with_name("epi_es.auto-audio.mp3")
    assert not got["audio"].is_file()  # intermedio borrado por defecto
    assert "Traducido" in result.output


def test_auto_subtitle_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """auto vía subtítulo: extrae intermedio, traduce y lo borra."""
    _patch_subtitler(monkeypatch)
    mkv = tmp_path / "epi.mkv"
    mkv.write_bytes(b"x")
    tracks = [SubtitleTrack(2, 0, "eng", "Full", "subrip", False, True)]
    monkeypatch.setattr(app, "list_subtitle_tracks", lambda m: tracks)
    monkeypatch.setattr(
        app, "select_track", lambda tracks, lang=None, sub_index=None: tracks[0]
    )

    def fake_extract(mkv_path: Path, out: Path, track: object, overwrite: bool) -> Path:
        Path(out).write_text(SAMPLE, encoding="utf-8")
        return Path(out)

    monkeypatch.setattr(app, "extract_subtitle", fake_extract)
    result = runner.invoke(app.app, ["auto", str(mkv)])
    assert result.exit_code == 0, result.output
    final = tmp_path / "epi_es.srt"
    assert final.is_file()
    assert "ES:Hello world" in final.read_text(encoding="utf-8")
    assert not tmp_path.joinpath("epi_es.extracted.srt").is_file()


def test_auto_no_tracks_suggests_from_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sin pistas de subtítulo → error que sugiere --from-audio."""
    _patch_subtitler(monkeypatch)
    monkeypatch.setattr(app, "list_subtitle_tracks", lambda m: [])
    mkv = tmp_path / "v.mkv"
    mkv.write_bytes(b"x")
    result = runner.invoke(app.app, ["auto", str(mkv)])
    assert result.exit_code != 0
    assert "--from-audio" in result.output


def test_build_subtitler_ollama_needs_no_api_key() -> None:
    """backend = "ollama" construye OllamaSubtitler sin API key."""
    from config import AppConfig, OllamaConfig
    from translator import OllamaSubtitler

    cfg = AppConfig(backend="ollama", ollama=OllamaConfig(model="m:1b", num_ctx=4096))
    sub = build_subtitler(cfg, window_size=20)
    assert isinstance(sub, OllamaSubtitler)
    assert (sub.model, sub.num_ctx, sub.window_size) == ("m:1b", 4096, 20)


def test_auto_from_audio_rejected_with_ollama(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--from-audio con backend ollama falla antes de extraer nada."""
    from config import AppConfig

    monkeypatch.setattr(app, "load_config", lambda path=None: AppConfig(backend="ollama"))
    monkeypatch.setattr(
        app, "extract_audio", lambda *a, **k: pytest.fail("no debe extraer audio")
    )
    mkv = tmp_path / "v.mkv"
    mkv.write_bytes(b"x")
    result = runner.invoke(app.app, ["auto", str(mkv), "--from-audio"])
    assert result.exit_code != 0
    assert "gemini" in result.output


def test_report_warnings_keeps_bracketed_sound_tags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Las acotaciones [grita] no se pierden como markup de Rich."""
    from types import SimpleNamespace

    warning = SimpleNamespace(
        index=16, kind="cps", value=24.3, limit=17.0, detail="-[grita] / -Guerra."
    )
    app.report_warnings([warning])
    assert "[grita]" in capsys.readouterr().err


def test_start_backend_reports_missing_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si Ollama no se puede arrancar, la CLI sale con un mensaje claro."""
    import typer

    from translator import OllamaSubtitler

    sub = OllamaSubtitler(http=object())
    monkeypatch.setattr(sub, "server_running", lambda: False)

    def no_binary() -> None:
        raise FileNotFoundError("No se encontró el programa 'ollama'.")

    monkeypatch.setattr(sub, "start_server", no_binary)
    with pytest.raises(typer.Exit):
        app.start_backend(sub)
