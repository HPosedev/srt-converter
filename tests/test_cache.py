"""Tests de cache.TranslationCache (JSON por chunk_index, sin red)."""

from pathlib import Path

import pytest

from cache import CACHE_SUFFIX, TranslationCache, cache_path_for


def test_cache_path_suffix(tmp_path: Path) -> None:
    """El temporal vive junto al SRT con el sufijo acordado."""
    src = tmp_path / "cap01.srt"
    assert cache_path_for(src).name == "cap01" + CACHE_SUFFIX
    assert cache_path_for(src).parent == tmp_path


def test_set_get_roundtrip(tmp_path: Path) -> None:
    """Lo guardado por chunk_index se recupera intacto."""
    cache = TranslationCache(tmp_path / "a.srt")
    assert cache.get(0) is None  # sin archivo → pendiente
    cache.set(0, ["Hola", "¿Qué tal?"])
    cache.set(2, ["Adiós"])
    assert cache.get(0) == ["Hola", "¿Qué tal?"]
    assert cache.get(1) is None
    assert cache.get(2) == ["Adiós"]


def test_pending_and_is_complete(tmp_path: Path) -> None:
    """pending() lista ausentes; is_complete() solo cuando están todos."""
    cache = TranslationCache(tmp_path / "a.srt")
    assert cache.pending(3) == [0, 1, 2]
    assert not cache.is_complete(3)
    cache.set(0, ["a"])
    cache.set(2, ["c"])
    assert cache.pending(3) == [1]
    cache.set(1, ["b"])
    assert cache.pending(3) == []
    assert cache.is_complete(3)
    assert cache.is_complete(0) is True  # cero chunks → nada pendiente


def test_clear_removes_file(tmp_path: Path) -> None:
    """clear() borra el JSON y devuelve si existía."""
    cache = TranslationCache(tmp_path / "a.srt")
    assert cache.clear() is False
    cache.set(0, ["x"])
    assert cache.path.is_file()
    assert cache.clear() is True
    assert not cache.path.is_file()


def test_corrupt_json_raises(tmp_path: Path) -> None:
    """JSON roto o con esquema inválido → ValueError claro."""
    cache = TranslationCache(tmp_path / "a.srt")
    cache.path.write_text("{no json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupta"):
        cache.load()
    cache.path.write_text('["no", "es un objeto"]', encoding="utf-8")
    with pytest.raises(ValueError, match="corrupta"):
        cache.load()
    cache.path.write_text('{"0": "debería ser lista"}', encoding="utf-8")
    with pytest.raises(ValueError, match="corrupta"):
        cache.load()


def test_resume_only_translates_missing_chunks(tmp_path: Path) -> None:
    """translate_file reutiliza chunks cacheados sin llamar a Gemini."""
    from srt_utils import parse_srt_content, replace_contents
    from translator import translate_file

    six = "".join(
        f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},500\nLine {i}\n\n"
        for i in range(1, 7)
    )
    src = tmp_path / "cap.srt"
    src.write_text(six, encoding="utf-8")

    class FakeSubtitler:
        """Solo traduce ventanas no cacheadas, contando llamadas."""

        window_size = 2
        calls = 0

        def _translate_window(self, window: list) -> list:
            type(self).calls += 1
            return replace_contents(window, [f"T{s.index}" for s in window])

    subs = parse_srt_content(six)
    first_window = subs[:2]
    TranslationCache(src).set(0, [f"T{s.index}" for s in first_window])

    FakeSubtitler.calls = 0
    dst, _ = translate_file(src, None, FakeSubtitler())  # type: ignore[arg-type]
    assert FakeSubtitler.calls == 2  # chunks 1 y 2; el 0 salió de caché
    assert "T1" in dst.read_text(encoding="utf-8")
    # Al completar, la caché se limpia.
    assert not TranslationCache(src).path.is_file()


def test_inconsistent_cache_entry_raises(tmp_path: Path) -> None:
    """Si la caché no encaja con la ventana → ValueError (no corromper)."""
    from translator import translate_file

    src = tmp_path / "cap.srt"
    src.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nHi\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nHo\n",
        encoding="utf-8",
    )
    TranslationCache(src).set(0, ["solo-uno"])  # ventana real tiene 2 bloques

    class FakeSubtitler:
        """No debería ni llamarse."""

        window_size = 50

        def _translate_window(self, window: list) -> list:  # pragma: no cover
            raise AssertionError("no debería llamarse")

    with pytest.raises(ValueError, match="inconsistente"):
        translate_file(src, None, FakeSubtitler())  # type: ignore[arg-type]
