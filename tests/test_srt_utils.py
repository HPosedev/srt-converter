"""Tests unitarios de srt_utils (encoding, roundtrip, auditoría, chunking)."""

from datetime import timedelta
from pathlib import Path

import pytest
import srt

from srt_utils import (
    audit_subtitles,
    chunk_subtitles,
    compose_srt_content,
    detect_encoding,
    normalize_text,
    parse_srt_content,
    read_srt_file,
    replace_contents,
    write_srt_file,
)

SAMPLE = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:05,000
Second line
with two rows
"""


def test_normalize_strips_bom_and_crlf() -> None:
    """El BOM y los CRLF se normalizan antes de parsear."""
    raw = "\ufeff1\r\n00:00:01,000 --> 00:00:03,000\r\nHi\r\n"
    assert normalize_text(raw).startswith("1\n00:00:01")


def test_normalize_strips_markdown_code_fences() -> None:
    """Los bloques de código markdown generados por LLMs se limpian."""
    fenced = "```srt\n1\n00:00:01,000 --> 00:00:03,000\nHola\n```"
    subs = parse_srt_content(fenced)
    assert len(subs) == 1
    assert subs[0].content == "Hola"


def test_parse_and_compose_roundtrip_preserves_timestamps() -> None:
    """Parsear y recomponer conserva índices y tiempos exactos."""
    subs = parse_srt_content(SAMPLE)
    assert len(subs) == 2
    assert subs[0].start == timedelta(seconds=1)
    assert subs[0].end == timedelta(seconds=3)
    recomposed = compose_srt_content(subs)
    again = parse_srt_content(recomposed)
    assert [(s.start, s.end) for s in again] == [(s.start, s.end) for s in subs]


def test_replace_contents_keeps_timestamps() -> None:
    """Solo cambia el texto; start/end/index quedan intactos."""
    subs = parse_srt_content(SAMPLE)
    translated = replace_contents(subs, ["Hola mundo", "Segunda línea\ncon dos filas"])
    assert [s.content for s in translated] == ["Hola mundo", "Segunda línea\ncon dos filas"]
    assert [(s.start, s.end, s.index) for s in translated] == [
        (s.start, s.end, s.index) for s in subs
    ]
    # El original no se muta.
    assert subs[0].content == "Hello world"


def test_replace_contents_rejects_length_mismatch() -> None:
    """Longitudes distintas rompen la garantía 1:1 y deben fallar."""
    subs = parse_srt_content(SAMPLE)
    with pytest.raises(ValueError, match="1:1"):
        replace_contents(subs, ["solo uno"])


def test_audit_detects_long_line_and_fast_cps() -> None:
    """Una línea >42 chars y un bloque rapidísimo generan avisos."""
    subs = parse_srt_content(
        "1\n00:00:01,000 --> 00:00:02,000\n"
        "Esta línea es deliberadamente mucho más larga que cuarenta y dos caracteres\n"
    )
    findings = audit_subtitles(subs)
    kinds = {w.kind for w in findings}
    assert "max_chars" in kinds
    assert "cps" in kinds  # ~76 chars en 1s >> 17 cps


def test_audit_clean_subtitle_has_no_warnings() -> None:
    """Un bloque corto y con tiempo suficiente no avisa."""
    subs = parse_srt_content("1\n00:00:01,000 --> 00:00:05,000\nHi\n")
    assert audit_subtitles(subs) == []


def test_chunking_window_size_validated() -> None:
    """Ventana por defecto 50 (Sprint 3); rango válido 1-200."""
    from srt_utils import DEFAULT_WINDOW_SIZE

    assert DEFAULT_WINDOW_SIZE == 50
    subs = parse_srt_content(SAMPLE) * 6  # 12 bloques
    chunks = chunk_subtitles(subs)
    assert [len(c) for c in chunks] == [12]  # cabe en una ventana de 50
    chunks_small = chunk_subtitles(subs, window_size=5)
    assert [len(c) for c in chunks_small] == [5, 5, 2]
    with pytest.raises(ValueError, match="entre 1 y 200"):
        chunk_subtitles(subs, window_size=0)
    with pytest.raises(ValueError, match="entre 1 y 200"):
        chunk_subtitles(subs, window_size=201)


def test_detect_encoding_utf8_sig_latin1(tmp_path: Path) -> None:
    """BOM → utf-8-sig; eñes en latin-1 → latin-1; ascii → utf-8."""
    bom = tmp_path / "bom.srt"
    bom.write_bytes(b"\xef\xbb\xbf" + SAMPLE.encode("utf-8"))
    assert detect_encoding(bom) == "utf-8-sig"

    latin = tmp_path / "latin.srt"
    latin.write_bytes("1\n00:00:01,000 --> 00:00:02,000\ncanci\xf3n\n".encode("latin-1"))
    assert detect_encoding(latin) == "latin-1"
    subs = read_srt_file(latin)
    assert subs[0].content == "canción"


def test_write_read_roundtrip(tmp_path: Path) -> None:
    """Escribir y releer conserva bloques y crea directorios padre."""
    subs = parse_srt_content(SAMPLE)
    dest = tmp_path / "sub" / "out.srt"
    write_srt_file(dest, subs)
    assert read_srt_file(dest)[1].content == "Second line\nwith two rows"


def test_parse_invalid_raises_value_error() -> None:
    """Contenido no-SRT produce ValueError claro (no SRTParseError crudo)."""
    with pytest.raises(ValueError, match="SRT inválido"):
        parse_srt_content("esto no es un srt")


def test_compose_empty_returns_empty() -> None:
    """Componer lista vacía no genera basura."""
    assert compose_srt_content([]) == ""


def test_cps_zero_duration_with_text_warns() -> None:
    """Duración 0 con texto → cps infinito → aviso."""
    sub = srt.Subtitle(
        index=1, start=timedelta(seconds=2), end=timedelta(seconds=2), content="Hi"
    )
    assert any(w.kind == "cps" for w in audit_subtitles([sub]))
