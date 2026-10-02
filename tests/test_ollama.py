"""Tests de translator.OllamaSubtitler con cliente HTTP falso (sin red)."""

import httpx
import pytest

from srt_utils import parse_srt_content
from translator import OllamaSubtitler

SAMPLE = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:05,000
How are you?
"""


class FakeHttp:
    """Imita ``httpx.Client.post`` con respuestas programadas."""

    def __init__(self, script: list) -> None:
        """Inicializa con respuestas (str), códigos (int) o excepciones."""
        self.script = list(script)
        self.calls: list[dict] = []

    def post(self, path: str, *, json: dict) -> httpx.Response:
        """Registra la llamada y devuelve o lanza el siguiente elemento."""
        self.calls.append({"path": path, "json": json})
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        request = httpx.Request("POST", f"http://ollama{path}")
        if isinstance(item, int):
            return httpx.Response(item, json={"error": "x"}, request=request)
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": item}}, request=request
        )


def _subtitler(script: list, **kwargs: object) -> tuple[OllamaSubtitler, FakeHttp]:
    """Crea OllamaSubtitler con FakeHttp y devuelve ambos."""
    http = FakeHttp(script)
    return OllamaSubtitler(http=http, **kwargs), http  # type: ignore[arg-type]


def test_translate_preserves_timestamps_and_sends_chat_payload() -> None:
    """Traduce vía /api/chat sin streaming ni thinking y con num_ctx."""
    subs = parse_srt_content(SAMPLE)
    sub, http = _subtitler(
        ["[#1] Hola mundo\n[#2] ¿Cómo estás?"], model="gemma4:26b-a4b", num_ctx=4096
    )
    out = sub.translate_srt_text(subs)
    assert [s.content for s in out] == ["Hola mundo", "¿Cómo estás?"]
    assert [(s.start, s.end) for s in out] == [(s.start, s.end) for s in subs]
    call = http.calls[0]
    assert call["path"] == "/api/chat"
    assert call["json"]["model"] == "gemma4:26b-a4b"
    assert call["json"]["stream"] is False
    assert call["json"]["think"] is False
    assert call["json"]["options"] == {"temperature": 0.3, "num_ctx": 4096}
    assert "[#1] Hello world" in call["json"]["messages"][0]["content"]


def test_strict_retry_uses_zero_temperature() -> None:
    """Tras un descuadre, el reintento estricto fuerza temperature=0.0."""
    subs = parse_srt_content(SAMPLE)
    sub, http = _subtitler(["[#1] Solo uno", "[#1] Hola\n[#2] Adiós"])
    out = sub.translate_srt_text(subs)
    assert [s.content for s in out] == ["Hola", "Adiós"]
    assert http.calls[1]["json"]["options"]["temperature"] == 0.0


def test_missing_model_explains_pull() -> None:
    """404 de Ollama → ValueError con el comando ollama pull."""
    subs = parse_srt_content(SAMPLE)
    sub, _ = _subtitler([404], model="nope:1b")
    with pytest.raises(ValueError, match="ollama pull nope:1b"):
        sub.translate_srt_text(subs)


def test_server_down_raises_connection_error_without_retry() -> None:
    """Servidor caído → ConnectionError claro, sin reintentos."""
    subs = parse_srt_content(SAMPLE)
    sub, http = _subtitler([httpx.ConnectError("Connection refused")])
    with pytest.raises(ConnectionError, match="ollama serve"):
        sub.translate_srt_text(subs)
    assert len(http.calls) == 1


def test_server_error_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un 500 se trata como transitorio y se reintenta con backoff."""
    import translator

    sleeps: list[float] = []
    monkeypatch.setattr(translator.time, "sleep", lambda s: sleeps.append(s))
    subs = parse_srt_content(SAMPLE)
    sub, http = _subtitler([500, "[#1] Hola\n[#2] Adiós"])
    out = sub.translate_srt_text(subs)
    assert [s.content for s in out] == ["Hola", "Adiós"]
    assert sleeps == [2.0]


def test_audio_transcription_not_supported() -> None:
    """El backend local no transcribe audio."""
    sub, _ = _subtitler([])
    with pytest.raises(NotImplementedError, match="gemini"):
        sub.transcribe_and_translate_audio("x.mp3")


def test_line_separator_becomes_real_line_break() -> None:
    """El " / " que el modelo copia del prompt vuelve a ser salto de línea."""
    subs = parse_srt_content(
        "1\n00:00:01,000 --> 00:00:02,000\n[Richie]\nMy boy didn't come home.\n"
    )
    sub, _ = _subtitler(["[#1] [Richie] / Mi hijo no ha vuelto a casa."])
    out = sub.translate_srt_text(subs)
    assert out[0].content == "[Richie]\nMi hijo no ha vuelto a casa."
