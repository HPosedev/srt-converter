"""Tests de resiliencia: backoff 429 y chunk halving (GenAI mockeado)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import translator
from srt_utils import parse_srt_content
from translator import GeminiSubtitler, TranslationMismatchError, translate_file

SAMPLE4 = "".join(
    f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},500\nLine {i}\n\n" for i in range(1, 5)
)


class Fake429(Exception):
    """Simula rate limit HTTP 429 con atributo de estado."""

    def __init__(self) -> None:
        """Guarda status 429."""
        super().__init__("429 Too Many Requests")
        self.status_code = 429


class ResourceExhausted(Exception):
    """Simula error de cuota solo por nombre (sin atributos)."""


class FakeResponse:
    """Imita GenerateContentResponse (solo `.text`)."""

    def __init__(self, text: str) -> None:
        """Guarda el texto."""
        self.text = text


class FakeModels:
    """Guion mixto de respuestas y excepciones + registro de llamadas."""

    def __init__(self, script: list) -> None:
        """Inicializa la cola de respuestas/excepciones."""
        self.script = list(script)
        self.calls: list[dict] = []

    def generate_content(
        self, *, model: str, contents: object, config: object = None
    ) -> FakeResponse:
        """Devuelve o lanza el siguiente elemento del guion."""
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return FakeResponse(item)


class FakeFiles:
    """Files API mínima (upload/delete con registro)."""

    def __init__(self) -> None:
        """Inicializa registros."""
        self.deleted: list[str] = []

    def upload(self, *, file: str) -> SimpleNamespace:
        """Devuelve archivo remoto ficticio."""
        return SimpleNamespace(name="files/x", uri="https://fake/x")

    def delete(self, *, name: str) -> None:
        """Registra el borrado."""
        self.deleted.append(name)


class FakeClient:
    """Cliente falso con models programable."""

    def __init__(self, script: list) -> None:
        """Crea models con guion y files vacío."""
        self.models = FakeModels(script)
        self.files = FakeFiles()


def _subtitler(script: list, **kwargs: object) -> tuple[GeminiSubtitler, FakeClient]:
    """Crea GeminiSubtitler con FakeClient."""
    client = FakeClient(script)
    return GeminiSubtitler(api_key="k", client=client, **kwargs), client  # type: ignore[arg-type]


def test_backoff_retries_429_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dos 429 → esperas 2s y 4s → éxito al tercer intento."""
    sleeps: list[float] = []
    monkeypatch.setattr(translator.time, "sleep", lambda s: sleeps.append(s))
    subs = parse_srt_content("1\n00:00:01,000 --> 00:00:02,000\nHi\n")
    sub, client = _subtitler([Fake429(), Fake429(), "[#1] Hola"])
    out = sub.translate_srt_text(subs)
    assert [s.content for s in out] == ["Hola"]
    assert sleeps == [2.0, 4.0]
    assert len(client.models.calls) == 3


def test_backoff_gives_up_after_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """429 persistente → 1 intento + 3 reintentos y propaga el error."""
    sleeps: list[float] = []
    monkeypatch.setattr(translator.time, "sleep", lambda s: sleeps.append(s))
    subs = parse_srt_content("1\n00:00:01,000 --> 00:00:02,000\nHi\n")
    sub, client = _subtitler([Fake429()] * 5)
    with pytest.raises(Fake429):
        sub.translate_srt_text(subs)
    assert len(client.models.calls) == 4
    assert sleeps == [2.0, 4.0, 8.0]


def test_backoff_retries_resource_exhausted_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Error de cuota detectado por nombre también reintenta."""
    sleeps: list[float] = []
    monkeypatch.setattr(translator.time, "sleep", lambda s: sleeps.append(s))
    subs = parse_srt_content("1\n00:00:01,000 --> 00:00:02,000\nHi\n")
    sub, _ = _subtitler([ResourceExhausted("quota ResourceExhausted"), "[#1] Hola"])
    assert [s.content for s in sub.translate_srt_text(subs)] == ["Hola"]
    assert sleeps == [2.0]


def test_non_transient_error_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API key inválida → sin reintentos ni esperas."""
    sleeps: list[float] = []
    monkeypatch.setattr(translator.time, "sleep", lambda s: sleeps.append(s))
    subs = parse_srt_content("1\n00:00:01,000 --> 00:00:02,000\nHi\n")
    sub, client = _subtitler([ValueError("API key not valid")])
    with pytest.raises(ValueError, match="API key"):
        sub.translate_srt_text(subs)
    assert len(client.models.calls) == 1
    assert sleeps == []


def test_halving_splits_window_of_four() -> None:
    """Ventana de 4 que descuadra dos veces se resuelve por mitades."""
    subs = parse_srt_content(SAMPLE4)
    sub, client = _subtitler(
        [
            "[#1] A",  # intento completo: descuadre
            "[#1] A",  # reintento estricto: descuadre otra vez
            "[#1] T1\n[#2] T2",  # mitad izquierda ok
            "[#3] T3\n[#4] T4",  # mitad derecha ok
        ],
        window_size=50,
    )
    out = sub.translate_srt_text(subs)
    assert [s.content for s in out] == ["T1", "T2", "T3", "T4"]
    assert [(s.start, s.end) for s in out] == [(s.start, s.end) for s in subs]
    assert len(client.models.calls) == 4
    assert sub.fallback_notices == []


def test_mismatch_error_is_runtime_error() -> None:
    """TranslationMismatchError hereda RuntimeError (compat)."""
    assert issubclass(TranslationMismatchError, RuntimeError)


def test_style_and_glossary_preserved_in_halving() -> None:
    """Las subventanas del fallback conservan glosario y estilo."""
    subs = parse_srt_content(SAMPLE4)
    sub, client = _subtitler(
        [
            "[#1] A",  # intento completo: descuadre
            "[#1] A",  # estricto: descuadre otra vez
            "[#1] T1\n[#2] T2",
            "[#3] T3\n[#4] T4",
        ],
        glossary={"Winterfell": "Invernalia"},
        style_instructions="Tono cinematográfico",
    )
    out = sub.translate_srt_text(subs)
    assert [s.content for s in out] == ["T1", "T2", "T3", "T4"]
    assert len(client.models.calls) == 4
    for call in client.models.calls:
        prompt = call["contents"]
        assert isinstance(prompt, str)
        assert "Invernalia" in prompt  # glosario intacto
        assert "Tono cinematográfico" in prompt  # estilo intacto


def test_translate_file_recovers_with_cache_after_transient(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Chunk 0 cacheado tras corte; 2ª pasada solo pide el chunk 1."""
    sleeps: list[float] = []
    monkeypatch.setattr(translator.time, "sleep", lambda s: sleeps.append(s))
    six = "".join(
        f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},500\nLine {i}\n\n" for i in range(1, 5)
    )
    src = tmp_path / "cap.srt"
    src.write_text(six, encoding="utf-8")

    # 1ª pasada: chunk 0 ok, chunk 1 falla con 429 persistente → propaga.
    sub1, _ = _subtitler(
        ["[#1] T1\n[#2] T2", Fake429(), Fake429(), Fake429(), Fake429()],
        window_size=2,
    )
    with pytest.raises(Fake429):
        translate_file(src, None, sub1)
    from cache import TranslationCache

    assert TranslationCache(src).get(0) == ["T1", "T2"]
    assert TranslationCache(src).get(1) is None

    # 2ª pasada: solo se pide el chunk pendiente.
    sub2, client2 = _subtitler(["[#3] T3\n[#4] T4"], window_size=2)
    dst, _ = translate_file(src, None, sub2)
    assert len(client2.models.calls) == 1
    text = dst.read_text(encoding="utf-8")
    assert "T1" in text and "T4" in text
    assert not TranslationCache(src).path.is_file()  # completa → limpia
