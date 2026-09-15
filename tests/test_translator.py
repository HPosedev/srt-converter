"""Tests de translator.GeminiSubtitler con cliente GenAI mockeado (sin red)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from srt_utils import parse_srt_content
from translator import GeminiSubtitler, translate_file

SAMPLE = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:05,000
How are you?
"""


class FakeResponse:
    """Imita GenerateContentResponse (solo se usa `.text`)."""

    def __init__(self, text: str) -> None:
        """Guarda el texto generado."""
        self.text = text


class FakeModels:
    """Cola de respuestas programadas + registro de llamadas."""

    def __init__(self, script: list[str]) -> None:
        """Inicializa con las respuestas a devolver en orden."""
        self.script = list(script)
        self.calls: list[dict] = []

    def generate_content(
        self, *, model: str, contents: object, config: object = None
    ) -> FakeResponse:
        """Devuelve la siguiente respuesta programada."""
        self.calls.append({"model": model, "contents": contents, "config": config})
        return FakeResponse(self.script.pop(0))


class FakeFiles:
    """Imita la Files API: upload/delete con registro."""

    def __init__(self) -> None:
        """Inicializa registros vacíos."""
        self.uploaded: list[str] = []
        self.deleted: list[str] = []

    def upload(self, *, file: str) -> SimpleNamespace:
        """Registra la subida y devuelve un archivo remoto ficticio."""
        self.uploaded.append(file)
        return SimpleNamespace(name="files/fake123", uri="https://fake/abc")

    def delete(self, *, name: str) -> None:
        """Registra el borrado remoto."""
        self.deleted.append(name)


class FakeClient:
    """Cliente GenAI falso (models + files)."""

    def __init__(self, script: list[str]) -> None:
        """Crea models con guion y files vacío."""
        self.models = FakeModels(script)
        self.files = FakeFiles()


def _subtitler(script: list[str], **kwargs: object) -> tuple[GeminiSubtitler, FakeClient]:
    """Crea GeminiSubtitler con FakeClient y devuelve ambos."""
    client = FakeClient(script)
    sub = GeminiSubtitler(api_key="test-key", client=client, **kwargs)  # type: ignore[arg-type]
    return sub, client


def test_translate_srt_text_preserves_timestamps() -> None:
    """El texto cambia pero start/end/index quedan intactos."""
    subs = parse_srt_content(SAMPLE)
    sub, client = _subtitler(["[#1] Hola mundo\n[#2] ¿Cómo estás?"])
    out = sub.translate_srt_text(subs)
    assert [s.content for s in out] == ["Hola mundo", "¿Cómo estás?"]
    assert [(s.start, s.end, s.index) for s in out] == [
        (s.start, s.end, s.index) for s in subs
    ]
    assert len(client.models.calls) == 1  # una ventana → una llamada


def test_translate_windows_split_calls() -> None:
    """6 bloques con window_size=2 → 3 llamadas a la API."""
    six = "".join(
        f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},500\nLine {i}\n\n"
        for i in range(1, 7)
    )
    subs = parse_srt_content(six)
    script = [
        "[#1] T1\n[#2] T2",
        "[#3] T3\n[#4] T4",
        "[#5] T5\n[#6] T6",
    ]
    sub, client = _subtitler(script, window_size=2)
    out = sub.translate_srt_text(subs)
    assert len(out) == 6
    assert len(client.models.calls) == 3
    assert out[4].content == "T5"


def test_translate_empty_returns_empty_without_calls() -> None:
    """Entrada vacía → sin llamadas a la API."""
    sub, client = _subtitler([])
    assert sub.translate_srt_text([]) == []
    assert client.models.calls == []


def test_translate_mismatched_response_halves_window() -> None:
    """Descuadre 1:1 → reintento estricto y luego división a la mitad.

    Comportamiento resiliente (Sprint 4): la ventana de 2 bloques falla
    dos veces y se resuelve traduciendo cada mitad por separado.
    """
    subs = parse_srt_content(SAMPLE)
    sub, client = _subtitler(
        ["[#1] Solo uno", "[#1] Solo uno", "[#1] Uno", "[#2] Dos"]
    )
    out = sub.translate_srt_text(subs)
    assert [s.content for s in out] == ["Uno", "Dos"]
    assert len(client.models.calls) == 4
    strict_call = client.models.calls[1]
    assert strict_call["config"] == {"temperature": 0.0}
    assert "RECORDATORIO ESTRICTO" in strict_call["contents"]
    assert sub.fallback_notices == []


def test_translate_single_line_failure_keeps_original() -> None:
    """Una línea aislada que sigue fallando conserva el original con aviso."""
    subs = parse_srt_content(SAMPLE)
    sub, _ = _subtitler(
        ["[#1] Algo\n[#2]   ", "[#1] Algo\n[#2]", "[#1] Algo", "[#9] Nope", "[#2]  "]
    )
    out = sub.translate_srt_text(subs)
    assert out[0].content == "Algo"
    assert out[1].content == "How are you?"  # original conservado
    assert len(sub.fallback_notices) == 1
    assert "#2" in sub.fallback_notices[0]


def test_translate_strict_mode_raises_on_mismatch() -> None:
    """Con resilient=False el descuadre lanza TranslationMismatchError."""
    from translator import TranslationMismatchError

    subs = parse_srt_content(SAMPLE)
    sub, _ = _subtitler(["[#1] Solo uno", "[#1] Solo uno"])
    with pytest.raises(TranslationMismatchError, match="descuadrada"):
        sub.translate_srt_text(subs, resilient=False)
    with pytest.raises(RuntimeError, match="descuadrada"):
        sub.translate_srt_text(parse_srt_content(SAMPLE), resilient=False)


def test_glossary_injected_in_prompt() -> None:
    """El glosario aparece en el prompt enviado al modelo."""
    subs = parse_srt_content(SAMPLE)
    sub, client = _subtitler(
        ["[#1] Hola mundo\n[#2] ¿Cómo estás?"],
        glossary={"Hello": "Hola"},
    )
    sub.translate_srt_text(subs)
    prompt = client.models.calls[0]["contents"]
    assert isinstance(prompt, str) and "Hola" in prompt


def test_transcribe_uploads_and_cleans_up(tmp_path: Path) -> None:
    """Sube el MP3, parsea el SRT devuelto y borra el remoto."""
    audio = tmp_path / "track.mp3"
    audio.write_bytes(b"fake-mp3")
    srt_text = "1\n00:00:01,000 --> 00:00:02,000\nHola\n"
    sub, client = _subtitler([srt_text])
    out = sub.transcribe_and_translate_audio(audio)
    assert len(out) == 1 and out[0].content == "Hola"
    assert client.files.uploaded == [str(audio)]
    assert client.files.deleted == ["files/fake123"]


def test_transcribe_deletes_remote_even_on_api_error(tmp_path: Path) -> None:
    """Si generate_content falla, el archivo remoto se borra igual."""

    class ExplodingModels(FakeModels):
        """Falla siempre al generar."""

        def generate_content(self, *, model: str, contents: object) -> FakeResponse:
            """Registra y lanza."""
            self.calls.append({"model": model, "contents": contents})
            raise RuntimeError("boom")

    audio = tmp_path / "t.mp3"
    audio.write_bytes(b"x")
    client = FakeClient([])
    client.models = ExplodingModels([])
    sub = GeminiSubtitler(api_key="k", client=client)
    with pytest.raises(RuntimeError, match="boom"):
        sub.transcribe_and_translate_audio(audio)
    assert client.files.deleted == ["files/fake123"]


def test_transcribe_invalid_srt_raises_but_cleans(tmp_path: Path) -> None:
    """SRT inválido del modelo → ValueError, pero con limpieza remota."""
    audio = tmp_path / "t.mp3"
    audio.write_bytes(b"x")
    sub, client = _subtitler(["esto no es un srt"])
    with pytest.raises(ValueError, match="no devolvió un SRT válido"):
        sub.transcribe_and_translate_audio(audio)
    assert client.files.deleted == ["files/fake123"]


def test_transcribe_missing_audio_raises(tmp_path: Path) -> None:
    """Audio inexistente → FileNotFoundError sin llamadas a la API."""
    sub, client = _subtitler(["x"])
    with pytest.raises(FileNotFoundError):
        sub.transcribe_and_translate_audio(tmp_path / "no.mp3")
    assert client.files.uploaded == []


def test_translate_file_writes_es_suffix(tmp_path: Path) -> None:
    """translate_file genera *_es.srt sin tocar el original."""
    src = tmp_path / "input.srt"
    src.write_text(SAMPLE, encoding="utf-8")
    sub, _ = _subtitler(["[#1] Hola mundo\n[#2] ¿Cómo estás?"])
    dst, warnings = translate_file(src, None, sub)
    assert dst.name == "input_es.srt"
    assert dst.read_text(encoding="utf-8").startswith("1\n00:00:01,000")
    assert src.read_text(encoding="utf-8") == SAMPLE  # original intacto
    assert isinstance(warnings, list)


def test_translate_file_refuses_overwriting_original(tmp_path: Path) -> None:
    """Destino == origen → ValueError (nunca sobrescribir)."""
    src = tmp_path / "a.srt"
    src.write_text(SAMPLE, encoding="utf-8")
    sub, _ = _subtitler([])
    with pytest.raises(ValueError, match="coincide con el original"):
        translate_file(src, src, sub)


def test_missing_api_key_without_client_raises() -> None:
    """Sin api_key ni client → ValueError inmediato."""
    with pytest.raises(ValueError, match="api_key"):
        GeminiSubtitler(api_key="")
