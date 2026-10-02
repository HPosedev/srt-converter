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

    def __init__(self, script: list, alive: list[bool] | None = None) -> None:
        """Inicializa con respuestas (str), códigos (int) o excepciones.

        ``alive`` programa las respuestas de ``GET /api/version`` (la última
        se repite); por defecto el servidor responde siempre.
        """
        self.script = list(script)
        self.calls: list[dict] = []
        self.alive = list(alive) if alive is not None else [True]

    def get(self, path: str, *, timeout: float) -> httpx.Response:
        """Simula ``/api/version``: 200 si está vivo, ConnectError si no."""
        up = self.alive.pop(0) if len(self.alive) > 1 else self.alive[0]
        if not up:
            raise httpx.ConnectError("Connection refused")
        return httpx.Response(200, json={"version": "0.33.3"})

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


class FakeProc:
    """Imita ``subprocess.Popen`` de ``ollama serve``."""

    def __init__(self, args: list, *, env: dict, exit_code: int | None = None, **_: object) -> None:
        """Registra argumentos; ``exit_code`` simula que muere al arrancar."""
        self.args = args
        self.env = env
        self.exit_code = exit_code
        self.terminated = False

    def poll(self) -> int | None:
        """Código de salida o None si sigue vivo."""
        return 0 if self.terminated else self.exit_code

    def terminate(self) -> None:
        """Marca el proceso como detenido."""
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        """Devuelve inmediatamente."""
        return 0


@pytest.fixture()
def fake_spawn(monkeypatch: pytest.MonkeyPatch) -> list[FakeProc]:
    """Sustituye which/Popen/atexit/sleep y devuelve los procesos lanzados."""
    import translator

    procs: list[FakeProc] = []

    def popen(args: list, **kwargs: object) -> FakeProc:
        proc = FakeProc(args, **kwargs)  # type: ignore[arg-type]
        procs.append(proc)
        return proc

    monkeypatch.setattr(translator.shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr(translator.subprocess, "Popen", popen)
    monkeypatch.setattr(translator.atexit, "register", lambda fn: None)
    monkeypatch.setattr(translator.time, "sleep", lambda s: None)
    return procs


def test_ensure_server_does_nothing_if_already_running(fake_spawn: list) -> None:
    """Con el servidor ya vivo no se lanza nada (ni se detendrá luego)."""
    sub, _ = _subtitler([])
    assert sub.ensure_server() is False
    assert fake_spawn == []


def test_start_server_spawns_waits_and_close_stops_it(fake_spawn: list) -> None:
    """Sin servidor: lanza 'ollama serve', espera a que responda y close lo para."""
    http = FakeHttp([], alive=[False, False, True])
    sub = OllamaSubtitler(http=http, host="http://localhost:11500")
    assert sub.ensure_server() is True
    (proc,) = fake_spawn
    assert proc.args == ["/usr/bin/ollama", "serve"]
    assert proc.env["OLLAMA_HOST"] == "localhost:11500"
    sub.close()
    assert proc.terminated


def test_start_server_without_binary_explains_install(
    monkeypatch: pytest.MonkeyPatch, fake_spawn: list
) -> None:
    """Sin el ejecutable 'ollama' → FileNotFoundError con cómo instalarlo."""
    import translator

    monkeypatch.setattr(translator.shutil, "which", lambda name: None)
    sub, _ = _subtitler([])
    with pytest.raises(FileNotFoundError, match="ollama-cuda"):
        sub.start_server()


def test_remote_host_is_never_started(fake_spawn: list) -> None:
    """Un host remoto caído no se intenta arrancar."""
    http = FakeHttp([], alive=[False])
    sub = OllamaSubtitler(http=http, host="http://gpu-box:11434")
    with pytest.raises(ConnectionError, match="no ser local"):
        sub.ensure_server()
    assert fake_spawn == []


def test_server_dying_on_start_raises(
    monkeypatch: pytest.MonkeyPatch, fake_spawn: list
) -> None:
    """Si 'ollama serve' muere al arrancar → RuntimeError explicativo."""
    import translator

    monkeypatch.setattr(
        translator.subprocess, "Popen", lambda args, **kw: FakeProc(args, exit_code=1, **kw)
    )
    http = FakeHttp([], alive=[False])
    sub = OllamaSubtitler(http=http)
    with pytest.raises(RuntimeError, match="terminó al arrancar"):
        sub.start_server()


def test_dangling_slash_with_real_line_break_is_removed() -> None:
    """'tú /⏎sabes' (separador + salto real) queda como salto limpio."""
    subs = parse_srt_content(
        "1\n00:00:01,000 --> 00:00:02,000\nWhat I'm saying is, you\nknow what I have to do.\n"
    )
    sub, _ = _subtitler(["[#1] Lo que digo es que tú /\nsabes lo que tengo que hacer."])
    out = sub.translate_srt_text(subs)
    assert out[0].content == "Lo que digo es que tú\nsabes lo que tengo que hacer."
