"""Traducción de subtítulos con Gemini (librería oficial ``google-genai``).

Dos flujos:

- **Con SRT ya extraído**: :meth:`GeminiSubtitler.translate_srt_text`
  traduce los textos por ventanas (defecto 50 bloques) preservando
  timestamps vía ``srt_utils.replace_contents``.
- **Sin SRT** (solo audio del MKV): :meth:`GeminiSubtitler.transcribe_and_translate_audio`
  sube el MP3 con la Files API (``client.files.upload``), pide a Gemini
  transcripción + traducción directa a SRT, parsea el resultado y borra el
  archivo remoto en ``finally`` (``client.files.delete``).

Protocolo de ventanas: cada bloque se envía numerado como ``[#índice] texto``;
el modelo debe devolver exactamente una sección por bloque con el mismo
marcador, lo que permite un mapeo 1:1 robusto incluso con saltos de línea
internos. Cualquier descuadre cuenta como error (no se inventan timestamps).

Reanudación: :func:`translate_file` persiste cada ventana traducida en
``TranslationCache`` (``cache.py``) y solo reenvía a Gemini los chunks
pendientes; al completar, limpia la caché.

Resiliencia (Sprint 4):

- Las llamadas a ``generate_content`` reintentan hasta 3 veces con backoff
  exponencial (2s, 4s, 8s) ante errores transitorios (HTTP 429 /
  ResourceExhausted, 5xx, timeouts). Los errores no transitorios
  (p. ej. API key inválida) se propagan de inmediato.
- Si la validación 1:1 falla, la ventana se reintenta una vez con
  ``temperature=0.0`` y recordatorio estricto; si persiste, se divide a
  la mitad recursivamente; una línea aislada que siga fallando conserva
  el texto original y registra un aviso en ``fallback_notices``.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

import srt

from cache import TranslationCache
from srt_utils import (
    audit_subtitles,
    chunk_subtitles,
    parse_srt_content,
    read_srt_file,
    replace_contents,
    write_srt_file,
)

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_WINDOW_SIZE = 50

#: Reintentos ante errores transitorios de la API.
MAX_RETRIES = 3
#: Esperas entre reintentos (backoff exponencial 2s, 4s, 8s).
RETRY_DELAYS = (2.0, 4.0, 8.0)

_NUMBERED_BLOCK_RE = re.compile(r"\[#(\d+)\]")


class TranslationMismatchError(RuntimeError):
    """La respuesta de Gemini no mapea 1:1 con la ventana enviada.

    Subclase de ``RuntimeError`` para compatibilidad con código que
    captura el error genérico.
    """


def _is_transient_error(exc: BaseException) -> bool:
    """Indica si un error de la API merece reintento con backoff.

    Args:
        exc: Excepción lanzada por ``generate_content``.

    Returns:
        ``True`` ante rate limit / 429 / ResourceExhausted, errores 5xx
        de servidor, timeouts y cortes de conexión. ``False`` ante
        errores de autenticación, argumentos inválidos, etc.
    """
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if value in (429, 500, 502, 503, 504):
            return True
        if isinstance(value, str) and value.strip() in {
            "429",
            "500",
            "502",
            "503",
            "504",
            "RESOURCE_EXHAUSTED",
            "UNAVAILABLE",
        }:
            return True
    haystack = f"{type(exc).__name__} {exc}".lower()
    markers = (
        "429",
        "resourceexhausted",
        "resource exhausted",
        "rate limit",
        "rate-limit",
        "ratelimit",
        "too many requests",
        "unavailable",
        "service unavailable",
        "overloaded",
        "timeout",
        "timed out",
        "deadline exceeded",
        "connection reset",
        "connection aborted",
        "temporarily",
        "try again",
        "500",
        "502",
        "503",
        "504",
    )
    return any(marker in haystack for marker in markers)


def _format_glossary(glossary: dict[str, str] | None) -> str:
    """Formatea el glosario para inyectarlo en el prompt.

    Args:
        glossary: Mapa término original → traducción fijada.

    Returns:
        Bloque de texto (o ``"(sin glosario)"`` si está vacío).
    """
    if not glossary:
        return "(sin glosario)"
    return "\n".join(f'- "{src}" siempre como "{dst}"' for src, dst in glossary.items())


class GeminiSubtitler:
    """Traductor de subtítulos basado en la API de Gemini."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        client: Any | None = None,
        glossary: dict[str, str] | None = None,
        src_lang: str = "en",
        dst_lang: str = "es",
        window_size: int = DEFAULT_WINDOW_SIZE,
        style_instructions: str = "",
    ) -> None:
        """Crea el traductor.

        Args:
            api_key: Clave de Gemini (obligatoria salvo que se inyecte
                ``client`` ya construido, p. ej. en tests).
            model: Modelo generativo a usar.
            client: Cliente ``genai.Client`` ya construido (inyección para
                tests). Si es ``None``, se crea con ``api_key``.
            glossary: Nombres/términos con traducción fijada.
            src_lang: Idioma origen (nombre o código).
            dst_lang: Idioma destino.
            window_size: Bloques por llamada (defecto 50).
            style_instructions: Directrices de tono/estilo (p. ej.
                ``[translation].style`` del TOML). Se inyectan en todos
                los prompts, incluidas las subventanas del fallback.

        Raises:
            ValueError: Si no hay ``api_key`` ni ``client``.
            ImportError: Si ``google-genai`` no está instalado y hay que
                crear el cliente interno.
        """
        if client is None and not api_key:
            raise ValueError("Falta api_key de Gemini (o inyecta un client).")
        self.api_key = api_key
        self.model = model
        self.client = client if client is not None else self._build_client(api_key)
        self.glossary: dict[str, str] = dict(glossary or {})
        self.src_lang = src_lang
        self.dst_lang = dst_lang
        self.window_size = window_size
        self.style_instructions = style_instructions
        self.fallback_notices: list[str] = []
        """Avisos de líneas conservadas en original tras agotar el fallback."""

    @staticmethod
    def _build_client(api_key: str) -> Any:
        """Construye el cliente oficial ``genai.Client``.

        Args:
            api_key: Clave de Gemini.

        Returns:
            Cliente construido.

        Raises:
            ImportError: Si ``google-genai`` no está instalado.
        """
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Falta la dependencia 'google-genai'. "
                "Instálala con: pip install google-genai"
            ) from exc
        return genai.Client(api_key=api_key)

    def _generate_text(self, contents: Any, temperature: float | None = None) -> str:
        """Llama a ``models.generate_content`` con reintentos y devuelve el texto.

        Ante errores transitorios (rate limit 429, 5xx, timeouts) reintenta
        hasta :data:`MAX_RETRIES` veces con esperas :data:`RETRY_DELAYS`.

        Args:
            contents: Prompt (str) o lista ``[audio_file, prompt]``.
            temperature: Si se indica, se envía como ``config`` del modelo
                (p. ej. ``0.0`` en el reintento estricto).

        Returns:
            Texto generado (sin espacios extremos).

        Raises:
            RuntimeError: Si la respuesta viene vacía.
            Exception: El último error transitorio tras agotar reintentos,
                o cualquier error no transitorio de inmediato.
        """
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["config"] = {"temperature": temperature}
        attempt = 0
        while True:
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=contents, **kwargs
                )
            except Exception as exc:  # noqa: BLE001 — clasificar transitorio o no
                if _is_transient_error(exc) and attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAYS[attempt])
                    attempt += 1
                    continue
                raise
            text = (getattr(response, "text", "") or "").strip()
            if not text:
                raise RuntimeError("Gemini devolvió una respuesta vacía.")
            return text

    def _build_translation_prompt(
        self, window: list[srt.Subtitle], strict: bool = False
    ) -> str:
        """Construye el prompt de traducción para una ventana.

        Args:
            window: Bloques consecutivos a traducir.
            strict: Si ``True``, añade un recordatorio de conteo exacto
                (reintento tras un descuadre 1:1).

        Returns:
            Prompt con glosario y bloques numerados ``[#índice]``.
        """
        numbered = "\n".join(
            f"[#{sub.index}] {sub.content.replace(chr(10), ' / ')}" for sub in window
        )
        prompt = (
            f"Traduce estos subtítulos del {self.src_lang} al {self.dst_lang}. "
            "Devuelve EXACTAMENTE una sección por bloque, en el mismo orden, "
            "cada una empezando por su marcador [#índice] seguido de la traducción. "
            "No añadas, quites ni reordenes bloques. Conserva el tono y la "
            "coherencia de pronombres entre líneas. Si el original tiene varias "
            "líneas, separa la traducción con saltos de línea reales dentro de "
            "su sección. No toques los timestamps (no los incluyas).\n"
            f"Glosario (obligatorio):\n{_format_glossary(self.glossary)}\n"
            f"{self._style_block()}"
            f"Bloques:\n{numbered}"
        )
        if strict:
            prompt += (
                f"\nRECORDATORIO ESTRICTO: el lote tiene {len(window)} bloques y "
                "tu respuesta debe tener EXACTAMENTE esas "
                f"{len(window)} secciones, ni una más ni una menos."
            )
        return prompt

    def _parse_numbered_response(
        self, text: str, expected_indices: list[int]
    ) -> list[str]:
        """Mapea la respuesta numerada a textos en orden de ventana.

        Acepta contenido multilínea por bloque: todo lo que haya entre un
        marcador ``[#n]`` y el siguiente pertenece a ese bloque.

        Args:
            text: Respuesta cruda del modelo.
            expected_indices: Índices ``Subtitle.index`` esperados, en orden.

        Returns:
            Traducciones en el mismo orden que ``expected_indices``.

        Raises:
            TranslationMismatchError: Si faltan/sobran marcadores, el orden
                no coincide o algún bloque viene vacío.
        """
        matches = list(_NUMBERED_BLOCK_RE.finditer(text))
        found = [int(m.group(1)) for m in matches]
        if found != list(expected_indices):
            raise TranslationMismatchError(
                f"Respuesta de Gemini descuadrada: se esperaban bloques "
                f"{list(expected_indices)} y llegaron {found}. "
                f"Fragmento: {text[:500]!r}"
            )
        translations: list[str] = []
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            translations.append(text[start:end].strip())
        if any(not t for t in translations):
            raise TranslationMismatchError(
                f"Gemini devolvió algún bloque vacío. Fragmento: {text[:500]!r}"
            )
        return translations

    def _translate_window(
        self,
        window: list[srt.Subtitle],
        strict: bool = False,
        temperature: float | None = None,
    ) -> list[srt.Subtitle]:
        """Traduce una ventana preservando timestamps.

        Args:
            window: Bloques consecutivos.
            strict: Añade el recordatorio de conteo exacto al prompt.
            temperature: Temperatura del modelo para esta llamada
                (``None`` = defecto del modelo).

        Returns:
            Bloques traducidos (mismos ``index``/``start``/``end``).

        Raises:
            TranslationMismatchError: Si la respuesta no mapea 1:1.
        """
        prompt = self._build_translation_prompt(window, strict=strict)
        raw = self._generate_text(prompt, temperature=temperature)
        translated_texts = self._parse_numbered_response(
            raw, [sub.index for sub in window]
        )
        return replace_contents(window, translated_texts)

    def _style_block(self) -> str:
        """Devuelve el bloque de estilo para los prompts (o vacío).

        Returns:
            Línea ``"Estilo: ...\\n"`` si hay directrices, ``""`` si no.
        """
        if self.style_instructions.strip():
            return f"Estilo (obligatorio): {self.style_instructions.strip()}\n"
        return ""

    def _translate_window_resilient(
        self, window: list[srt.Subtitle]
    ) -> tuple[list[srt.Subtitle], list[str]]:
        """Traduce una ventana con fallback ante descuadres 1:1.

        1. Intento normal; 2. reintento estricto (``temperature=0.0``);
        3. división recursiva a la mitad; 4. una línea aislada que siga
        fallando conserva el original y genera un aviso.

        Todas las llamadas (incluidas las subventanas) usan el mismo
        ``self``: glosario y directrices de estilo viajan intactos.

        Args:
            window: Bloques consecutivos (no vacía).

        Returns:
            Tupla ``(bloques, avisos)``: mismos tiempos siempre; los avisos
            describen líneas conservadas en original.
        """
        try:
            return self._translate_window(window), []
        except TranslationMismatchError:
            pass
        try:
            return self._translate_window(window, strict=True, temperature=0.0), []
        except TranslationMismatchError:
            pass
        if len(window) <= 1:
            sub = window[0]
            notice = (
                f"#{sub.index}: sin traducción fiable tras reintentos; "
                "se conserva el texto original."
            )
            return list(window), [notice]
        mid = len(window) // 2
        left, notices_left = self._translate_window_resilient(window[:mid])
        right, notices_right = self._translate_window_resilient(window[mid:])
        return left + right, notices_left + notices_right

    def translate_srt_text(
        self, subtitles: list[srt.Subtitle], resilient: bool = True
    ) -> list[srt.Subtitle]:
        """Traduce subtítulos ya extraídos, conservando timestamps.

        Args:
            subtitles: Bloques en inglés (u origen configurado).
            resilient: Si ``True`` (defecto), los descuadres 1:1 usan el
                fallback (reintento estricto → división → conservar
                original con aviso en :attr:`fallback_notices`). Si
                ``False``, cualquier descuadre lanza
                :class:`TranslationMismatchError`.

        Returns:
            Bloques traducidos al destino, mismos tiempos e índices.
            Lista vacía si la entrada está vacía (sin llamadas a la API).

        Raises:
            TranslationMismatchError: Solo con ``resilient=False``.
        """
        if not subtitles:
            return []
        self.fallback_notices = []
        if not resilient:
            translated: list[srt.Subtitle] = []
            for window in chunk_subtitles(subtitles, window_size=self.window_size):
                translated.extend(self._translate_window(window))
            return translated
        translated = []
        for window in chunk_subtitles(subtitles, window_size=self.window_size):
            done, notices = self._translate_window_resilient(window)
            translated.extend(done)
            self.fallback_notices.extend(notices)
        return translated

    def _build_transcription_prompt(self) -> str:
        """Construye el prompt de transcripción+traducción desde audio.

        Returns:
            Prompt que exige SRT válido completo en destino.
        """
        return (
            f"Transcribe este audio ({self.src_lang}) y tradúcelo al {self.dst_lang}. "
            "Devuelve ÚNICAMENTE un archivo SRT válido completo, con índices "
            "secuenciales desde 1, timestamps 'HH:MM:SS,mmm --> HH:MM:SS,mmm' y "
            "texto traducido. Sin explicaciones ni bloques de código.\n"
            f"Glosario (obligatorio):\n{_format_glossary(self.glossary)}\n"
            f"{self._style_block()}"
        )

    def transcribe_and_translate_audio(
        self,
        audio_path: str | Path,
        src_lang: str | None = None,
        dst_lang: str | None = None,
    ) -> list[srt.Subtitle]:
        """Transcribe el audio y lo traduce a SRT en un paso con Gemini.

        Sube el archivo con la Files API y **siempre** borra el archivo
        remoto afterwards (``finally``), haya éxito o error.

        Args:
            audio_path: MP3/WAV local (p. ej. salida de ``extract_audio``).
            src_lang: Override del idioma origen para esta llamada.
            dst_lang: Override del idioma destino para esta llamada.

        Returns:
            Bloques SRT traducidos y parseados.

        Raises:
            FileNotFoundError: Si el audio local no existe.
            ValueError: Si el SRT devuelto no parsea.
            RuntimeError: Si Gemini devuelve vacío.
        """
        local = Path(audio_path)
        if not local.is_file():
            raise FileNotFoundError(f"No existe el audio: {local}")
        if src_lang is not None:
            self.src_lang = src_lang
        if dst_lang is not None:
            self.dst_lang = dst_lang

        remote = self.client.files.upload(file=str(local))
        try:
            raw_srt = self._generate_text([remote, self._build_transcription_prompt()])
        finally:
            try:
                self.client.files.delete(name=remote.name)
            except Exception:
                pass  # la limpieza no debe ocultar el resultado/error principal
        try:
            subtitles = parse_srt_content(raw_srt)
        except ValueError as exc:
            raise ValueError(
                f"Gemini no devolvió un SRT válido: {exc}. "
                f"Fragmento: {raw_srt[:500]!r}"
            ) from exc
        if not subtitles:
            raise ValueError("Gemini devolvió un SRT vacío.")
        return subtitles


def _translate_one_window(
    subtitler: GeminiSubtitler, window: list[srt.Subtitle]
) -> tuple[list[srt.Subtitle], list[str]]:
    """Traduce una ventana usando el camino resiliente si existe.

    Acepta dobles de test mínimos que solo implementan ``_translate_window``.

    Args:
        subtitler: Traductor (real o doble de test).
        window: Bloques consecutivos.

    Returns:
        Tupla ``(bloques, avisos)``.
    """
    resilient = getattr(subtitler, "_translate_window_resilient", None)
    if resilient is not None:
        return resilient(window)
    return subtitler._translate_window(window), []


def translate_file(
    srt_path: str | Path,
    output: str | Path | None,
    subtitler: GeminiSubtitler,
    max_chars: int = 42,
    max_cps: float = 17.0,
    use_cache: bool = True,
    show_progress: bool = True,
) -> tuple[Path, list]:
    """Traduce un archivo ``.srt`` completo a ``*_es.srt`` con reanudación.

    Nunca sobrescribe el original: si ``output`` es ``None`` se deriva
    ``<stem>_es.srt``; si coincide con la entrada se rechaza.

    La caché (``<stem>.subtrans-cache.json`` junto al origen) guarda cada
    ventana traducida a medida que llega de Gemini; si el proceso se
    interrumpe, la siguiente ejecución solo reenvía los chunks pendientes.
    Tras escribir el destino con éxito y con todos los chunks completos,
    la caché se limpia.

    Muestra una barra de progreso ventana por ventana cuando la salida es
    un terminal interactivo (se omite bajo pipes/tests).

    Args:
        srt_path: SRT origen.
        output: Destino explícito o ``None`` para derivarlo.
        subtitler: Instancia configurada de :class:`GeminiSubtitler`.
        max_chars: Límite para la auditoría de longitud.
        max_cps: Límite para la auditoría de velocidad.
        use_cache: Si ``False``, traduce todo sin leer ni escribir caché
            (útil para tests o para forzar una pasada limpia).
        show_progress: Muestra barra de progreso Rich (solo si hay TTY).

    Returns:
        Tupla ``(ruta_destino, avisos)`` donde avisos es la lista de
        ``LengthWarning`` (solo avisos, el texto no se modifica).

    Raises:
        ValueError: Si el destino coincide con el origen o si una entrada
            de caché no encaja con su ventana (tamaños distintos).
    """
    src = Path(srt_path)
    dst = Path(output) if output is not None else src.with_name(src.stem + "_es.srt")
    if dst.resolve() == src.resolve():
        raise ValueError(
            f"El destino ({dst}) coincide con el original; "
            "usa otro nombre (p. ej. *_es.srt)."
        )
    subtitles = read_srt_file(src)
    if not subtitles:
        write_srt_file(dst, [])
        return dst, []
    windows = chunk_subtitles(subtitles, window_size=subtitler.window_size)
    cache = TranslationCache(src) if use_cache else None
    notices: list[str] = getattr(subtitler, "fallback_notices", None) or []

    translated_windows: list[list[srt.Subtitle]] = []
    use_bar = show_progress and sys.stderr.isatty()
    progress = None
    task_id = None
    if use_bar:
        from rich.progress import Progress

        progress = Progress(transient=True)
        task_id = progress.add_task("Traduciendo", total=len(windows))
        progress.start()
    try:
        for chunk_index, window in enumerate(windows):
            cached_texts = cache.get(chunk_index) if cache is not None else None
            if cached_texts is not None:
                if len(cached_texts) != len(window):
                    raise ValueError(
                        f"Caché inconsistente en {cache.path if cache else '?'}: "
                        f"chunk {chunk_index} guarda {len(cached_texts)} textos "
                        f"para {len(window)} bloques (¿cambió window_size?). "
                        "Borra la caché para retraducir desde cero."
                    )
                translated_windows.append(replace_contents(window, cached_texts))
            else:
                done, window_notices = _translate_one_window(subtitler, window)
                notices.extend(window_notices)
                if cache is not None:
                    cache.set(chunk_index, [sub.content for sub in done])
                translated_windows.append(done)
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)
    finally:
        if progress is not None:
            progress.stop()

    if isinstance(getattr(subtitler, "fallback_notices", None), list):
        subtitler.fallback_notices.extend(
            n for n in notices if n not in subtitler.fallback_notices
        )
    translated = [sub for window in translated_windows for sub in window]
    warnings = audit_subtitles(translated, max_chars=max_chars, max_cps=max_cps)
    write_srt_file(dst, translated)
    if cache is not None and cache.is_complete(len(windows)):
        cache.clear()
    return dst, warnings
