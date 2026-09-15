"""Utilidades de parseo, reconstrucción y auditoría de archivos SRT.

Este módulo es la única capa autorizada para leer/escribir SRT en el
proyecto. Garantiza:

- Detección y normalización de encoding (UTF-8 con/sin BOM, latin-1).
- Parseo/reconstrucción con la librería ``srt`` preservando timestamps.
- Auditoría de longitud **sin modificar el texto** (solo avisos):
  ~42 caracteres por línea y ~17 caracteres/segundo.
- Agrupación en ventanas consecutivas para traducción con contexto.
- Reemplazo de contenidos preservando timestamps exactos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import srt

DEFAULT_MAX_CHARS_PER_LINE = 42
DEFAULT_MAX_CPS = 17.0
DEFAULT_WINDOW_SIZE = 50
MIN_WINDOW_SIZE = 1
MAX_WINDOW_SIZE = 200

WarningKind = Literal["max_chars", "cps"]


@dataclass(frozen=True)
class LengthWarning:
    """Aviso de legibilidad para un bloque de subtítulo."""

    index: int
    """Número de subtítulo (``srt.Subtitle.index``)."""
    kind: WarningKind
    """``max_chars`` si una línea supera el límite, ``cps`` si va muy rápido."""
    value: float
    """Valor medido (nº de caracteres o caracteres/segundo)."""
    limit: float
    """Límite configurado que se ha superado."""
    detail: str
    """Texto o línea concreta que provoca el aviso."""
    message: str
    """Mensaje humano ya formateado."""


def detect_encoding(path: str | Path) -> str:
    """Detecta el encoding de un archivo SRT.

    Estrategia mínima y determinista, sin dependencias externas:

    - Si hay BOM UTF-8 (``\\xef\\xbb\\xbf``) → ``"utf-8-sig"``.
    - Si el contenido decodifica como UTF-8 estricto → ``"utf-8"``.
    - En otro caso → ``"latin-1"`` (decodifica cualquier byte).

    Args:
        path: Ruta al archivo ``.srt``.

    Returns:
        Nombre de codec válido para ``open(..., encoding=...)``.
    """
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def strip_markdown_fences(text: str) -> str:
    """Elimina delimitadores de bloques de código markdown si están presentes.

    Gemini a veces envuelve la salida SRT en ```srt ... ``` a pesar de
    indicar lo contrario en el prompt.
    """
    s = text.strip()
    if s.startswith("```"):
        newline = s.find("\n")
        if newline != -1:
            s = s[newline + 1:]
        else:
            s = ""
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def normalize_text(text: str) -> str:
    """Normaliza texto SRT recién leído.

    - Elimina BOM (``\\ufeff``) residual.
    - Elimina delimitadores de bloques markdown (`````srt ... `````).
    - Normaliza saltos ``\\r\\n`` / ``\\r`` a ``\\n``.
    - Elimina espacios trailing por línea (no toca líneas vacías
      estructurales ni el contenido traducible en sí).

    Args:
        text: Contenido crudo del archivo.

    Returns:
        Texto normalizado listo para ``srt.parse``.
    """
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    text = strip_markdown_fences(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    return "\n".join(lines)


def parse_srt_content(text: str) -> list[srt.Subtitle]:
    """Parsea contenido SRT ya normalizado.

    Args:
        text: Texto SRT con saltos ``\\n``.

    Returns:
        Lista de ``srt.Subtitle`` en orden de archivo.

    Raises:
        ValueError: Si ``srt`` no puede parsear el contenido.
    """
    try:
        return list(srt.parse(normalize_text(text)))
    except srt.SRTParseError as exc:
        raise ValueError(f"Contenido SRT inválido: {exc}") from exc


def compose_srt_content(subtitles: list[srt.Subtitle]) -> str:
    """Reconstruye contenido SRT a partir de subtítulos.

    Los timestamps se re-emiten desde los objetos ``Subtitle`` sin
    alteración: esta función nunca toca ``start``/``end``.

    Args:
        subtitles: Bloques a serializar.

    Returns:
        Texto SRT terminado en un único ``\\n`` final (``""`` si vacío).
    """
    if not subtitles:
        return ""
    composed = srt.compose(subtitles)
    if not composed.endswith("\n"):
        composed += "\n"
    return composed


def read_srt_file(path: str | Path) -> list[srt.Subtitle]:
    """Lee un archivo ``.srt`` detectando su encoding.

    Args:
        path: Ruta al archivo.

    Returns:
        Bloques parseados.

    Raises:
        FileNotFoundError: Si la ruta no existe.
        ValueError: Si el contenido no es SRT válido.
    """
    file_path = Path(path)
    encoding = detect_encoding(file_path)
    text = file_path.read_text(encoding=encoding)
    return parse_srt_content(text)


def write_srt_file(
    path: str | Path,
    subtitles: list[srt.Subtitle],
    encoding: str = "utf-8",
) -> Path:
    """Escribe bloques SRT a disco (siempre UTF-8 por defecto, sin BOM).

    Crea el directorio padre si no existe. No hace control de
    sobrescritura: la política de "nunca sobrescribir el original"
    (sufijo ``_es.srt``) vive en ``translator.py`` / ``app.py``.

    Args:
        path: Destino ``.srt``.
        subtitles: Bloques a escribir.
        encoding: Codec de salida (por defecto ``"utf-8"``).

    Returns:
        La ruta escrita como ``Path``.
    """
    file_path = Path(path)
    if file_path.parent != Path("") and str(file_path.parent):
        file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(compose_srt_content(subtitles), encoding=encoding)
    return file_path


def subtitle_duration_seconds(sub: srt.Subtitle) -> float:
    """Devuelve la duración visible de un bloque en segundos.

    Args:
        sub: Bloque individual.

    Returns:
        Segundos (``end - start``). Mínimo ``0.0`` si hay timestamps
        degenerados; nunca negativo.
    """
    delta = sub.end - sub.start
    return max(0.0, delta.total_seconds())


def _text_chars(sub: srt.Subtitle) -> int:
    """Cuenta caracteres legibles (sin saltos de línea)."""
    return sum(len(line) for line in sub.content.splitlines())


def check_line_length(
    sub: srt.Subtitle, max_chars: int = DEFAULT_MAX_CHARS_PER_LINE
) -> list[LengthWarning]:
    """Comprueba que ninguna línea visual supere ``max_chars``.

    Args:
        sub: Bloque a auditar.
        max_chars: Límite de caracteres por línea (def. 42).

    Returns:
        Un ``LengthWarning`` por cada línea que exceda el límite.
    """
    warnings: list[LengthWarning] = []
    for line in sub.content.splitlines():
        if len(line) > max_chars:
            warnings.append(
                LengthWarning(
                    index=sub.index,
                    kind="max_chars",
                    value=float(len(line)),
                    limit=float(max_chars),
                    detail=line,
                    message=(
                        f"#{sub.index}: línea de {len(line)} chars "
                        f"(límite {max_chars}): {line!r}"
                    ),
                )
            )
    return warnings


def check_reading_speed(
    sub: srt.Subtitle, max_cps: float = DEFAULT_MAX_CPS
) -> list[LengthWarning]:
    """Comprueba la velocidad de lectura (caracteres/segundo).

    CPS = caracteres totales (sin saltos) / duración en segundos.
    Bloques con duración <= 0 se avisan con valor ``inf`` si tienen texto.

    Args:
        sub: Bloque a auditar.
        max_cps: Límite recomendado (def. 17.0).

    Returns:
        Cero o un ``LengthWarning`` de tipo ``"cps"``.
    """
    duration = subtitle_duration_seconds(sub)
    chars = _text_chars(sub)
    cps = (chars / duration) if duration > 0 else (float("inf") if chars else 0.0)
    if cps > max_cps:
        return [
            LengthWarning(
                index=sub.index,
                kind="cps",
                value=cps,
                limit=max_cps,
                detail=sub.content,
                message=(
                    f"#{sub.index}: {cps:.1f} cps "
                    f"(límite {max_cps:.1f}, {chars} chars en "
                    f"{duration:.2f}s)"
                ),
            )
        ]
    return []


def audit_subtitles(
    subtitles: list[srt.Subtitle],
    max_chars: int = DEFAULT_MAX_CHARS_PER_LINE,
    max_cps: float = DEFAULT_MAX_CPS,
) -> list[LengthWarning]:
    """Audita todos los bloques sin modificarlos (solo avisos).

    Args:
        subtitles: Bloques a auditar.
        max_chars: Límite por línea.
        max_cps: Límite de velocidad lectora.

    Returns:
        Avisos en orden de subtítulo (primero ``max_chars``, luego ``cps``).
    """
    findings: list[LengthWarning] = []
    for sub in subtitles:
        findings.extend(check_line_length(sub, max_chars=max_chars))
        findings.extend(check_reading_speed(sub, max_cps=max_cps))
    return findings


def format_warning(warning: LengthWarning) -> str:
    """Formatea un aviso para stderr/log.

    Args:
        warning: Aviso generado por la auditoría.

    Returns:
        Su mensaje humano.
    """
    return warning.message


def chunk_subtitles(
    subtitles: list[srt.Subtitle],
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> list[list[srt.Subtitle]]:
    """Agrupa bloques consecutivos en ventanas para traducir con contexto.

    La traducción línea-a-línea aislada rompe pronombres/tono; por eso el
    traductor envía ventanas consecutivas por llamada.

    Nota Sprint 3: la ventana por defecto es 50 bloques (equilibrio entre
    contexto para Gemini y blast-radius ante un fallo de parseo en una
    ventana). Es configurable vía TOML/CLI (``window_size``, rango 1-200):
    ``translator.py`` + ``cache.py`` guardan progreso por ventana y validan
    el mapeo 1:1 antes de continuar.

    Args:
        subtitles: Bloques en orden temporal.
        window_size: Tamaño de ventana (debe estar entre 1 y 200,
            por defecto 50).

    Returns:
        Lista de ventanas; la última puede ser más pequeña.

    Raises:
        ValueError: Si ``window_size`` está fuera de [1, 200].
    """
    if not MIN_WINDOW_SIZE <= window_size <= MAX_WINDOW_SIZE:
        raise ValueError(
            f"window_size debe estar entre {MIN_WINDOW_SIZE} y "
            f"{MAX_WINDOW_SIZE}, recibido {window_size}"
        )
    return [
        subtitles[i : i + window_size] for i in range(0, len(subtitles), window_size)
    ]


def replace_contents(
    subtitles: list[srt.Subtitle], new_contents: list[str]
) -> list[srt.Subtitle]:
    """Devuelve copias con el texto sustituido y timestamps intactos.

    Es la única vía que deberá usar ``translator.py`` para aplicar
    traducciones: garantiza que ``start``/``end``/``index`` no cambian.

    Args:
        subtitles: Originales (no se mutan).
        new_contents: Textos traducidos, uno por bloque y en orden.

    Returns:
        Nuevos ``Subtitle`` con mismos tiempos e índices.

    Raises:
        ValueError: Si las longitudes no coinciden.
    """
    if len(subtitles) != len(new_contents):
        raise ValueError(
            f" Nº de textos ({len(new_contents)}) != nº de subtítulos "
            f"({len(subtitles)}): los timestamps deben conservarse 1:1"
        )
    return [
        srt.Subtitle(
            index=sub.index,
            start=sub.start,
            end=sub.end,
            content=text,
            proprietary=sub.proprietary,
        )
        for sub, text in zip(subtitles, new_contents)
    ]
