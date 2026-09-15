"""Carga de configuración TOML (Gemini, idiomas, glosario, límites).

Esquema esperado (``config.toml``)::

    [gemini]
    api_key = "AIza..."   # o variable de entorno GEMINI_API_KEY
    model = "gemini-3.8-flash"

    [translation]
    source = "en"
    target = "es"
    window_size = 50
    style = "Español de España, tono natural y cinematográfico"

    [limits]
    max_chars = 42
    max_cps = 17.0

    [glossary]
    "Jon Snow" = "Jon Nieve"

Compatibilidad: la sección antigua ``[languages]`` (``src``/``dst``) y
``[limits].window_size`` siguen leyéndose como fallback cuando falta
``[translation]``. Todas las secciones son opcionales: lo ausente se
rellena con valores por defecto. ``api_key`` también puede venir de
``GEMINI_API_KEY``.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_WINDOW_SIZE = 50


@dataclass
class GeminiConfig:
    """Credenciales y modelo de Gemini."""

    api_key: str = ""
    """Clave de la API de Gemini (o ``GEMINI_API_KEY``)."""
    model: str = DEFAULT_GEMINI_MODEL
    """Modelo generativo (p. ej. ``gemini-3.8-flash``)."""


@dataclass
class AppConfig:
    """Configuración efectiva de la aplicación."""

    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    src_lang: str = "en"
    dst_lang: str = "es"
    glossary: dict[str, str] = field(default_factory=dict)
    max_chars: int = 42
    max_cps: float = 17.0
    window_size: int = DEFAULT_WINDOW_SIZE
    style_instructions: str = ""
    """Directrices de estilo/tono (sección ``[translation].style``)."""

    @property
    def api_key(self) -> str:
        """Atajo a ``gemini.api_key`` (compatibilidad con sprint 1)."""
        return self.gemini.api_key


def default_config_search_paths() -> list[Path]:
    """Rutas de búsqueda de configuración, en orden de prioridad.

    Returns:
        ``[./config.toml, ~/.config/subtrans/config.toml]``.
    """
    return [
        Path("config.toml"),
        Path.home() / ".config" / "subtrans" / "config.toml",
    ]


def load_config(path: str | Path | None = None) -> AppConfig:
    """Carga la configuración TOML con valores por defecto.

    Args:
        path: Ruta explícita al ``.toml``. Si es ``None``, se prueba
            :func:`default_config_search_paths` en orden y, si ninguna
            existe, se devuelven los valores por defecto (con
            ``GEMINI_API_KEY`` del entorno si está definida).

    Returns:
        Configuración efectiva.

    Raises:
        FileNotFoundError: Si ``path`` explícito no existe.
        ValueError: Si el TOML es inválido o tiene tipos incorrectos.
    """
    if path is not None:
        explicit = Path(path)
        if not explicit.is_file():
            raise FileNotFoundError(f"No existe el archivo de config: {explicit}")
        return _parse_toml(explicit.read_bytes(), origin=explicit)

    for candidate in default_config_search_paths():
        if candidate.is_file():
            return _parse_toml(candidate.read_bytes(), origin=candidate)
    return _from_dict({})  # defecto + entorno


def _parse_toml(raw: bytes, origin: Path) -> AppConfig:
    """Parsea bytes TOML a ``AppConfig``.

    Args:
        raw: Contenido del archivo.
        origin: Ruta origen (solo para mensajes de error).

    Returns:
        Configuración efectiva.

    Raises:
        ValueError: Si el TOML no parsea.
    """
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Config inválida en {origin}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config inválida en {origin}: raíz debe ser tabla TOML.")
    return _from_dict(data)


def _from_dict(data: dict) -> AppConfig:
    """Construye ``AppConfig`` desde un dict ya parseado.

    Args:
        data: Tablas TOML (puede estar vacío).

    Returns:
        Configuración efectiva con fallbacks y entorno aplicado.
    """
    gemini_raw = data.get("gemini", {}) or {}
    translation_raw = data.get("translation", {}) or {}
    lang_raw = data.get("languages", {}) or {}
    limits_raw = data.get("limits", {}) or {}
    gloss_raw = data.get("glossary", {}) or {}

    env_key = os.environ.get("GEMINI_API_KEY", "")
    gemini = GeminiConfig(
        api_key=str(gemini_raw.get("api_key", env_key) or env_key or ""),
        model=str(gemini_raw.get("model", DEFAULT_GEMINI_MODEL)),
    )
    glossary = {str(k): str(v) for k, v in dict(gloss_raw).items()}
    src = translation_raw.get("source", lang_raw.get("src", "en"))
    dst = translation_raw.get("target", lang_raw.get("dst", "es"))
    window = translation_raw.get(
        "window_size", limits_raw.get("window_size", DEFAULT_WINDOW_SIZE)
    )
    return AppConfig(
        gemini=gemini,
        src_lang=str(src),
        dst_lang=str(dst),
        glossary=glossary,
        max_chars=int(limits_raw.get("max_chars", 42)),
        max_cps=float(limits_raw.get("max_cps", 17.0)),
        window_size=int(window),
        style_instructions=str(translation_raw.get("style", "")),
    )
