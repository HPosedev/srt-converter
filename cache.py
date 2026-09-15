"""Caché de progreso para reanudar traducciones por ventanas.

Cada archivo ``.srt`` origen tiene un JSON temporal junto a él::

    <stem>.subtrans-cache.json

que mapea ``chunk_index`` (posición de la ventana según
``srt_utils.chunk_subtitles``) a la lista de textos ya traducidos.
Si el proceso se corta, :func:`translate_file` solo reenvía a Gemini
los chunks ausentes. Cuando todos los chunks están completos, la caché
puede limpiarse con :meth:`TranslationCache.clear`.
"""

from __future__ import annotations

import json
from pathlib import Path

CACHE_SUFFIX = ".subtrans-cache.json"


def cache_path_for(srt_path: str | Path) -> Path:
    """Devuelve la ruta del JSON temporal junto al SRT.

    Args:
        srt_path: Archivo ``.srt`` origen.

    Returns:
        Ruta ``<stem>.subtrans-cache.json`` junto al SRT.
    """
    p = Path(srt_path)
    return p.with_name(p.stem + CACHE_SUFFIX)


class TranslationCache:
    """Progreso de traducción por ``chunk_index`` persistido en JSON."""

    def __init__(self, srt_path: str | Path) -> None:
        """Asocia la caché al SRT origen (no crea el archivo aún).

        Args:
            srt_path: Archivo ``.srt`` origen cuya traducción se cachea.
        """
        self.srt_path = Path(srt_path)
        self.path = cache_path_for(self.srt_path)

    def load(self) -> dict[int, list[str]]:
        """Carga el progreso guardado.

        Returns:
            Mapa ``chunk_index → textos traducidos``. Vacío si no hay
            archivo de caché todavía.

        Raises:
            ValueError: Si el JSON existe pero está corrupto o tiene un
                esquema inesperado (se incluye la ruta en el mensaje).
        """
        if not self.path.is_file():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Caché corrupta en {self.path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(
                f"Caché corrupta en {self.path}: la raíz debe ser un objeto JSON."
            )
        progress: dict[int, list[str]] = {}
        for key, value in raw.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Caché corrupta en {self.path}: clave {key!r} no es un índice."
                ) from exc
            if not isinstance(value, list) or not all(
                isinstance(t, str) for t in value
            ):
                raise ValueError(
                    f"Caché corrupta en {self.path}: chunk {index} debe ser "
                    "una lista de strings."
                )
            progress[index] = list(value)
        return progress

    def get(self, chunk_index: int) -> list[str] | None:
        """Devuelve los textos cacheados de una ventana, o ``None``.

        Args:
            chunk_index: Posición de la ventana.

        Returns:
            Lista de textos traducidos, o ``None`` si ese chunk aún no
            se tradujo (o la caché no existe / está vacía).
        """
        return self.load().get(chunk_index)

    def set(self, chunk_index: int, texts: list[str]) -> Path:
        """Guarda los textos traducidos de una ventana.

        Lee-modifica-escribe el JSON completo para no perder otros chunks.

        Args:
            chunk_index: Posición de la ventana.
            texts: Textos traducidos de esa ventana, en orden.

        Returns:
            Ruta del archivo de caché escrito.
        """
        progress = self.load()
        progress[chunk_index] = list(texts)
        serialised = {str(k): v for k, v in sorted(progress.items())}
        if self.path.parent != Path("") and str(self.path.parent):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(serialised, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.path

    def pending(self, total_chunks: int) -> list[int]:
        """Devuelve los índices de chunks aún sin traducir.

        Args:
            total_chunks: Nº total de ventanas del SRT.

        Returns:
            Índices en ``range(total_chunks)`` ausentes en la caché.
        """
        done = set(self.load())
        return [i for i in range(total_chunks) if i not in done]

    def is_complete(self, total_chunks: int) -> bool:
        """Indica si todos los chunks están cacheados.

        Args:
            total_chunks: Nº total de ventanas del SRT.

        Returns:
            ``True`` si no queda ningún chunk pendiente (con
            ``total_chunks <= 0`` se considera completo).
        """
        if total_chunks <= 0:
            return True
        return not self.pending(total_chunks)

    def clear(self) -> bool:
        """Borra el archivo de caché tras una traducción completa.

        Returns:
            ``True`` si se borró el archivo, ``False`` si no existía.
        """
        if self.path.is_file():
            self.path.unlink()
            return True
        return False
