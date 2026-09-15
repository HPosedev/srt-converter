"""Extracción de subtítulos y audio embebidos en MKV vía ffprobe/ffmpeg.

Responsabilidades:

- Verificar que ``ffmpeg`` y ``ffprobe`` existen en el ``PATH``.
- Listar pistas de subtítulos (idioma, forzados/completos) y de audio
  (idioma, codec, canales).
- Seleccionar pista (por idioma o por índice) cuando hay varias.
- Extraer la pista de subtítulo elegida a ``.srt`` con ``ffmpeg``.
- Exportar la pista de audio elegida a MP3 mono 16 kHz / 64 kbps para
  transcripción con Gemini (Files API).

No se hace OCR: las pistas de imagen (PGS/DVD) no son convertibles a
texto con ``-c:s srt`` y se rechazan con un error explicativo.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

FFPROBE_BIN = "ffprobe"
FFMPEG_BIN = "ffmpeg"

#: Codecs de subtítulo basados en texto, convertibles a SRT.
TEXT_SUBTITLE_CODECS = frozenset(
    {"subrip", "ass", "ssa", "mov_text", "webvtt", "srt", "ttml"}
)
#: Codecs de subtítulo basados en imagen (requieren OCR, fuera de alcance).
IMAGE_SUBTITLE_CODECS = frozenset(
    {"hdmv_pgs_subtitle", "dvd_subtitle", "dvd_sub", "xsub", "pgssub"}
)


class FFmpegNotFoundError(RuntimeError):
    """Falta ``ffmpeg`` o ``ffprobe`` en el sistema."""


@dataclass(frozen=True)
class SubtitleTrack:
    """Describe una pista de subtítulos dentro del MKV."""

    index: int
    """Índice global de stream en el contenedor (el ``index`` de ffprobe)."""
    sub_index: int
    """Posición 0-based solo entre subtítulos (``0:s:N`` de ffmpeg)."""
    language: str
    """Código de idioma (``und`` si el MKV no lo declara)."""
    title: str
    """Título/etiqueta de la pista (puede ser vacío)."""
    codec_name: str
    """Codec según ffprobe (p. ej. ``subrip``, ``hdmv_pgs_subtitle``)."""
    forced: bool
    """``True`` si la pista está marcada como forzada."""
    is_default: bool
    """``True`` si es la pista por defecto del contenedor."""

    @property
    def kind(self) -> str:
        """Clasificación humana: ``"forzados"`` o ``"completos"``."""
        return "forzados" if self.forced else "completos"

    def describe(self) -> str:
        """Devuelve una línea legible para listados CLI."""
        lang = self.language or "und"
        title = f" [{self.title}]" if self.title else ""
        flags: list[str] = [self.kind]
        if self.is_default:
            flags.append("default")
        return (
            f"s:{self.sub_index} (stream 0:{self.index}) "
            f"[{self.codec_name}] lang={lang}{title} ({', '.join(flags)})"
        )


@dataclass(frozen=True)
class AudioTrack:
    """Describe una pista de audio dentro del MKV."""

    index: int
    """Índice global de stream en el contenedor (para ``-map 0:<index>``)."""
    audio_index: int
    """Posición 0-based solo entre audios (``0:a:N`` de ffmpeg)."""
    lang: str
    """Código de idioma (``und`` si el MKV no lo declara)."""
    title: str
    """Título/etiqueta de la pista (puede ser vacío)."""
    codec: str
    """Codec según ffprobe (p. ej. ``aac``, ``ac3``, ``opus``)."""
    channels: int
    """Nº de canales (0 si ffprobe no lo informa)."""
    is_default: bool = False
    """``True`` si es la pista de audio por defecto del contenedor."""

    def describe(self) -> str:
        """Devuelve una línea legible para listados CLI."""
        lang = self.lang or "und"
        title = f" [{self.title}]" if self.title else ""
        flags = f", default" if self.is_default else ""
        return (
            f"a:{self.audio_index} (stream 0:{self.index}) "
            f"[{self.codec}, {self.channels}ch] lang={lang}{title}{flags}"
        )


def check_dependencies(
    ffprobe_bin: str = FFPROBE_BIN, ffmpeg_bin: str = FFMPEG_BIN
) -> None:
    """Verifica que ``ffprobe`` y ``ffmpeg`` estén disponibles.

    Args:
        ffprobe_bin: Binario de ffprobe a buscar.
        ffmpeg_bin: Binario de ffmpeg a buscar.

    Raises:
        FFmpegNotFoundError: Si falta alguno, con la orden de
            instalación sugerida para CachyOS/Arch.
    """
    missing = [
        name
        for name in (ffprobe_bin, ffmpeg_bin)
        if shutil.which(name) is None
    ]
    if missing:
        raise FFmpegNotFoundError(
            f"No se encontró en el PATH: {', '.join(missing)}. "
            "Instálalos con: sudo pacman -S ffmpeg "
            "(provee ffmpeg y ffprobe)."
        )


def _run_ffprobe_streams(mkv_path: Path, select_streams: str, entries: str) -> dict:
    """Ejecuta ffprobe y devuelve el JSON de streams seleccionados.

    Args:
        mkv_path: Ruta al archivo ``.mkv`` (debe existir).
        select_streams: Selector de ffprobe (``"s"`` subtítulos, ``"a"`` audio).
        entries: Campos de ``-show_entries stream=...``.

    Returns:
        Diccionario parseado de la salida ``{"streams": [...]}``.

    Raises:
        FFmpegNotFoundError: Si ``ffprobe`` no existe.
        FileNotFoundError: Si el ``.mkv`` no existe.
        RuntimeError: Si ffprobe falla o su salida no es JSON válido.
    """
    if not mkv_path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {mkv_path}")
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-select_streams",
        select_streams,
        "-show_entries",
        entries,
        "-of",
        "json",
        str(mkv_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError(
            "No se encontró `ffprobe` en el PATH. "
            "Instálalo con: sudo pacman -S ffmpeg."
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe falló para {mkv_path}: {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout or '{"streams": []}')
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Salida de ffprobe no es JSON válido: {exc}") from exc


def _run_ffprobe(mkv_path: Path) -> dict:
    """Ejecuta ffprobe y devuelve el JSON de streams de subtítulos.

    Args:
        mkv_path: Ruta al archivo ``.mkv``.

    Returns:
        Diccionario parseado de la salida ``{"streams": [...]}``.

    Raises:
        FFmpegNotFoundError: Si ``ffprobe`` no existe.
        FileNotFoundError: Si el ``.mkv`` no existe.
        RuntimeError: Si ffprobe falla o su salida no es JSON válido.
    """
    return _run_ffprobe_streams(
        mkv_path,
        "s",
        "stream=index,codec_name:stream_tags=language,title:"
        "stream_disposition=forced,default",
    )


def _run_ffprobe_audio(mkv_path: Path) -> dict:
    """Ejecuta ffprobe y devuelve el JSON de streams de audio.

    Args:
        mkv_path: Ruta al archivo ``.mkv``.

    Returns:
        Diccionario parseado de la salida ``{"streams": [...]}``.

    Raises:
        FFmpegNotFoundError: Si ``ffprobe`` no existe.
        FileNotFoundError: Si el ``.mkv`` no existe.
        RuntimeError: Si ffprobe falla o su salida no es JSON válido.
    """
    return _run_ffprobe_streams(
        mkv_path,
        "a",
        "stream=index,codec_name,channels:stream_tags=language,title:"
        "stream_disposition=default",
    )


def list_subtitle_tracks(mkv_path: str | Path) -> list[SubtitleTrack]:
    """Lista las pistas de subtítulos embebidas en un MKV.

    Muestra idioma y si son forzados/completos (vía :meth:`SubtitleTrack.kind`).

    Args:
        mkv_path: Ruta al archivo ``.mkv``.

    Returns:
        Pistas en orden de aparición. Vacía si el MKV no trae subtítulos.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        FFmpegNotFoundError: Si falta ffprobe.
        RuntimeError: Si ffprobe falla.
    """
    check_dependencies()
    data = _run_ffprobe(Path(mkv_path))
    tracks: list[SubtitleTrack] = []
    for sub_index, stream in enumerate(data.get("streams", [])):
        tags = stream.get("tags") or {}
        disp = stream.get("disposition") or {}
        tracks.append(
            SubtitleTrack(
                index=int(stream.get("index", sub_index)),
                sub_index=sub_index,
                language=str(tags.get("language", "und") or "und"),
                title=str(tags.get("title", "") or ""),
                codec_name=str(stream.get("codec_name", "unknown") or "unknown"),
                forced=bool(disp.get("forced", 0)),
                is_default=bool(disp.get("default", 0)),
            )
        )
    return tracks


def format_tracks_table(tracks: list[SubtitleTrack]) -> str:
    """Formatea la lista de pistas para mostrarla en el CLI.

    Args:
        tracks: Pistas obtenidas con :func:`list_subtitle_tracks`.

    Returns:
        Tabla multi-línea; mensaje alternativo si no hay pistas.
    """
    if not tracks:
        return "El MKV no contiene pistas de subtítulos."
    return "\n".join(f"  [{t.sub_index}] {t.describe()}" for t in tracks)


def select_track(
    tracks: list[SubtitleTrack],
    lang: str | None = None,
    sub_index: int | None = None,
) -> SubtitleTrack:
    """Selecciona una pista cuando el MKV trae varias.

    Prioridad: ``sub_index`` (explícito) > ``lang`` > única disponible.

    Args:
        tracks: Lista de pistas (no vacía).
        lang: Filtro de idioma (p. ej. ``"eng"``). Insensible a mayúsculas.
            También acepta ``"en"`` como prefijo de ``"eng"`` y viceversa.
        sub_index: Índice ``s:N`` (el ``sub_index``) elegido por el usuario.

    Returns:
        La pista seleccionada.

    Raises:
        ValueError: Si la lista está vacía, no hay coincidencia, o hay
            ambigüedad y hay que precisar ``sub_index``.
    """
    if not tracks:
        raise ValueError("El MKV no contiene pistas de subtítulos.")
    if sub_index is not None:
        for track in tracks:
            if track.sub_index == sub_index:
                return track
        valid = ", ".join(str(t.sub_index) for t in tracks)
        raise ValueError(
            f"No existe la pista s:{sub_index}. Índices válidos: {valid}."
        )
    if lang is None:
        if len(tracks) == 1:
            return tracks[0]
        raise ValueError(
            "El MKV tiene varias pistas; indica --lang o --track.\n"
            + format_tracks_table(tracks)
        )
    wanted = lang.strip().lower()
    matches = [
        t
        for t in tracks
        if t.language.lower() == wanted
        or t.language.lower().startswith(wanted)
        or wanted.startswith(t.language.lower())
    ]
    if not matches:
        raise ValueError(
            f"Ninguna pista con idioma {lang!r}.\n" + format_tracks_table(tracks)
        )
    if len(matches) > 1:
        raise ValueError(
            f"Idioma {lang!r} ambiguo ({len(matches)} pistas); "
            f"precisa --track.\n" + format_tracks_table(matches)
        )
    return matches[0]


def is_text_based(track: SubtitleTrack) -> bool:
    """Indica si la pista es convertible a SRT como texto.

    Args:
        track: Pista a evaluar.

    Returns:
        ``False`` para PGS/DVD (imagen, requieren OCR).
    """
    return track.codec_name not in IMAGE_SUBTITLE_CODECS


def list_audio_tracks(mkv_path: str | Path) -> list[AudioTrack]:
    """Lista las pistas de audio embebidas en un MKV.

    Args:
        mkv_path: Ruta al archivo ``.mkv``.

    Returns:
        Pistas en orden de aparición (``audio_index`` 0-based entre
        audios). Vacía si el contenedor no trae audio.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        FFmpegNotFoundError: Si falta ffprobe.
        RuntimeError: Si ffprobe falla.
    """
    check_dependencies()
    data = _run_ffprobe_audio(Path(mkv_path))
    tracks: list[AudioTrack] = []
    for audio_index, stream in enumerate(data.get("streams", [])):
        tags = stream.get("tags") or {}
        disp = stream.get("disposition") or {}
        try:
            channels = int(stream.get("channels", 0) or 0)
        except (TypeError, ValueError):
            channels = 0
        tracks.append(
            AudioTrack(
                index=int(stream.get("index", audio_index)),
                audio_index=audio_index,
                lang=str(tags.get("language", "und") or "und"),
                title=str(tags.get("title", "") or ""),
                codec=str(stream.get("codec_name", "unknown") or "unknown"),
                channels=channels,
                is_default=bool(disp.get("default", 0)),
            )
        )
    return tracks


def format_audio_tracks_table(tracks: list[AudioTrack]) -> str:
    """Formatea la lista de pistas de audio para mostrarla en el CLI.

    Args:
        tracks: Pistas obtenidas con :func:`list_audio_tracks`.

    Returns:
        Tabla multi-línea; mensaje alternativo si no hay pistas.
    """
    if not tracks:
        return "El MKV no contiene pistas de audio."
    return "\n".join(f"  [{t.audio_index}] {t.describe()}" for t in tracks)


def _matches_lang(track_lang: str, wanted: str) -> bool:
    """Compara idiomas de forma tolerante (``en`` ≃ ``eng``).

    Args:
        track_lang: Idioma declarado por la pista.
        wanted: Idioma pedido por el usuario.

    Returns:
        ``True`` si coinciden exacto o por prefijo en ambos sentidos.
    """
    have = track_lang.strip().lower()
    want = wanted.strip().lower()
    return have == want or have.startswith(want) or want.startswith(have)


def select_audio_track(
    tracks: list[AudioTrack],
    track_index: int | None = None,
    lang: str | None = None,
) -> AudioTrack:
    """Selecciona una pista de audio del contenedor.

    Prioridad: índice explícito > coincidencia de idioma > pista
    predeterminada > primera pista.

    Args:
        tracks: Lista de pistas (no vacía).
        track_index: Índice ``a:N`` (el ``audio_index``) elegido.
        lang: Filtro de idioma (p. ej. ``"eng"`` o ``"en"``).

    Returns:
        La pista seleccionada.

    Raises:
        ValueError: Si no hay pistas, el índice no existe, o ningún
            idioma coincide (se muestra la tabla disponible).
    """
    if not tracks:
        raise ValueError(
            "El MKV no contiene pistas de audio; no se puede extraer sonido."
        )
    if track_index is not None:
        for track in tracks:
            if track.audio_index == track_index:
                return track
        valid = ", ".join(str(t.audio_index) for t in tracks)
        raise ValueError(
            f"No existe la pista de audio a:{track_index}. "
            f"Índices válidos: {valid}."
        )
    if lang is not None:
        matches = [t for t in tracks if _matches_lang(t.lang, lang)]
        if not matches:
            raise ValueError(
                f"Ninguna pista de audio con idioma {lang!r}.\n"
                + format_audio_tracks_table(tracks)
            )
        for track in matches:
            if track.is_default:
                return track
        return matches[0]
    for track in tracks:
        if track.is_default:
            return track
    return tracks[0]


def extract_subtitle(
    mkv_path: str | Path,
    output: str | Path,
    track: SubtitleTrack | int,
    overwrite: bool = False,
) -> Path:
    """Extrae una pista de subtítulos a archivo ``.srt`` con ffmpeg.

    Args:
        mkv_path: Origen ``.mkv``.
        output: Destino ``.srt`` (debe terminar en ``.srt``).
        track: :class:`SubtitleTrack` o índice global de stream
            (``SubtitleTrack.index``) para ``-map 0:<index>``.
        overwrite: Si ``False`` (defecto) falla cuando el destino existe.

    Returns:
        Ruta del ``.srt`` generado.

    Raises:
        FileNotFoundError: Si el ``.mkv`` no existe.
        FFmpegNotFoundError: Si falta ffmpeg.
        FileExistsError: Si el destino existe y ``overwrite=False``.
        ValueError: Si el destino no es ``.srt`` o la pista es de imagen
            (PGS/DVD, necesita OCR).
        RuntimeError: Si ffmpeg devuelve error.
    """
    check_dependencies()
    src = Path(mkv_path)
    dst = Path(output)
    if not src.is_file():
        raise FileNotFoundError(f"No existe el archivo: {src}")
    if dst.suffix.lower() != ".srt":
        raise ValueError(f"El destino debe terminar en .srt, recibido: {dst}")
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"Ya existe {dst}; usa overwrite=True o elige otra ruta."
        )
    stream_index = track.index if isinstance(track, SubtitleTrack) else int(track)
    if isinstance(track, SubtitleTrack) and not is_text_based(track):
        raise ValueError(
            f"La pista s:{track.sub_index} es '{track.codec_name}' (imagen: "
            "PGS/DVD). ffmpeg no puede convertirla a texto sin OCR; "
            "elige una pista de texto (subrip/ass/ssa/mov_text)."
        )
    if dst.parent != Path("") and str(dst.parent):
        dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-y" if overwrite else "-n",
        "-i",
        str(src),
        "-map",
        f"0:{stream_index}",
        "-c:s",
        "srt",
        str(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError(
            "No se encontró `ffmpeg` en el PATH. "
            "Instálalo con: sudo pacman -S ffmpeg."
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falló: {(proc.stderr or '').strip()[-2000:]}")
    return dst


def extract_audio(
    mkv_path: str | Path,
    out_path: str | Path,
    audio_index: int = 0,
    overwrite: bool = False,
    track: AudioTrack | None = None,
    stream_index: int | None = None,
) -> Path:
    """Exporta una pista de audio del MKV a MP3 mono 16 kHz / 64 kbps.

    Formato pensado para la Files API de Gemini: poco peso, suficiente
    para transcripción/traducción. Usa ``-ac 1 -ar 16000 -b:a 64k
    -c:a libmp3lame``.

    La pista se elige con prioridad: ``track`` > ``stream_index`` >
    ``audio_index``. Con ``track``/``stream_index`` se mapea por índice
    global (``-map 0:<index>``); con ``audio_index`` por posición entre
    audios (``-map 0:a:<N>``).

    Args:
        mkv_path: Origen ``.mkv``.
        out_path: Destino ``.mp3``.
        audio_index: Índice 0-based entre pistas de audio (``0:a:N``).
            Ignorado si se pasa ``track`` o ``stream_index``.
        overwrite: Si ``False`` (defecto) falla cuando el destino existe.
        track: Pista elegida con :func:`select_audio_track` (opcional).
        stream_index: Índice global de stream (opcional).

    Returns:
        Ruta del ``.mp3`` generado.

    Raises:
        FileNotFoundError: Si el ``.mkv`` no existe.
        FFmpegNotFoundError: Si falta ffmpeg.
        FileExistsError: Si el destino existe y ``overwrite=False``.
        ValueError: Si el destino no termina en ``.mp3`` o el índice
            de audio es negativo.
        RuntimeError: Si ffmpeg devuelve error (p. ej. sin pista de audio).
    """
    check_dependencies()
    src = Path(mkv_path)
    dst = Path(out_path)
    if not src.is_file():
        raise FileNotFoundError(f"No existe el archivo: {src}")
    if dst.suffix.lower() != ".mp3":
        raise ValueError(f"El destino debe terminar en .mp3, recibido: {dst}")
    if audio_index < 0:
        raise ValueError(f"audio_index debe ser >= 0, recibido: {audio_index}")
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"Ya existe {dst}; usa overwrite=True o elige otra ruta."
        )
    if track is not None:
        map_arg = f"0:{track.index}"
    elif stream_index is not None:
        map_arg = f"0:{int(stream_index)}"
    else:
        map_arg = f"0:a:{audio_index}"
    if dst.parent != Path("") and str(dst.parent):
        dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-y" if overwrite else "-n",
        "-i",
        str(src),
        "-map",
        map_arg,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "64k",
        "-c:a",
        "libmp3lame",
        str(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError(
            "No se encontró `ffmpeg` en el PATH. "
            "Instálalo con: sudo pacman -S ffmpeg."
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falló: {(proc.stderr or '').strip()[-2000:]}")
    return dst
