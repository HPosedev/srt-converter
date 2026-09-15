"""Entrypoint CLI (Typer): ``extract``, ``translate`` y ``auto``.

- ``extract``: lista pistas de subtítulos/audio de un MKV, o extrae una
  a ``.srt`` (subtítulos) o ``.mp3`` (audio con ``--audio``).
- ``translate``: traduce un ``.srt`` (o carpeta con ``--batch``) con
  Gemini, con caché de reanudación y auditoría de legibilidad final.
- ``auto``: flujo integral MKV → ``*_es.srt`` (extrae subtítulo y lo
  traduce; con ``--from-audio`` transcribe la pista de audio elegida
  con ``--audio-lang``/``--audio-track``).
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from config import AppConfig, load_config
from extractor import (
    AudioTrack,
    FFmpegNotFoundError,
    extract_audio,
    extract_subtitle,
    format_audio_tracks_table,
    format_tracks_table,
    is_text_based,
    list_audio_tracks,
    list_subtitle_tracks,
    select_audio_track,
    select_track,
)
from srt_utils import audit_subtitles, write_srt_file
from translator import GeminiSubtitler, translate_file

app = typer.Typer(help="Extrae y traduce subtítulos MKV/SRT (EN->ES con Gemini).")


def build_subtitler(cfg: AppConfig, window_size: int | None = None) -> GeminiSubtitler:
    """Construye el traductor desde la configuración efectiva.

    Args:
        cfg: Configuración cargada del TOML/entorno.
        window_size: Override de la ventana (CLI); ``None`` usa la config.

    Returns:
        Instancia lista de :class:`GeminiSubtitler`.

    Raises:
        typer.BadParameter: Si no hay API key de Gemini.
    """
    if not cfg.gemini.api_key:
        raise typer.BadParameter(
            "Falta la API key de Gemini: define [gemini].api_key en config.toml "
            "o la variable GEMINI_API_KEY."
        )
    return GeminiSubtitler(
        api_key=cfg.gemini.api_key,
        model=cfg.gemini.model,
        glossary=cfg.glossary,
        src_lang=cfg.src_lang,
        dst_lang=cfg.dst_lang,
        window_size=window_size or cfg.window_size,
        style_instructions=cfg.style_instructions,
    )


def report_warnings(warnings: list, where: str = "") -> None:
    """Muestra los avisos de auditoría por stderr, con tabla Rich.

    Args:
        warnings: Avisos de ``audit_subtitles`` (solo avisos, nada se modifica).
        where: Etiqueta opcional del archivo auditado.
    """
    if not warnings:
        typer.echo(f"Sin avisos de legibilidad{(' en ' + where) if where else ''}.")
        return
    prefix = f" [{where}]" if where else ""
    typer.echo(f"{len(warnings)} aviso(s) de legibilidad{prefix}:", err=True)
    by_kind: dict[str, int] = {}
    for warning in warnings:
        by_kind[getattr(warning, "kind", "?")] = by_kind.get(getattr(warning, "kind", "?"), 0) + 1
    typer.echo(
        "Resumen: " + ", ".join(f"{kind}: {count}" for kind, count in sorted(by_kind.items())),
        err=True,
    )
    table = Table(title=None, show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Tipo")
    table.add_column("Medido", justify="right")
    table.add_column("Límite", justify="right")
    table.add_column("Detalle")
    for warning in warnings[:25]:
        kind = str(getattr(warning, "kind", "?"))
        value = getattr(warning, "value", "?")
        limit = getattr(warning, "limit", "?")
        if kind == "cps":
            measured = f"{float(value):.1f} cps" if isinstance(value, (int, float)) else str(value)
            bound = f"{float(limit):.1f} cps" if isinstance(limit, (int, float)) else str(limit)
        else:
            measured = f"{int(value)} chars" if isinstance(value, (int, float)) else str(value)
            bound = f"{int(limit)} chars" if isinstance(limit, (int, float)) else str(limit)
        detail = str(getattr(warning, "detail", ""))[:60]
        table.add_row(str(getattr(warning, "index", "?")), kind, measured, bound, detail)
    if len(warnings) > 25:
        table.add_row("…", f"+{len(warnings) - 25} más", "", "", "")
    Console(stderr=True).print(table)


def iter_batch_sources(directory: Path) -> list[Path]:
    """Lista los ``.srt`` de una carpeta, excluyendo salidas ``*_es.srt``.

    Args:
        directory: Carpeta a recorrer (no recursivo).

    Returns:
        Rutas ordenadas listas para traducir.
    """
    return sorted(
        p
        for p in directory.glob("*.srt")
        if p.is_file() and not p.stem.endswith("_es")
    )


AUTO_VIDEO_EXTS = (".mkv", ".mp4")


def iter_auto_sources(directory: Path) -> list[Path]:
    """Lista los vídeos de una carpeta para el flujo ``auto``.

    Args:
        directory: Carpeta a recorrer (no recursivo).

    Returns:
        Vídeos ``.mkv``/``.mp4`` ordenados.
    """
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in AUTO_VIDEO_EXTS
    )


def expected_final(video: Path) -> Path:
    """Calcula el destino ``*_es.srt`` de un vídeo.

    Args:
        video: Archivo de vídeo origen.

    Returns:
        Ruta junto al vídeo con sufijo ``_es.srt``.
    """
    return video.with_name(video.stem + "_es.srt")


def auto_temp_paths(final: Path) -> tuple[Path, Path]:
    """Devuelve los temporales asociados a un destino final.

    Args:
        final: Destino ``*_es.srt`` del flujo ``auto``.

    Returns:
        Tupla ``(intermedio .extracted.srt, audio .auto-audio.mp3)``.
    """
    return (
        final.with_name(final.stem + ".extracted.srt"),
        final.with_name(final.stem + ".auto-audio.mp3"),
    )


def cleanup_temps(*paths: Path) -> None:
    """Elimina archivos temporales si existen (sin fallar si no).

    Args:
        paths: Rutas a borrar. La caché de traducción (``.subtrans-cache.json``)
            NO se toca nunca aquí: es lo que permite reanudar.
    """
    for path in paths:
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass


def run_translate_single(
    source: Path,
    output: Path | None,
    subtitler: GeminiSubtitler,
    max_chars: int,
    max_cps: float,
    use_cache: bool,
) -> Path:
    """Traduce un SRT, informa avisos y devuelve el destino.

    Args:
        source: SRT origen.
        output: Destino o ``None`` para derivar ``*_es.srt``.
        subtitler: Traductor configurado.
        max_chars: Límite de caracteres por línea (auditoría).
        max_cps: Límite de velocidad lectora (auditoría).
        use_cache: Reanudar con caché por ventana.

    Returns:
        Ruta del SRT traducido.
    """
    dst, warnings = translate_file(
        source, output, subtitler,
        max_chars=max_chars, max_cps=max_cps, use_cache=use_cache,
    )
    typer.echo(f"Traducido → {dst}")
    for notice in getattr(subtitler, "fallback_notices", None) or []:
        typer.echo(f"Aviso de traducción: {notice}", err=True)
    report_warnings(warnings, where=str(dst))
    return dst


def choose_audio_track(
    mkv: Path, audio_lang: str | None, audio_track: int | None
) -> AudioTrack:
    """Lista el audio del MKV y selecciona pista con fallback amable.

    Args:
        mkv: Archivo ``.mkv`` de origen.
        audio_lang: Idioma preferido (``None`` = pista predeterminada).
        audio_track: Índice ``a:N`` explícito (prioridad máxima).

    Returns:
        Pista de audio elegida. Si el filtro de idioma no coincide pero
        hay una única pista, se usa esa con un aviso en vez de fallar.

    Raises:
        typer.BadParameter: Sin pistas de audio o sin coincidencia.
    """
    try:
        tracks = list_audio_tracks(mkv)
    except FFmpegNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not tracks:
        raise typer.BadParameter(
            f"{mkv} no trae pistas de audio; no se puede usar --from-audio."
        )
    try:
        return select_audio_track(tracks, track_index=audio_track, lang=audio_lang)
    except ValueError as exc:
        if audio_track is None and len(tracks) == 1:
            typer.echo(
                f"Aviso: {exc} Se usa la única pista disponible: "
                f"{tracks[0].describe()}",
                err=True,
            )
            return tracks[0]
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def extract(
    mkv: Path = typer.Argument(..., help="Archivo .mkv de origen."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Destino .srt (o .mp3 con --audio)."),
    lang: str | None = typer.Option(None, "--lang", help="Idioma de subtítulo (p. ej. eng)."),
    track: int | None = typer.Option(None, "--track", help="Índice s:N si hay varias."),
    list_only: bool = typer.Option(False, "--list", help="Solo listar pistas de subtítulos."),
    list_audio: bool = typer.Option(False, "--list-audio", help="Solo listar pistas de audio."),
    audio: bool = typer.Option(False, "--audio", help="Extraer audio a .mp3 en vez de subtítulos."),
    audio_lang: str | None = typer.Option(None, "--audio-lang", help="Idioma de audio (p. ej. eng)."),
    audio_track: int | None = typer.Option(None, "--audio-track", help="Índice a:N de audio."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Sobrescribir destino."),
) -> None:
    """Lista pistas (--list/--list-audio) o extrae subtítulo/audio."""
    if list_audio:
        try:
            typer.echo(format_audio_tracks_table(list_audio_tracks(mkv)))
        except FFmpegNotFoundError as exc:
            raise typer.BadParameter(str(exc)) from exc
        return
    if audio:
        chosen_audio = choose_audio_track(mkv, audio_lang, audio_track)
        typer.echo(f"Pista de audio elegida: {chosen_audio.describe()}")
        dest = output if output is not None else mkv.with_name(mkv.stem + ".mp3")
        result = extract_audio(mkv, dest, track=chosen_audio, overwrite=overwrite)
        typer.echo(f"Audio extraído → {result}")
        return
    try:
        tracks = list_subtitle_tracks(mkv)
    except FFmpegNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if list_only or output is None:
        typer.echo(format_tracks_table(tracks))
        if output is None and not list_only:
            raise typer.BadParameter("Indica --output o usa --list.")
        return
    chosen = select_track(tracks, lang=lang, sub_index=track)
    typer.echo(f"Pista elegida: {chosen.describe()}")
    result = extract_subtitle(mkv, output, chosen, overwrite=overwrite)
    typer.echo(f"Extraído → {result}")


@app.command()
def translate(
    source: Path = typer.Argument(..., help="Archivo .srt o carpeta con --batch."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Destino (defecto *_es.srt)."),
    batch: bool = typer.Option(False, "--batch", help="Traducir carpeta completa."),
    config: Path | None = typer.Option(None, "--config", help="TOML de configuración."),
    window_size: int | None = typer.Option(None, "--window-size", help="Ventana (override config)."),
    max_chars: int | None = typer.Option(None, "--max-chars", help="Límite chars/línea (auditoría)."),
    max_cps: float | None = typer.Option(None, "--max-cps", help="Límite chars/seg (auditoría)."),
    no_cache: bool = typer.Option(False, "--no-cache", help="No usar caché de reanudación."),
) -> None:
    """Traduce SRT a castellano con Gemini (con caché y auditoría final)."""
    cfg = load_config(config)
    subtitler = build_subtitler(cfg, window_size=window_size)
    chars = max_chars if max_chars is not None else cfg.max_chars
    cps = max_cps if max_cps is not None else cfg.max_cps
    use_cache = not no_cache

    if batch:
        if not source.is_dir():
            raise typer.BadParameter(f"--batch requiere una carpeta: {source}")
        if output is not None:
            raise typer.BadParameter("--output no es compatible con --batch.")
        sources = iter_batch_sources(source)
        if not sources:
            typer.echo(f"Sin archivos .srt en {source}.")
            return
        failures = 0
        for item in sources:
            try:
                run_translate_single(item, None, subtitler, chars, cps, use_cache)
            except Exception as exc:  # noqa: BLE001 — batch sigue con el resto
                failures += 1
                typer.echo(f"Error en {item}: {exc}", err=True)
        typer.echo(f"Hechos {len(sources) - failures}/{len(sources)} archivos.")
        if failures:
            raise typer.Exit(code=1)
        return

    if not source.is_file():
        raise typer.BadParameter(f"No existe el archivo: {source}")
    run_translate_single(source, output, subtitler, chars, cps, use_cache)


@app.command()
def auto(
    source: Path = typer.Argument(..., help="Archivo de vídeo o carpeta (lote)."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Destino *_es.srt (solo archivo único)."),
    lang: str | None = typer.Option(None, "--lang", help="Idioma de subtítulo (p. ej. eng)."),
    track: int | None = typer.Option(None, "--track", help="Índice s:N si hay varias."),
    from_audio: bool = typer.Option(False, "--from-audio", help="Transcribir audio con Gemini."),
    audio_lang: str = typer.Option("eng", "--audio-lang", help="Idioma de audio con --from-audio."),
    audio_track: int | None = typer.Option(None, "--audio-track", help="Índice a:N de audio (prioridad máxima)."),
    keep_intermediate: bool = typer.Option(False, "--keep-intermediate", help="Conservar .srt/.mp3 intermedios."),
    config: Path | None = typer.Option(None, "--config", help="TOML de configuración."),
    window_size: int | None = typer.Option(None, "--window-size", help="Ventana (override config)."),
    max_chars: int | None = typer.Option(None, "--max-chars", help="Límite chars/línea (auditoría)."),
    max_cps: float | None = typer.Option(None, "--max-cps", help="Límite chars/seg (auditoría)."),
    no_cache: bool = typer.Option(False, "--no-cache", help="No usar caché de reanudación."),
) -> None:
    """Flujo integral: vídeo(s) → *_es.srt (vía subtítulo o vía audio).

    Acepta un archivo o una carpeta: en modo carpeta procesa todos los
    ``.mkv``/``.mp4``, omite los que ya tengan su ``*_es.srt`` y sigue
    con el siguiente si alguno falla. Ctrl+C limpia los temporales
    (``.extracted.srt``/``.mp3``) antes de abortar; la caché de
    reanudación se conserva a propósito.
    """
    cfg = load_config(config)
    subtitler = build_subtitler(cfg, window_size=window_size)
    chars = max_chars if max_chars is not None else cfg.max_chars
    cps = max_cps if max_cps is not None else cfg.max_cps
    use_cache = not no_cache

    if source.is_dir():
        if output is not None:
            raise typer.BadParameter("--output no es compatible con directorios.")
        videos = iter_auto_sources(source)
        if not videos:
            typer.echo(f"Sin vídeos .mkv/.mp4 en {source}.")
            return
        failures = 0
        skipped = 0
        done = 0
        for video in videos:
            final = expected_final(video)
            if final.is_file():
                skipped += 1
                typer.echo(f"Omitido {video.name} (ya existe {final.name}).")
                continue
            try:
                run_auto_single(
                    video, final, subtitler=subtitler, lang=lang, track=track,
                    from_audio=from_audio, audio_lang=audio_lang,
                    audio_track=audio_track,
                    keep_intermediate=keep_intermediate, chars=chars, cps=cps,
                    use_cache=use_cache,
                )
                done += 1
            except KeyboardInterrupt:
                cleanup_temps(*auto_temp_paths(final))
                typer.echo("\nInterrumpido: temporales eliminados.", err=True)
                raise typer.Exit(code=130)
            except Exception as exc:  # noqa: BLE001 — el lote sigue con el resto
                failures += 1
                typer.echo(f"Error en {video.name}: {exc}", err=True)
        typer.echo(f"Hechos {done}/{len(videos)} vídeos ({skipped} omitidos).")
        if failures:
            raise typer.Exit(code=1)
        return

    if not source.is_file():
        raise typer.BadParameter(f"No existe: {source}")
    final = output if output is not None else expected_final(source)
    try:
        run_auto_single(
            source, final, subtitler=subtitler, lang=lang, track=track,
            from_audio=from_audio, audio_lang=audio_lang, audio_track=audio_track,
            keep_intermediate=keep_intermediate, chars=chars, cps=cps,
            use_cache=use_cache,
        )
    except KeyboardInterrupt:
        cleanup_temps(*auto_temp_paths(final))
        typer.echo("\nInterrumpido: temporales eliminados.", err=True)
        raise typer.Exit(code=130)


@app.command(hidden=True)
def gui() -> None:
    """Lanza la interfaz gráfica (requiere PySide6)."""
    try:
        from gui import HAS_PYSIDE6, main as gui_main
        if not HAS_PYSIDE6:
            raise ImportError("PySide6 no disponible")
    except ImportError:
        raise typer.BadParameter(
            "PySide6 no está instalado. Instálalo con:\n"
            "  pip install subtitle-translator[gui]"
        ) from None
    gui_main()


def run_auto_single(
    video: Path,
    final: Path,
    *,
    subtitler: GeminiSubtitler,
    lang: str | None,
    track: int | None,
    from_audio: bool,
    audio_lang: str,
    audio_track: int | None,
    keep_intermediate: bool,
    chars: int,
    cps: float,
    use_cache: bool,
) -> Path:
    """Ejecuta el flujo integral para un único vídeo.

    Args:
        video: Archivo de vídeo origen.
        final: Destino ``*_es.srt``.
        subtitler: Traductor configurado.
        lang: Idioma de subtítulo preferido.
        track: Índice ``s:N`` explícito.
        from_audio: Transcribir audio en vez de extraer subtítulo.
        audio_lang: Idioma de audio preferido.
        audio_track: Índice ``a:N`` explícito.
        keep_intermediate: Conservar ``.srt``/``.mp3`` intermedios.
        chars: Límite chars/línea (auditoría).
        cps: Límite chars/seg (auditoría).
        use_cache: Reanudar con caché por ventana.

    Returns:
        Ruta del ``*_es.srt`` generado.

    Raises:
        typer.BadParameter: Sin pistas utilizables.
    """
    if from_audio:
        chosen_audio = choose_audio_track(video, audio_lang, audio_track)
        typer.echo(f"Pista de audio elegida: {chosen_audio.describe()}")
        _, mp3 = auto_temp_paths(final)
        typer.echo(f"Extrayendo audio → {mp3}")
        extract_audio(video, mp3, track=chosen_audio, overwrite=True)
        try:
            subtitles = subtitler.transcribe_and_translate_audio(mp3)
        finally:
            if not keep_intermediate:
                cleanup_temps(mp3)
        warnings = audit_subtitles(subtitles, max_chars=chars, max_cps=cps)
        write_srt_file(final, subtitles)
        typer.echo(f"Traducido → {final}")
        report_warnings(warnings, where=str(final))
        return final

    try:
        tracks = list_subtitle_tracks(video)
    except FFmpegNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not tracks:
        raise typer.BadParameter(
            f"{video} no trae pistas de subtítulos; reintenta con --from-audio."
        )
    chosen = select_track(tracks, lang=lang, sub_index=track)
    if not is_text_based(chosen):
        raise typer.BadParameter(
            f"La pista elegida es '{chosen.codec_name}' (imagen, sin OCR); "
            "reintenta con --from-audio o elige una pista de texto."
        )
    typer.echo(f"Pista elegida: {chosen.describe()}")
    intermediate, _mp3 = auto_temp_paths(final)
    extract_subtitle(video, intermediate, chosen, overwrite=True)
    try:
        run_translate_single(intermediate, final, subtitler, chars, cps, use_cache)
    finally:
        if not keep_intermediate:
            cleanup_temps(intermediate)
    return final


def main() -> None:
    """Entrypoint para ``project.scripts``."""
    app()


if __name__ == "__main__":
    main()
