"""Interfaz gráfica de escritorio con PySide6 para subtitle-translator.

Proporciona una ventana principal con:
- Zona de arrastrar y soltar para archivos .mkv, .mp4 o .srt.
- Selectores de pistas de subtítulo y audio.
- Selector de modo: Automático, Solo extraer o Desde audio.
- Ajustes: idioma destino, ventana de contexto, mantener caché.
- Barra de progreso y etiqueta de estado.
- Visor de registro/auditoría.
- Botones de iniciar y cancelar.
- Ejecución asíncrona con QThread.

Uso::

    pip install subtitle-translator[gui]
    subtrans-gui

O desde la CLI::

    subtrans gui
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    from PySide6.QtCore import QEvent, Qt, QThread, Signal, Slot
    from PySide6.QtGui import QDragEnterEvent, QDropEvent
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QSpinBox,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False

from config import AppConfig, load_config
from extractor import (
    AudioTrack,
    FFmpegNotFoundError,
    SubtitleTrack,
    extract_audio,
    extract_subtitle,
    list_audio_tracks,
    list_subtitle_tracks,
    select_audio_track,
    select_track,
)
from srt_utils import audit_subtitles, write_srt_file
from translator import GeminiSubtitler, translate_file

SUPPORTED_VIDEO_EXTS = {".mkv", ".mp4"}
SUPPORTED_SRT_EXTS = {".srt"}


# ---------------------------------------------------------------------------
# Worker (QThread) – ejecuta extracción/traducción sin bloquear la UI
# ---------------------------------------------------------------------------


class Worker(QThread):
    """Hilo que ejecuta la lógica pesada emitiendo señales a la UI."""

    progress = Signal(int, int)  # (current, total)
    text_log = Signal(str)
    completed = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        source: Path,
        mode: str,
        *,
        subtitle_track_index: int | None = None,
        audio_track_index: int | None = None,
        audio_lang: str = "eng",
        dst_lang: str = "es",
        window_size: int = 50,
        use_cache: bool = True,
        config_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.source = source
        self.mode = mode
        self.subtitle_track_index = subtitle_track_index
        self.audio_track_index = audio_track_index
        self.audio_lang = audio_lang
        self.dst_lang = dst_lang
        self.window_size = window_size
        self.use_cache = use_cache
        self.config_path = config_path
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: C901 – dispatched by mode
        try:
            if self.mode == "auto":
                self._run_auto()
            elif self.mode == "extract":
                self._run_extract()
            elif self.mode == "translate":
                self._run_translate()
            elif self.mode == "from_audio":
                self._run_from_audio()
            else:
                self.error.emit(f"Modo desconocido: {self.mode}")
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    # -- helpers ------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.text_log.emit(msg)

    def _update_progress(self, current: int, total: int) -> None:
        self.progress.emit(current, total)

    # -- auto mode ----------------------------------------------------------

    def _run_auto(self) -> None:
        src = self.source
        final = src.with_name(src.stem + "_es.srt")
        cfg = load_config(self.config_path)
        self._log(f"Cargando configuración…")
        subtitler = self._build_subtitler(cfg)
        if not subtitler:
            return

        self._log(f"Extrayendo subtítulos de {src.name}…")
        self._update_progress(0, 3)
        tracks = list_subtitle_tracks(src)
        if not tracks:
            self.error.emit(f"{src} no tiene pistas de subtítulos. Prueba con --from-audio.")
            return

        chosen = select_track(tracks, sub_index=self.subtitle_track_index)
        self._log(f"Pista elegida: {chosen.describe()}")
        intermediate = final.with_name(final.stem + ".extracted.srt")
        extract_subtitle(src, intermediate, chosen, overwrite=True)
        self._update_progress(1, 3)

        self._log("Traduciendo…")
        self._update_progress(2, 3)
        chars = cfg.max_chars
        cps = cfg.max_cps
        try:
            dst, warnings = translate_file(
                intermediate, final, subtitler,
                max_chars=chars, max_cps=cps, use_cache=self.use_cache,
            )
        finally:
            if intermediate.is_file():
                try:
                    intermediate.unlink()
                except OSError:
                    pass
        self._update_progress(3, 3)

        for notice in getattr(subtitler, "fallback_notices", None) or []:
            self._log(f"Aviso de traducción: {notice}")
        if warnings:
            self._log(f"{len(warnings)} aviso(s) de legibilidad en {dst}")
        self.completed.emit(str(dst))

    # -- extract mode -------------------------------------------------------

    def _run_extract(self) -> None:
        src = self.source
        self._log(f"Listando pistas de {src.name}…")
        self._update_progress(0, 2)
        tracks = list_subtitle_tracks(src)
        if not tracks:
            self.error.emit(f"{src} no tiene pistas de subtítulos.")
            return

        chosen = select_track(tracks, sub_index=self.subtitle_track_index)
        self._log(f"Pista elegida: {chosen.describe()}")
        dest = src.with_name(src.stem + ".srt")
        extract_subtitle(src, dest, chosen, overwrite=True)
        self._update_progress(1, 2)
        self._update_progress(2, 2)
        self.completed.emit(str(dest))

    # -- translate mode -----------------------------------------------------

    def _run_translate(self) -> None:
        src = self.source
        final = src.with_name(src.stem + "_es.srt")
        cfg = load_config(self.config_path)
        self._log("Cargando configuración…")
        subtitler = self._build_subtitler(cfg)
        if not subtitler:
            return

        self._log(f"Traduciendo {src.name}…")
        self._update_progress(0, 2)
        chars = cfg.max_chars
        cps = cfg.max_cps
        self._update_progress(1, 2)
        dst, warnings = translate_file(
            src, final, subtitler,
            max_chars=chars, max_cps=cps, use_cache=self.use_cache,
        )
        self._update_progress(2, 2)

        for notice in getattr(subtitler, "fallback_notices", None) or []:
            self._log(f"Aviso de traducción: {notice}")
        if warnings:
            self._log(f"{len(warnings)} aviso(s) de legibilidad en {dst}")
        self.completed.emit(str(dst))

    # -- from audio mode ----------------------------------------------------

    def _run_from_audio(self) -> None:
        src = self.source
        final = src.with_name(src.stem + "_es.srt")
        cfg = load_config(self.config_path)
        self._log(f"Cargando configuración…")
        subtitler = self._build_subtitler(cfg)
        if not subtitler:
            return

        self._log(f"Buscando pistas de audio en {src.name}…")
        self._update_progress(0, 3)
        audio_tracks = list_audio_tracks(src)
        if not audio_tracks:
            self.error.emit(f"{src} no tiene pistas de audio.")
            return

        try:
            chosen_audio = select_audio_track(
                audio_tracks, track_index=self.audio_track_index, lang=self.audio_lang
            )
        except ValueError as exc:
            if self.audio_track_index is None and len(audio_tracks) == 1:
                chosen_audio = audio_tracks[0]
                self._log(f"Aviso: {exc} Se usa la única pista disponible: {chosen_audio.describe()}")
            else:
                self.error.emit(str(exc))
                return

        self._log(f"Pista de audio elegida: {chosen_audio.describe()}")
        mp3 = final.with_name(final.stem + ".auto-audio.mp3")
        self._log(f"Extrayendo audio → {mp3}")
        extract_audio(src, mp3, track=chosen_audio, overwrite=True)
        self._update_progress(1, 3)

        self._log("Transcribiendo y traduciendo con Gemini…")
        self._update_progress(2, 3)
        try:
            subtitles = subtitler.transcribe_and_translate_audio(mp3)
        finally:
            if mp3.is_file():
                try:
                    mp3.unlink()
                except OSError:
                    pass
        chars = cfg.max_chars
        cps = cfg.max_cps
        warnings = audit_subtitles(subtitles, max_chars=chars, max_cps=cps)
        write_srt_file(final, subtitles)
        self._update_progress(3, 3)

        for notice in getattr(subtitler, "fallback_notices", None) or []:
            self._log(f"Aviso de traducción: {notice}")
        if warnings:
            self._log(f"{len(warnings)} aviso(s) de legibilidad en {final}")
        self.completed.emit(str(final))

    # -- helpers ------------------------------------------------------------

    def _build_subtitler(self, cfg: AppConfig) -> GeminiSubtitler | None:
        if not cfg.gemini.api_key:
            self.error.emit(
                "Falta la API key de Gemini: define [gemini].api_key en config.toml "
                "o la variable GEMINI_API_KEY."
            )
            return None
        return GeminiSubtitler(
            api_key=cfg.gemini.api_key,
            model=cfg.gemini.model,
            glossary=cfg.glossary,
            src_lang=cfg.src_lang,
            dst_lang=self.dst_lang or cfg.dst_lang,
            window_size=self.window_size or cfg.window_size,
            style_instructions=cfg.style_instructions,
        )


# ---------------------------------------------------------------------------
# Ventana principal
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Ventana principal de SubTrans GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SubTrans – Traductor de subtítulos")
        self.resize(850, 680)
        self.setMinimumSize(780, 560)
        self._worker: Worker | None = None
        self._subtitle_tracks: list[SubtitleTrack] = []
        self._audio_tracks: list[AudioTrack] = []
        self._loaded_path: Path | None = None
        self._build_ui()

    # -- construcción de la UI ----------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(14, 14, 14, 14)

        # --- Zona de arrastrar y soltar ---
        drop_group = QGroupBox("Archivo de origen")
        drop_layout = QVBoxLayout(drop_group)
        drop_layout.setSpacing(8)
        drop_layout.setContentsMargins(10, 14, 10, 10)

        self._drop_area = QLabel()
        self._drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_area.setMinimumHeight(115)
        self._drop_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._drop_area.setCursor(Qt.CursorShape.PointingHandCursor)
        self._drop_area.setAcceptDrops(True)
        self._drop_area.installEventFilter(self)
        self._update_drop_area_style("default")
        drop_layout.addWidget(self._drop_area)

        browse_row = QHBoxLayout()
        browse_row.setSpacing(8)
        self._file_path = QLineEdit()
        self._file_path.setReadOnly(True)
        self._file_path.setPlaceholderText("Ningún archivo seleccionado")
        browse_row.addWidget(self._file_path, 1)
        browse_btn = QPushButton("Examinar…")
        browse_btn.clicked.connect(self._browse_file)
        browse_row.addWidget(browse_btn)
        drop_layout.addLayout(browse_row)
        main_layout.addWidget(drop_group, 0)

        # --- Fila central de 2 columnas (Pistas/Modo y Ajustes) ---
        mid_row = QHBoxLayout()
        mid_row.setSpacing(10)

        # Columna 1: Pistas y Modo
        tracks_group = QGroupBox("Pistas y Modo")
        tracks_layout = QVBoxLayout(tracks_group)
        tracks_layout.setSpacing(8)

        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Modo:")
        mode_lbl.setFixedWidth(65)
        mode_row.addWidget(mode_lbl)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Automático", "Solo extraer subtítulo", "Desde audio"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo, 1)
        tracks_layout.addLayout(mode_row)

        sub_row = QHBoxLayout()
        sub_lbl = QLabel("Subtítulo:")
        sub_lbl.setFixedWidth(65)
        sub_row.addWidget(sub_lbl)
        self._subtitle_combo = QComboBox()
        self._subtitle_combo.currentIndexChanged.connect(self._on_subtitle_track_changed)
        sub_row.addWidget(self._subtitle_combo, 1)
        tracks_layout.addLayout(sub_row)

        audio_row = QHBoxLayout()
        audio_lbl = QLabel("Audio:")
        audio_lbl.setFixedWidth(65)
        audio_row.addWidget(audio_lbl)
        self._audio_combo = QComboBox()
        self._audio_combo.currentIndexChanged.connect(self._on_audio_track_changed)
        audio_row.addWidget(self._audio_combo, 1)
        tracks_layout.addLayout(audio_row)

        mid_row.addWidget(tracks_group, 1)

        # Columna 2: Ajustes de traducción
        settings_group = QGroupBox("Ajustes de traducción")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(8)

        lang_row = QHBoxLayout()
        lang_lbl = QLabel("Idioma destino:")
        lang_lbl.setFixedWidth(130)
        lang_row.addWidget(lang_lbl)
        self._dst_lang = QLineEdit("es")
        self._dst_lang.setMaximumWidth(80)
        lang_row.addWidget(self._dst_lang)
        lang_row.addStretch()
        settings_layout.addLayout(lang_row)

        win_row = QHBoxLayout()
        win_lbl = QLabel("Ventana de contexto:")
        win_lbl.setFixedWidth(130)
        win_row.addWidget(win_lbl)
        self._window_size = QSpinBox()
        self._window_size.setRange(1, 200)
        self._window_size.setValue(50)
        self._window_size.setMaximumWidth(80)
        win_row.addWidget(self._window_size)
        win_row.addStretch()
        settings_layout.addLayout(win_row)

        cache_row = QHBoxLayout()
        self._keep_cache = QCheckBox("Mantener caché de traducción")
        self._keep_cache.setChecked(True)
        cache_row.addWidget(self._keep_cache)
        settings_layout.addLayout(cache_row)

        mid_row.addWidget(settings_group, 1)

        main_layout.addLayout(mid_row, 0)

        # --- Barra de progreso y estado ---
        progress_group = QGroupBox("Progreso")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(4)
        self._progress_bar = QProgressBar()
        self._progress_bar.setValue(0)
        progress_layout.addWidget(self._progress_bar)
        self._status_label = QLabel("Listo")
        progress_layout.addWidget(self._status_label)
        main_layout.addWidget(progress_group, 0)

        # --- Visor de registro ---
        log_group = QGroupBox("Registro")
        log_layout = QVBoxLayout(log_group)
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMinimumHeight(80)
        log_layout.addWidget(self._log_view)
        main_layout.addWidget(log_group, 1)

        # --- Botones ---
        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(10)
        self._start_btn = QPushButton("Iniciar traducción")
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._start)
        buttons_row.addWidget(self._start_btn)

        self._cancel_btn = QPushButton("Cancelar")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        buttons_row.addWidget(self._cancel_btn)
        main_layout.addLayout(buttons_row, 0)

        self._log("SubTrans GUI iniciado. Arrastra un archivo o haz clic en Examinar.")

    # -- event filter para drag & drop --------------------------------------

    def _update_drop_area_style(self, state: str) -> None:
        """Actualiza el aspecto y texto de la zona de arrastre."""
        if state == "hover":
            self._drop_area.setText(
                "<p style='margin:0; font-size: 14px; font-weight: bold;'>📥 Suelta el archivo aquí</p>"
                "<p style='margin:4px 0 0 0; color: #60a5fa; font-size: 11px;'>Formatos soportados: .mkv, .mp4, .srt</p>"
            )
            self._drop_area.setStyleSheet(
                "QLabel { border: 2px dashed #3b82f6; border-radius: 8px; "
                "background-color: rgba(59, 130, 246, 0.08); padding: 14px; color: #60a5fa; }"
            )
        elif state == "loaded" and self._loaded_path:
            self._drop_area.setText(
                f"<p style='margin:0; font-size: 14px; font-weight: bold;'>🎬 {self._loaded_path.name}</p>"
                "<p style='margin:4px 0 0 0; color: #10b981; font-size: 11px;'>✓ Archivo cargado correctamente · Haz clic o arrastra otro para cambiarlo</p>"
            )
            self._drop_area.setStyleSheet(
                "QLabel { border: 2px solid #10b981; border-radius: 8px; "
                "background-color: rgba(16, 185, 129, 0.08); padding: 14px; color: #34d399; }"
            )
        else:
            self._drop_area.setText(
                "<p style='margin:0; font-size: 14px; font-weight: bold;'>📂 Arrastra un archivo .mkv, .mp4 o .srt aquí</p>"
                "<p style='margin:4px 0 0 0; color: #888; font-size: 11px;'>o haz clic en esta zona para examinar en tu equipo</p>"
            )
            self._drop_area.setStyleSheet(
                "QLabel { border: 2px dashed #4a5568; border-radius: 8px; "
                "background-color: rgba(255, 255, 255, 0.02); padding: 14px; color: #cbd5e1; }"
                "QLabel:hover { border-color: #3b82f6; background-color: rgba(59, 130, 246, 0.05); }"
            )

    def eventFilter(self, obj: Any, event: Any) -> bool:
        if obj is not self._drop_area:
            return super().eventFilter(obj, event)
        event_type = event.type()
        if event_type == QEvent.Type.DragEnter:
            mime = event.mimeData()
            if mime.hasUrls():
                for url in mime.urls():
                    path = Path(url.toLocalFile())
                    if path.suffix.lower() in SUPPORTED_VIDEO_EXTS | SUPPORTED_SRT_EXTS:
                        event.acceptProposedAction()
                        self._update_drop_area_style("hover")
                        return True
            return False
        drag_leave = getattr(QEvent.Type, "DragLeave", None)
        if drag_leave is not None and event_type == drag_leave:
            self._update_drop_area_style("loaded" if self._loaded_path else "default")
            return True
        if event_type == QEvent.Type.Drop:
            mime = event.mimeData()
            if mime.hasUrls() and mime.urls():
                path = Path(mime.urls()[0].toLocalFile())
                if path.suffix.lower() in SUPPORTED_VIDEO_EXTS | SUPPORTED_SRT_EXTS:
                    self._load_file(path)
                    event.acceptProposedAction()
                    return True
            self._update_drop_area_style("loaded" if self._loaded_path else "default")
            return False
        mouse_press = getattr(QEvent.Type, "MouseButtonPress", None)
        if mouse_press is not None and event_type == mouse_press:
            if hasattr(event, "button") and event.button() == Qt.MouseButton.LeftButton:
                self._browse_file()
                return True
        return super().eventFilter(obj, event)

    # -- slots --------------------------------------------------------------

    @Slot()
    def _browse_file(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_VIDEO_EXTS | SUPPORTED_SRT_EXTS))
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo de vídeo o subtítulo", "", f"Archivos soportados ({exts})"
        )
        if path:
            self._load_file(Path(path))

    def _load_file(self, path: Path) -> None:
        if not path.is_file():
            self._log(f"Error: no existe el archivo {path}")
            return
        self._loaded_path = path
        self._file_path.setText(str(path))
        self._update_drop_area_style("loaded")
        self._log(f"Archivo cargado: {path.name}")

        suffix = path.suffix.lower()
        if suffix in SUPPORTED_SRT_EXTS:
            self._subtitle_tracks.clear()
            self._audio_tracks.clear()
            self._subtitle_combo.clear()
            self._audio_combo.clear()
            self._subtitle_combo.addItem("(SRT directo – sin pistas de vídeo)")
            self._audio_combo.addItem("(SRT directo – sin pistas de audio)")
            self._mode_combo.clear()
            self._mode_combo.addItem("Traducir SRT")
            self._mode_combo.setCurrentIndex(0)
            self._mode_combo.setEnabled(False)
            self._start_btn.setEnabled(True)
            return

        # Es un vídeo (.mkv / .mp4) – cargar pistas
        self._mode_combo.clear()
        self._mode_combo.addItems(["Automático", "Solo extraer subtítulo", "Desde audio"])
        self._mode_combo.setCurrentIndex(0)
        self._mode_combo.setEnabled(True)
        self._subtitle_combo.clear()
        self._audio_combo.clear()
        self._subtitle_tracks.clear()
        self._audio_tracks.clear()

        try:
            self._subtitle_tracks = list_subtitle_tracks(path)
        except FFmpegNotFoundError as exc:
            self._log(f"Error: {exc}")
            self._subtitle_tracks = []
        except Exception as exc:  # noqa: BLE001
            self._log(f"Error listando subtítulos: {exc}")
            self._subtitle_tracks = []

        if self._subtitle_tracks:
            for t in self._subtitle_tracks:
                self._subtitle_combo.addItem(t.describe(), t.sub_index)
        else:
            self._subtitle_combo.addItem("(sin pistas de subtítulo)")

        try:
            self._audio_tracks = list_audio_tracks(path)
        except FFmpegNotFoundError as exc:
            self._log(f"Error: {exc}")
            self._audio_tracks = []
        except Exception as exc:  # noqa: BLE001
            self._log(f"Error listando audio: {exc}")
            self._audio_tracks = []

        if self._audio_tracks:
            for t in self._audio_tracks:
                self._audio_combo.addItem(t.describe(), t.audio_index)
        else:
            self._audio_combo.addItem("(sin pistas de audio)")

        self._start_btn.setEnabled(True)
        self._log(f"Subtítulos: {len(self._subtitle_tracks)} pistas, Audio: {len(self._audio_tracks)} pistas")

    @Slot(int)
    def _on_subtitle_track_changed(self, index: int) -> None:
        if 0 <= index < len(self._subtitle_tracks):
            track = self._subtitle_tracks[index]
            self._log(f"Subtítulo seleccionado: {track.describe()}")

    @Slot(int)
    def _on_audio_track_changed(self, index: int) -> None:
        if 0 <= index < len(self._audio_tracks):
            track = self._audio_tracks[index]
            self._log(f"Audio seleccionado: {track.describe()}")

    @Slot(int)
    def _on_mode_changed(self, index: int) -> None:
        modes = ["auto", "extract", "from_audio"]
        mode = modes[index] if index < len(modes) else "auto"
        self._log(f"Modo cambiado a: {mode}")

    @Slot()
    def _start(self) -> None:
        source_text = self._file_path.text()
        if not source_text:
            QMessageBox.warning(self, "Error", "Selecciona un archivo primero.")
            return
        source = Path(source_text)
        if not source.is_file():
            QMessageBox.warning(self, "Error", f"No existe el archivo: {source}")
            return

        if source.suffix.lower() in SUPPORTED_SRT_EXTS:
            mode = "translate"
        else:
            mode_idx = self._mode_combo.currentIndex()
            modes = ["auto", "extract", "from_audio"]
            mode = modes[mode_idx] if 0 <= mode_idx < len(modes) else "auto"

        subtitle_idx = None
        if self._subtitle_combo.currentIndex() >= 0 and self._subtitle_tracks:
            subtitle_idx = self._subtitle_combo.currentData()

        audio_idx = None
        if self._audio_combo.currentIndex() >= 0 and self._audio_tracks:
            audio_idx = self._audio_combo.currentData()

        self._worker = Worker(
            source,
            mode,
            subtitle_track_index=subtitle_idx,
            audio_track_index=audio_idx,
            audio_lang="eng",
            dst_lang=self._dst_lang.text() or "es",
            window_size=self._window_size.value(),
            use_cache=self._keep_cache.isChecked(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.text_log.connect(self._on_log)
        self._worker.completed.connect(self._on_completed)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)

        self._start_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._status_label.setText("Ejecutando…")
        self._progress_bar.setValue(0)
        self._log(f"Iniciando modo '{mode}' con {source.name}…")
        self._worker.start()

    @Slot()
    def _cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.terminate()
            self._log("Operación cancelada.")
            self._status_label.setText("Cancelado")
            self._start_btn.setEnabled(True)
            self._cancel_btn.setEnabled(False)

    # -- señales del Worker -------------------------------------------------

    @Slot(int, int)
    def _on_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._progress_bar.setMaximum(total)
            self._progress_bar.setValue(current)
            self._status_label.setText(f"Procesando… {current}/{total}")

    @Slot(str)
    def _on_log(self, msg: str) -> None:
        self._log(msg)

    @Slot(str)
    def _on_completed(self, path: str) -> None:
        self._status_label.setText(f"Completado: {Path(path).name}")
        self._progress_bar.setValue(self._progress_bar.maximum())
        self._log(f"¡Listo! Archivo generado: {path}")
        QMessageBox.information(self, "Completado", f"Traducción completada:\n{path}")

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._status_label.setText("Error")
        self._log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Error", msg)

    def _on_finished(self) -> None:
        self._start_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)

    # -- helpers ------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self._log_view.append(msg)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def main() -> None:
    """Lanza la interfaz gráfica. Punto de entrada para ``subtrans-gui``."""
    if not HAS_PYSIDE6:
        print(
            "PySide6 no está instalado. Instálalo con:\n"
            "  pip install subtitle-translator[gui]",
            file=sys.stderr,
        )
        sys.exit(1)
    app = QApplication(sys.argv)
    app.setApplicationName("SubTrans")
    app.setApplicationVersion("0.1.0")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
