"""Tests de la interfaz gráfica (gui.py).

Tests desacoplados de PySide6: mockean las clases de Qt y verifican
la lógica de negocio (Worker, carga de archivos, selección de modo,
signal/slot connections).

Si ``pytest-qt`` y ``PySide6`` están disponibles, se ejecutan tests
adicionales con display real usando ``qtbot``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock de PySide6 si no está instalado
# ---------------------------------------------------------------------------

HAS_PYSIDE6 = False
try:
    import PySide6  # noqa: F401

    HAS_PYSIDE6 = True
except ImportError:
    pass

# Si no hay PySide6, creamos mocks mínimos para que el módulo gui importe
if not HAS_PYSIDE6:
    mock_qt = MagicMock()
    mock_qt.Qt = MagicMock()
    mock_qt.Qt.AlignmentFlag = MagicMock()
    mock_qt.Qt.AlignmentFlag.AlignCenter = 0
    mock_qt.QEvent = MagicMock()
    mock_qt.QEvent.Type = MagicMock()
    mock_qt.QEvent.Type.DragEnter = 0
    mock_qt.QEvent.Type.Drop = 0

    class _MockQThread:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            pass

    mock_qt.QThread = _MockQThread
    mock_qt.Signal = lambda *a: MagicMock()
    mock_qt.Slot = lambda *a: (lambda f: f)
    for name in (
        "QDragEnterEvent", "QDropEvent", "QApplication", "QCheckBox",
        "QComboBox", "QDoubleSpinBox", "QFileDialog", "QGroupBox",
        "QHBoxLayout", "QLabel", "QLineEdit", "QMainWindow",
        "QMessageBox", "QProgressBar", "QPushButton", "QSpinBox",
        "QTextEdit", "QVBoxLayout", "QWidget",
    ):
        setattr(mock_qt, name, type(name, (), {"__init__": lambda s, **kw: None}))
    sys.modules["PySide6"] = mock_qt
    sys.modules["PySide6.QtCore"] = mock_qt
    sys.modules["PySide6.QtGui"] = mock_qt
    sys.modules["PySide6.QtWidgets"] = mock_qt


# Importar después del mockeo
from gui import HAS_PYSIDE6 as gui_has_pyside6, Worker


# ---------------------------------------------------------------------------
# Tests desacoplados (no requieren display)
# ---------------------------------------------------------------------------


class TestWorkerInit:
    """Tests de inicialización del Worker."""

    def test_worker_creation(self, tmp_path: Path) -> None:
        source = tmp_path / "test.mkv"
        source.write_text("fake")
        worker = Worker(source, "auto")
        assert worker.source == source
        assert worker.mode == "auto"
        assert worker.subtitle_track_index is None
        assert worker.audio_track_index is None
        assert worker.dst_lang == "es"
        assert worker.window_size == 50
        assert worker.use_cache is True
        assert worker._cancelled is False

    def test_worker_with_params(self, tmp_path: Path) -> None:
        source = tmp_path / "test.mkv"
        source.write_text("fake")
        worker = Worker(
            source,
            "from_audio",
            subtitle_track_index=0,
            audio_track_index=1,
            audio_lang="spa",
            dst_lang="fr",
            window_size=100,
            use_cache=False,
        )
        assert worker.mode == "from_audio"
        assert worker.subtitle_track_index == 0
        assert worker.audio_track_index == 1
        assert worker.audio_lang == "spa"
        assert worker.dst_lang == "fr"
        assert worker.window_size == 100
        assert worker.use_cache is False

    def test_worker_cancel(self, tmp_path: Path) -> None:
        source = tmp_path / "test.mkv"
        source.write_text("fake")
        worker = Worker(source, "auto")
        assert worker._cancelled is False
        worker.cancel()
        assert worker._cancelled is True


class TestWorkerRun:
    """Tests de ejecución del Worker con mocks."""

    def test_worker_unknown_mode(self, tmp_path: Path) -> None:
        source = tmp_path / "test.mkv"
        source.write_text("fake")
        worker = Worker(source, "unknown_mode")
        error_mock = MagicMock()
        worker.error = error_mock
        worker.run()
        error_mock.emit.assert_called_once()
        assert "Modo desconocido" in error_mock.emit.call_args[0][0]

    def test_worker_extract_no_tracks(self, tmp_path: Path) -> None:
        source = tmp_path / "test.mkv"
        source.write_text("fake")
        worker = Worker(source, "extract")
        error_mock = MagicMock()
        worker.error = error_mock
        with patch("gui.list_subtitle_tracks", return_value=[]):
            worker.run()
        error_mock.emit.assert_called_once()
        assert "pistas" in error_mock.emit.call_args[0][0].lower()

    def test_worker_auto_no_tracks(self, tmp_path: Path) -> None:
        source = tmp_path / "test.mkv"
        source.write_text("fake")
        worker = Worker(source, "auto")
        error_mock = MagicMock()
        worker.error = error_mock
        with patch("gui.list_subtitle_tracks", return_value=[]):
            worker.run()
        error_mock.emit.assert_called_once()
        assert "pistas" in error_mock.emit.call_args[0][0].lower()

    def test_worker_from_audio_no_tracks(self, tmp_path: Path) -> None:
        source = tmp_path / "test.mkv"
        source.write_text("fake")
        worker = Worker(source, "from_audio")
        error_mock = MagicMock()
        worker.error = error_mock
        with patch("gui.list_audio_tracks", return_value=[]):
            worker.run()
        error_mock.emit.assert_called_once()
        assert "pistas" in error_mock.emit.call_args[0][0].lower()

    def test_worker_auto_no_api_key(self, tmp_path: Path) -> None:
        source = tmp_path / "test.mkv"
        source.write_text("fake")
        worker = Worker(source, "auto")
        error_mock = MagicMock()
        worker.error = error_mock
        mock_track = MagicMock()
        mock_track.describe.return_value = "s:0 test"
        with (
            patch("gui.list_subtitle_tracks", return_value=[mock_track]),
            patch("gui.list_audio_tracks", return_value=[]),
            patch("gui.load_config") as mock_cfg,
        ):
            mock_cfg.return_value.gemini.api_key = ""
            mock_cfg.return_value.glossary = {}
            mock_cfg.return_value.src_lang = "en"
            mock_cfg.return_value.dst_lang = "es"
            mock_cfg.return_value.window_size = 50
            mock_cfg.return_value.max_chars = 42
            mock_cfg.return_value.max_cps = 17.0
            mock_cfg.return_value.style_instructions = ""
            worker.run()
        error_mock.emit.assert_called_once()
        assert "API key" in error_mock.emit.call_args[0][0]

    def test_worker_extract_success(self, tmp_path: Path) -> None:
        source = tmp_path / "test.mkv"
        source.write_text("fake")
        worker = Worker(source, "extract")
        completed_mock = MagicMock()
        worker.completed = completed_mock
        log_mock = MagicMock()
        worker.text_log = log_mock
        progress_mock = MagicMock()
        worker.progress = progress_mock
        mock_track = MagicMock()
        mock_track.describe.return_value = "s:0 [subrip] lang=eng (completos)"
        mock_track.sub_index = 0
        with (
            patch("gui.list_subtitle_tracks", return_value=[mock_track]),
            patch("gui.select_track", return_value=mock_track),
            patch("gui.extract_subtitle", return_value=tmp_path / "test.srt"),
        ):
            worker.run()
        completed_mock.emit.assert_called_once()
        assert str(tmp_path / "test.srt") in completed_mock.emit.call_args[0][0]


class TestWorkerFromAudioSuccess:
    """Tests del Worker en modo from_audio con mocks."""

    def test_from_audio_success(self, tmp_path: Path) -> None:
        source = tmp_path / "test.mkv"
        source.write_text("fake")
        worker = Worker(source, "from_audio", audio_lang="eng")
        completed_mock = MagicMock()
        worker.completed = completed_mock
        log_mock = MagicMock()
        worker.text_log = log_mock
        progress_mock = MagicMock()
        worker.progress = progress_mock

        mock_audio = MagicMock()
        mock_audio.describe.return_value = "a:0 [aac, 2ch] lang=eng"
        mock_subtitler = MagicMock()
        mock_subtitler.transcribe_and_translate_audio.return_value = []
        mock_subtitler.fallback_notices = []

        with (
            patch("gui.load_config") as mock_cfg,
            patch("gui.list_audio_tracks", return_value=[mock_audio]),
            patch("gui.select_audio_track", return_value=mock_audio),
            patch("gui.extract_audio", return_value=tmp_path / "test.mp3"),
            patch("gui.GeminiSubtitler", return_value=mock_subtitler),
            patch("gui.audit_subtitles", return_value=[]),
            patch("gui.write_srt_file", return_value=tmp_path / "test_es.srt"),
        ):
            mock_cfg.return_value.gemini.api_key = "fake-key"
            mock_cfg.return_value.gemini.model = "gemini-3.6-flash"
            mock_cfg.return_value.glossary = {}
            mock_cfg.return_value.src_lang = "en"
            mock_cfg.return_value.dst_lang = "es"
            mock_cfg.return_value.window_size = 50
            mock_cfg.return_value.max_chars = 42
            mock_cfg.return_value.max_cps = 17.0
            mock_cfg.return_value.style_instructions = ""
            worker.run()

        completed_mock.emit.assert_called_once()
        assert str(tmp_path / "test_es.srt") in completed_mock.emit.call_args[0][0]


class TestWorkerAutoSuccess:
    """Tests del Worker en modo auto con mocks."""

    def test_auto_success(self, tmp_path: Path) -> None:
        source = tmp_path / "test.mkv"
        source.write_text("fake")
        worker = Worker(source, "auto")
        completed_mock = MagicMock()
        worker.completed = completed_mock
        log_mock = MagicMock()
        worker.text_log = log_mock
        progress_mock = MagicMock()
        worker.progress = progress_mock

        mock_track = MagicMock()
        mock_track.describe.return_value = "s:0 [subrip] lang=eng"
        mock_subtitler = MagicMock()
        mock_subtitler.fallback_notices = []

        with (
            patch("gui.load_config") as mock_cfg,
            patch("gui.list_subtitle_tracks", return_value=[mock_track]),
            patch("gui.select_track", return_value=mock_track),
            patch("gui.extract_subtitle", return_value=tmp_path / "test.extracted.srt"),
            patch("gui.translate_file", return_value=(tmp_path / "test_es.srt", [])),
            patch("gui.GeminiSubtitler", return_value=mock_subtitler),
        ):
            mock_cfg.return_value.gemini.api_key = "fake-key"
            mock_cfg.return_value.gemini.model = "gemini-3.6-flash"
            mock_cfg.return_value.glossary = {}
            mock_cfg.return_value.src_lang = "en"
            mock_cfg.return_value.dst_lang = "es"
            mock_cfg.return_value.window_size = 50
            mock_cfg.return_value.max_chars = 42
            mock_cfg.return_value.max_cps = 17.0
            mock_cfg.return_value.style_instructions = ""
            worker.run()

        completed_mock.emit.assert_called_once()
        assert str(tmp_path / "test_es.srt") in completed_mock.emit.call_args[0][0]


class TestWorkerTranslateSuccess:
    """Tests del Worker en modo translate (SRT directo) con mocks."""

    def test_translate_success(self, tmp_path: Path) -> None:
        source = tmp_path / "test.srt"
        source.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        worker = Worker(source, "translate")
        completed_mock = MagicMock()
        worker.completed = completed_mock
        log_mock = MagicMock()
        worker.text_log = log_mock
        progress_mock = MagicMock()
        worker.progress = progress_mock

        mock_subtitler = MagicMock()
        mock_subtitler.fallback_notices = []

        with (
            patch("gui.load_config") as mock_cfg,
            patch("gui.translate_file", return_value=(tmp_path / "test_es.srt", [])),
            patch("gui.GeminiSubtitler", return_value=mock_subtitler),
        ):
            mock_cfg.return_value.gemini.api_key = "fake-key"
            mock_cfg.return_value.gemini.model = "gemini-3.6-flash"
            mock_cfg.return_value.glossary = {}
            mock_cfg.return_value.src_lang = "en"
            mock_cfg.return_value.dst_lang = "es"
            mock_cfg.return_value.window_size = 50
            mock_cfg.return_value.max_chars = 42
            mock_cfg.return_value.max_cps = 17.0
            mock_cfg.return_value.style_instructions = ""
            worker.run()

        completed_mock.emit.assert_called_once()
        assert str(tmp_path / "test_es.srt") in completed_mock.emit.call_args[0][0]


class TestGuiModule:
    """Tests del módulo gui (importación y constants)."""

    def test_has_pyside6_constant(self) -> None:
        import gui

        assert hasattr(gui, "HAS_PYSIDE6")
        assert isinstance(gui.HAS_PYSIDE6, bool)

    def test_supported_extensions(self) -> None:
        import gui

        assert ".mkv" in gui.SUPPORTED_VIDEO_EXTS
        assert ".mp4" in gui.SUPPORTED_VIDEO_EXTS
        assert ".srt" in gui.SUPPORTED_SRT_EXTS

    def test_main_function_exists(self) -> None:
        import gui

        assert callable(gui.main)


class TestGuiSubcommand:
    """Test del subcomando gui en app.py."""

    def test_gui_command_is_registered(self) -> None:
        from typer.testing import CliRunner

        from app import app

        runner = CliRunner()
        result = runner.invoke(app, ["gui", "--help"])
        assert result.exit_code == 0
        assert "PySide6" in result.output or "gui" in result.output.lower()


# ---------------------------------------------------------------------------
# Tests con pytest-qt (solo si PySide6 y pytest-qt disponibles)
# ---------------------------------------------------------------------------

HAS_QT = HAS_PYSIDE6
try:
    import pytest_qt  # noqa: F401

    HAS_QT = HAS_QT and True
except ImportError:
    HAS_QT = False


@pytest.mark.skipif(not HAS_QT, reason="PySide6/pytest-qt no disponible")
class TestMainWindowWithQt:
    """Tests de MainWindow con display real (requiere pytest-qt)."""

    def test_window_creation(self, qtbot: Any) -> None:
        from gui import MainWindow

        window = MainWindow()
        qtbot.addWidget(window)
        assert window.windowTitle() == "SubTrans – Traductor de subtítulos"
        assert window._file_path.text() == ""
        assert window._start_btn.isEnabled() is False

    def test_window_log(self, qtbot: Any) -> None:
        from gui import MainWindow

        window = MainWindow()
        qtbot.addWidget(window)
        window._log("Mensaje de prueba")
        assert "Mensaje de prueba" in window._log_view.toPlainText()
