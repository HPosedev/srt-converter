"""Tests de config.py (esquema TOML con [gemini])."""

from pathlib import Path

import pytest

from config import DEFAULT_GEMINI_MODEL, load_config


FULL_TOML = """\
[gemini]
api_key = "AIza-test"
model = "gemini-3.6-flash"

[languages]
src = "en"
dst = "es"

[limits]
max_chars = 40
max_cps = 20.0
window_size = 100

[glossary]
"Winterfell" = "Invernalia"
"""


def test_load_full_toml(tmp_path: Path) -> None:
    """Todas las secciones se mapean a AppConfig."""
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(FULL_TOML, encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.gemini.api_key == "AIza-test"
    assert cfg.gemini.model == "gemini-3.6-flash"
    assert (cfg.src_lang, cfg.dst_lang) == ("en", "es")
    assert (cfg.max_chars, cfg.max_cps, cfg.window_size) == (40, 20.0, 100)
    assert cfg.glossary == {"Winterfell": "Invernalia"}
    assert cfg.api_key == "AIza-test"  # atajo compat


def test_load_defaults_when_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin archivo → defecto (ventana 50, modelo flash) salvo entorno."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("config.default_config_search_paths", lambda: [])
    cfg = load_config()
    assert cfg.gemini.api_key == ""
    assert cfg.gemini.model == DEFAULT_GEMINI_MODEL
    assert cfg.window_size == 50


def test_env_api_key_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GEMINI_API_KEY rellena api_key cuando el TOML no la trae."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "env-key-123")
    monkeypatch.setattr("config.default_config_search_paths", lambda: [])
    cfg = load_config()
    assert cfg.gemini.api_key == "env-key-123"


def test_explicit_missing_path_raises(tmp_path: Path) -> None:
    """Ruta explícita inexistente → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "no.toml")


def test_invalid_toml_raises(tmp_path: Path) -> None:
    """TOML roto → ValueError claro."""
    bad = tmp_path / "bad.toml"
    bad.write_text("[gemini\napi_key = ", encoding="utf-8")
    with pytest.raises(ValueError, match="Config inválida"):
        load_config(bad)


def test_translation_section_mapping(tmp_path: Path) -> None:
    """[translation] mapea source/target/style y prevalece."""
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[gemini]\napi_key = "k"\n\n'
        "[translation]\nsource = \"en\"\ntarget = \"es\"\n"
        "window_size = 30\nstyle = \"Tono natural\"\n\n"
        "[languages]\nsrc = \"fr\"\ndst = \"de\"\n\n"
        "[limits]\nwindow_size = 99\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert (cfg.src_lang, cfg.dst_lang) == ("en", "es")
    assert cfg.window_size == 30
    assert cfg.style_instructions == "Tono natural"


def test_legacy_languages_fallback(tmp_path: Path) -> None:
    """Sin [translation], [languages]/[limits] siguen valiendo."""
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        "[languages]\nsrc = \"en\"\ndst = \"pt\"\n\n[limits]\nwindow_size = 42\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert (cfg.src_lang, cfg.dst_lang) == ("en", "pt")
    assert cfg.window_size == 42
    assert cfg.style_instructions == ""
