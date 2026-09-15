# 🎬 Subtitle Translator (`subtrans`)

Herramienta completa en Python para extraer subtítulos y pistas de audio de vídeos (`.mkv`, `.mp4`), traducirlos con **Google Gemini AI** (`google-genai`) conservando los timestamps originales 1:1 y exportarlos como archivos SRT de alta calidad.

Incluye interfaz de línea de comandos (**CLI**) moderna y una interfaz gráfica de escritorio (**GUI**) con PySide6.

---

## ✨ Características

- 🎯 **Preservación exacta de timestamps (1:1):** Los tiempos de inicio y fin (`start` / `end`) nunca se alteran. Solo se reemplaza el contenido textual.
- 🛡️ **Protección de archivos originales:** Nunca sobrescribe los archivos originales; genera automáticamente el sufijo `_es.srt` (o según el idioma configurado).
- 🔄 **Traducción contextual por ventanas:** Agrupa bloques consecutivos (por defecto 50 bloques) para mantener la coherencia temporal, pronombres y tono cinematográfico.
- 🧠 **Resiliencia ante fallos y límites de API:**
  - Reintentos automáticos con backoff exponencial (2s, 4s, 8s) ante errores de cuota (HTTP 429 / `ResourceExhausted`) y problemas transitorios de red.
  - Subdivisión recursiva de ventanas (*chunk halving*) si el modelo descuadra el conteo de bloques.
  - Conservación del texto original con aviso explícito si un bloque individual no logra traducirse tras agotar intentos.
- 💾 **Caché de reanudación automática:** Guarda el progreso en `<archivo>.subtrans-cache.json`. Si el proceso se interrumpe (Ctrl+C, corte de red, etc.), la siguiente ejecución continúa exactamente donde se quedó. Al finalizar con éxito, la caché se limpia de forma automática.
- 🎙️ **Transcripción directa desde audio:** Si el vídeo no incluye pistas de subtítulos en texto, puede extraer el audio a MP3 (mono, 16 kHz) y usar la API de archivos de Gemini para transcribir y traducir directamente a SRT en un solo paso.
- 🔍 **Auditoría de legibilidad:** Revisa estándares profesionales de subtitulado (~42 caracteres por línea y ~17 caracteres por segundo) mostrando advertencias claras con tablas Rich en la terminal sin alterar el texto.
- 📖 **Glosario y directrices de estilo:** Permite forzar traducciones fijas para nombres propios, lugares o terminología, así como definir el registro lingüístico (p. ej. *"Español de España, tono natural y cinematográfico"*).
- 🖥️ **Interfaz Gráfica (GUI):** Soporte para arrastrar y soltar archivos, selección interactiva de pistas de subtítulos/audio, visualización de logs en tiempo real y barras de progreso.

---

## 📋 Requisitos previos

1. **Python 3.11** o superior.
2. **FFmpeg y FFprobe** instalados en el sistema:
   - **Arch Linux / CachyOS:** `sudo pacman -S ffmpeg`
   - **Debian / Ubuntu:** `sudo apt update && sudo apt install ffmpeg`
   - **Fedora:** `sudo dnf install ffmpeg`
   - **macOS (Homebrew):** `brew install ffmpeg`
   - **Windows:** Instalar vía [ffmpeg.org](https://ffmpeg.org/download.html) o `winget install Gyan.FFmpeg`.
3. **API Key de Google Gemini:** Consíguela gratis en [Google AI Studio](https://aistudio.google.com/).

---

## 🚀 Instalación

Clona el repositorio e instala en un entorno virtual:

```bash
# 1. Clonar el repositorio
git clone https://github.com/HPosedev/srt-converter.git
cd srt-converter

# 2. Crear y activar el entorno virtual
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# 3. Instalar con dependencias estándar
pip install -e .

# Opcional: instalar con dependencias para tests y GUI
pip install -e ".[dev,gui]"
```

---

## ⚙️ Configuración

Puedes configurar las credenciales mediante el archivo `config.toml` o mediante variables de entorno:

### Opción A: Archivo `config.toml` (Recomendada)
Copia la plantilla `config.example.toml` a `config.toml` en la raíz del proyecto (o en `~/.config/subtrans/config.toml`):

```bash
cp config.example.toml config.toml
```

Edita `config.toml`:

```toml
[gemini]
api_key = "AIzaSy..."               # Tu API key de Gemini
model = "gemini-3.6-flash"         # Modelo a utilizar

[translation]
source = "en"
target = "es"
window_size = 50                   # Bloques por petición (1 - 200)
style = "Español de España, tono natural y cinematográfico"

[limits]
max_chars = 42                     # Límite auditoría chars/línea
max_cps = 17.0                     # Límite auditoría chars/segundo

[glossary]
"Winterfell" = "Invernalia"
"King's Landing" = "Desembarco del Rey"
"Night's Watch" = "Guardia de la Noche"
```

> ⚠️ **Seguridad:** El archivo `config.toml` está en `.gitignore` para evitar subir tus claves a GitHub. **Nunca comitas ni compartas tus API keys.**

### Opción B: Variable de entorno
Si prefieres no usar un archivo de configuración para la clave:

```bash
export GEMINI_API_KEY="tu-api-key-de-gemini"
```

---

## 💻 Uso en Terminal (CLI)

El comando `subtrans` proporciona tres modos principales: `extract`, `translate` y `auto`.

### 1. `subtrans extract` — Explorar o extraer pistas de vídeo

```bash
# Listar pistas de subtítulos disponibles
subtrans extract video.mkv --list

# Listar pistas de audio disponibles
subtrans extract video.mkv --list-audio

# Extraer subtítulo en inglés a un archivo .srt
subtrans extract video.mkv --output subs_en.srt --lang eng

# Extraer una pista específica por su índice (s:0, s:1...)
subtrans extract video.mkv --output subs.srt --track 0

# Extraer la pista de audio a MP3 mono
subtrans extract video.mkv --audio --output audio.mp3 --audio-lang eng
```

### 2. `subtrans translate` — Traducir archivos SRT existentes

```bash
# Traducir un archivo individual (genera capitulo_01_es.srt)
subtrans translate capitulo_01.srt

# Especificar archivo de salida personalizado
subtrans translate capitulo_01.srt --output capitulo_castellano.srt

# Traducir por lotes todos los .srt de una carpeta
subtrans translate ./temporada_1/ --batch

# Ajustar tamaño de ventana o deshabilitar caché
subtrans translate archivo.srt --window-size 30 --no-cache
```

### 3. `subtrans auto` — Flujo integral MKV/MP4 → `*_es.srt`

Procesa el vídeo completo: extrae la pista de subtítulo adecuada, la traduce con Gemini y limpia los archivos intermedios.

```bash
# Modo estándar: busca pista de subtítulos, la extrae y traduce
subtrans auto pelicula.mkv

# Vídeo sin subtítulos de texto: transcribe y traduce desde el audio
subtrans auto video_sin_subs.mp4 --from-audio --audio-lang eng

# Procesar una carpeta completa de episodios (omite los ya traducidos)
subtrans auto /ruta/a/serie/temporada_01/

# Conservar los archivos temporales (.extracted.srt o .mp3)
subtrans auto video.mkv --keep-intermediate
```

---

## 🖥️ Interfaz Gráfica (GUI)

Para abrir la interfaz de escritorio:

```bash
subtrans-gui
# O también:
subtrans gui
```

- **Arrastra y suelta** cualquier archivo `.mkv`, `.mp4` o `.srt`.
- Selecciona pistas de audio y subtítulos mediante menús desplegables.
- Elige entre los modos: **Automático**, **Solo extraer** o **Desde audio** (o **Traducir SRT directo** al soltar un `.srt`).
- Monitorea el progreso con la barra porcentual y el visor de registros.

---

## 🧪 Ejecución de Tests

La suite incluye tests unitarios y de integración con clientes simulados (*mocks*) sin consumir cuota real de Gemini ni requerir pantalla:

```bash
pytest
```

---

## 📁 Estructura del Proyecto

```
srt-converter/
├── app.py               # Punto de entrada CLI con Typer (extract, translate, auto, gui)
├── config.py            # Carga y validación de configuración TOML / variables de entorno
├── config.example.toml  # Plantilla de configuración de ejemplo
├── extractor.py         # Integración con ffprobe y ffmpeg (listado y extracción de pistas)
├── translator.py        # Motor de traducción con Gemini (ventanas, reintentos, Files API)
├── srt_utils.py         # Parser, serializador y auditoría de subtítulos SRT
├── cache.py             # Sistema de persistencia y reanudación de ventanas
├── gui.py               # Interfaz gráfica con PySide6 y QThread
├── pyproject.toml       # Metadatos del paquete y dependencias del proyecto
└── tests/               # Suite exhaustiva de pruebas unitarias con pytest
```

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo `LICENSE` para más información.
