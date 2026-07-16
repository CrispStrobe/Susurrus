# Susurrus: Audio Transcription, TTS & Translation Suite

Susurrus is a professional, modular audio suite providing transcription, text-to-speech, translation, and speech-to-speech through a unified GUI and CLI. Built with a clean architecture, it supports 42+ ASR backends (via CrispASR), 34+ TTS engines, multi-language translation, speaker diarization, EU AI Act compliance, and extensive customization options.

### Part of the Crisp ecosystem

| Project | Role |
|---|---|
| **Susurrus** | This repo — Python GUI + CLI with 42+ ASR, 34+ TTS, translation, S2S |
| **[CrispASR](https://github.com/CrispStrobe/CrispASR)** | C++ ASR/TTS engine (v0.8.12) — 42+ ASR + 34+ TTS backends, ggml inference. Two integration paths: subprocess (binary) or FFI (libcrispasr Python bindings) |
| **[CrisperWeaver](https://github.com/CrispStrobe/CrisperWeaver)** | Flutter transcription app powered by CrispASR — desktop + mobile, fully offline |
| **[CrispTTS](https://github.com/CrispStrobe/CrispTTS)** | Python TTS suite — 20+ handlers, German focus |
| **[CrispEmbed](https://github.com/CrispStrobe/CrispEmbed)** | Text embedding engine (ggml) — XLM-R, Qwen3-Embed, Gemma3, dense + sparse + ColBERT |

## Features

### Transcription (42+ backends)

- **CrispASR engine** (42+ ASR sub-backends): whisper, parakeet, canary, canary-ctc, canary-qwen, cohere, cohere-ar, qwen3, qwen3-1.7b, qwen3-ja-anime, mega-asr, voxtral, voxtral4b, granite, granite-4.1, moonshine, moonshine-streaming, kyutai-stt, kyutai-stt-2.6b, fastconformer-ctc, wav2vec2, hubert, data2vec, vibevoice, firered-asr, funasr, paraformer, sensevoice, glm-asr, omniasr, mimo-asr, moss-audio, moss-transcribe, moss-diarize, gemma4-e2b, gemma4-e4b, nemotron, mini-omni2, ark-asr, higgs-stt, parakeet-ctc-ja, reazonspeech, and more
- **CrispASR FFI** (`crispasr-ffi`): In-process inference via Python ctypes to libcrispasr — zero IPC overhead, persistent model sessions, native word-level timestamps and confidence scores
- **CrispASR subprocess** (`crispasr`): Binary execution with full parameter passthrough — works with just the binary, no shared library needed
- **Python backends**: mlx-whisper, faster-whisper (batched + sequenced), transformers, whisper.cpp, ctranslate2, whisper-jax, insanely-fast-whisper, OpenAI Whisper, Voxtral (local + API)
- **Flexible Input**: Local files, URLs, video audio extraction
- **Audio Format Support**: MP3, WAV, FLAC, M4A, AAC, OGG, OPUS, WebM, MP4, WMA
- **Language Detection**: Automatic or manual, multiple LID backends (whisper, silero, firered, ecapa)
- **Word-level Timestamps**: Native or CTC aligner-based
- **Performance Metrics**: Real-time factor (RTF) and words-per-second (WPS)
- **Backend Availability Probing**: Auto-detects which CrispASR backends are compiled in

### Text-to-Speech (34+ engines)

- **CrispASR TTS** (34+ C++ backends): kokoro, orpheus, qwen3-tts, chatterbox, vibevoice, vibevoice-1.5b, indextts, voxcpm2-tts, melotts, piper, bark, dia, zonos, csm, cosyvoice3-tts, f5-tts, fastpitch, parler-tts, outetts, pocket-tts, speecht5, kugelaudio, lfm2-audio, mini-omni2, tada, dots-tts, bananamind-tts, bananamind-tts-de, voxtral-tts, omnivoice, irodori-tts, irodori-tts-voicedesign, moss-tts, moss-tts-local
- **Python-native TTS** (5 backends): Edge TTS (cloud), Piper (MIT, ONNX), Kokoro ONNX (Apache 2.0), Chatterbox (MIT), SpeechT5 (MIT)
- **Voice cloning**: Reference audio support for applicable backends
- **Text extraction**: Load text from TXT, Markdown, HTML, PDF, EPUB files for synthesis
- **Voice selection**: Per-backend voice lists with configurable presets
- **Local playback**: `--tts-play` for direct speaker output

### Speech-to-Speech

- **Audio-in → Audio-out**: Supported by lfm2-audio and mini-omni2 backends via `--s2s` flag
- **FFI support**: `speech_to_speech()` method for in-process S2S with optional intermediate transcript

### Translation

- **CrispASR translation**: m2m100 (100 languages), m2m100-f16 (exact HF parity), MadLad (419 languages), Gemma4-E2B (140+ languages)
- **Bidirectional**: Any source → any target language pair

### Speaker Diarization

- **PyAnnote.audio**: State-of-the-art neural diarization (requires HF token)
- **CrispASR methods**: energy, xcorr, vad-turns, pyannote, sherpa, ecapa
- **Language-specific models**: English, German, Chinese, Spanish, Japanese
- **Configurable**: Min/max speaker counts, cluster thresholds

### Advanced CrispASR Features

- **VAD**: Silero, FireRed, with configurable thresholds
- **Streaming**: Live microphone, stdin, rolling-window transcription
- **Server mode**: OpenAI-compatible HTTP API
- **Grammar constraints**: GBNF constrained decoding
- **Punctuation restoration**: FireRedPunc post-processing
- **Forced alignment**: CTC aligner for word timestamps
- **Speaker verification**: TitaNet embeddings, speaker profile DB
- **Model auto-download**: Registry-based with SHA-256 verification
- **Companion model resolution**: Auto-resolves codec/voice dependencies
- **GPU layer offloading**: Partial GPU offload via `-ngl N` for LLM-based backends
- **Wyoming protocol**: Home Assistant Assist integration via `--wyoming-port`
- **Hotwords/contextual biasing**: `--hotwords` for domain-specific vocabulary boosting
- **Standalone alignment**: `--align-only` mode for aligning text to audio without ASR
- **TADA voice reference creation**: `--make-ref` to create voice GGUF from WAV

### EU AI Act Compliance

- **Watermark ON by default**: All TTS output gets AudioSeal watermark (CrispASR default)
- **C2PA signing ON by default**: Content Credentials signed with bundled cert
- **Voice cloning gated**: `--i-have-rights` required for .wav reference cloning
- **`--no-watermark`**: Explicit opt-out with Art. 50 responsibility-shift warning
- **`--detect-watermark`**: Standalone AI-content detection (confidence + verdict)
- **`--verify-c2pa`**: Verify C2PA Content Credentials in audio files
- **c2pa-audio integration**: Python-native signing for non-CrispASR TTS backends via [c2pa-audio](https://github.com/CrispStrobe/c2pa-audio)

### GUI

- **4-tab layout**: Transcription / Text-to-Speech / Translation / History
- **Segment list view**: Per-segment display with speaker color chips, confidence badges, inline editing
- **Waveform display**: PCM visualization with segment highlights, auto-loads on file selection
- **Live mic streaming**: "Stream Mic" button — real-time transcription from microphone
- **Watermark detection**: "Detect Watermark" button — check if audio is AI-generated
- **Batch queue**: Multi-file sequential processing with status tracking
- **History browser**: Search, load, delete past transcriptions (auto-saved)
- **Voice clone wizard**: 3-step guided dialog with EU AI Act consent (Tools menu)
- **Server toggle**: Start/stop OpenAI-compatible HTTP server from Tools menu
- **Light/dark themes**: Toggle via Ctrl+T, persisted in QSettings
- **Log viewer**: Real-time log display with level filtering (View → Show Logs)
- **i18n**: English + German translations (90+ strings)
- **CrispASR advanced settings**: Collapsible panel for VAD, diarization, LID, alignment, grammar, streaming
- **TTS panel**: Text input, file loading, backend/voice selection, reference audio, C2PA/watermark controls
- **Translation panel**: Source/target language, backend selection
- **Export formats**: SRT, VTT, JSON, CSV, TXT (format picker in Save dialog)
- **Drag-and-drop**: First file → input, additional files → batch queue
- **Keyboard shortcuts**: F5=Transcribe, Ctrl+S=Save, Ctrl+T=Theme, Ctrl+H=History
- **Settings persistence**: QSettings across sessions

## Installation

### Quick Start

```bash
git clone https://github.com/CrispStrobe/Susurrus.git
cd Susurrus

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -e ".[gui]"

# GUI
python main.py

# CLI (no GUI needed)
pip install -e .
python cli.py --list-backends
```

### Optional Dependencies

```bash
# GUI (PyQt6)
pip install -e ".[gui]"

# GPU backends (torch + torchaudio)
pip install -e ".[gpu]"

# TTS backends
pip install -e ".[tts]"

# Text extraction (PDF, EPUB, HTML, Markdown)
pip install -e ".[text-extraction]"

# Everything
pip install -e ".[all]"

# Dev tools
pip install -e ".[dev]"
```

### Prerequisites

- **Python 3.9+**
- **FFmpeg** (for audio format conversion)
- **CrispASR binary** (auto-downloaded if not found) or **libcrispasr.so** (for FFI backend)

## CLI Usage

```bash
# List all backends
python cli.py --list-backends

# Transcription — CrispASR sub-backend with auto-download
python cli.py --backend crispasr:parakeet --model auto --file audio.wav --auto-download

# Transcription — CrispASR FFI (in-process, requires libcrispasr.so)
python cli.py --backend crispasr-ffi --model /path/to/model.gguf --file audio.wav

# Transcription — faster-whisper
python cli.py --backend faster-sequenced --model large-v3 --file audio.wav

# TTS — Edge TTS (cloud, no model download)
python cli.py --mode tts --tts-backend edge-tts --text "Hello world" --tts-output out.wav

# TTS — CrispASR Orpheus
python cli.py --mode tts --backend crispasr:orpheus --model auto --text "Hello" --voice Tara

# Translation — m2m100
python cli.py --mode translate --backend crispasr:m2m100 --model auto \
  --text "Hello world" --source-lang en --target-lang de

# Streaming (live microphone)
python cli.py --mode stream --backend crispasr --model auto --mic --auto-download

# Server mode
python cli.py --mode server --backend crispasr --model auto --port 8080

# CrispASR with VAD, diarization, punctuation
python cli.py --backend crispasr:parakeet --model auto --file audio.wav \
  --vad --diarize --diarize-method pyannote --punc-model auto --auto-download
```

### Python API

```python
# Transcription
from workers.transcription.backends import get_backend

backend = get_backend("crispasr:parakeet", model_id="auto", device="cpu", auto_download=True)
for start, end, text in backend.transcribe("audio.wav"):
    print(f"[{start:.2f} --> {end:.2f}] {text}")
backend.cleanup()

# TTS
from workers.tts.backends import get_tts_backend

tts = get_tts_backend("edge-tts", voice="de-DE-KatjaNeural")
tts.synthesize("Hallo Welt", "output.wav")
tts.cleanup()

# Translation
from workers.translation.backends import get_translation_backend

tr = get_translation_backend("crispasr:m2m100", model_id="auto", auto_download=True)
print(tr.translate("Hello world", "en", "de"))
tr.cleanup()
```

## Architecture

```
susurrus/
├── cli.py                          # Multi-mode CLI (transcribe/tts/translate/stream/server)
├── config.py                       # Backend maps, TTS config, companion models
├── main.py                         # GUI entry point
├── gui/
│   ├── main_window.py              # 4-tab main window + wiring
│   ├── themes.py                   # Light/dark themes, speaker/confidence colors
│   └── widgets/
│       ├── segment_list_widget.py  # Per-segment output with editing
│       ├── history_panel.py        # History browser tab
│       ├── batch_panel.py          # Batch queue panel
│       ├── waveform_widget.py      # PCM waveform display
│       ├── log_viewer.py           # Real-time log viewer
│       ├── tts_settings.py         # TTS panel
│       ├── translation_settings.py # Translation panel
│       ├── crispasr_advanced_settings.py  # CrispASR options
│       ├── collapsible_box.py      # Collapsible UI section
│       ├── diarization_settings.py
│       ├── voxtral_settings.py
│       └── advanced_options.py
├── workers/
│   ├── transcription/backends/     # Lazy-loaded via get_backend()
│   │   ├── base.py                 # TranscriptionBackend ABC
│   │   ├── crispasr_backend.py     # Subprocess (full PARAM_MAP)
│   │   ├── crispasr_ffi_backend.py # FFI (in-process via libcrispasr)
│   │   ├── faster_whisper_backend.py
│   │   ├── voxtral_backend.py
│   │   └── ...                     # 11 total ASR backends
│   ├── tts/backends/
│   │   ├── base.py                 # TTSBackend ABC
│   │   ├── crispasr_tts_backend.py # CrispASR TTS (34+ engines)
│   │   ├── edge_tts_backend.py
│   │   ├── piper_tts_backend.py
│   │   ├── kokoro_onnx_tts_backend.py
│   │   ├── chatterbox_tts_backend.py
│   │   └── speecht5_tts_backend.py
│   ├── translation/backends/
│   │   ├── base.py                 # TranslationBackend ABC
│   │   └── crispasr_translation_backend.py
│   ├── batch_queue.py              # Sequential multi-file processing
│   ├── tts_thread.py               # QThread for TTS/Translation
│   └── transcription_thread.py     # QThread with progress parsing
└── utils/
    ├── crispasr_utils.py           # Binary discovery, probing, SHA verification, metrics
    ├── export_formats.py           # SRT/VTT/JSON/CSV/TXT export
    ├── history_service.py          # JSON-based transcription history
    ├── progress_parser.py          # CrispASR stderr progress parsing
    ├── segment_model.py            # Segment class with speaker names, editing
    ├── c2pa_signing.py             # C2PA Content Credentials (via c2pa-audio)
    ├── i18n.py                     # English + German translations
    ├── semantic_search.py          # CrispEmbed semantic search (with fallback)
    ├── text_extraction.py          # PDF/EPUB/HTML/MD extraction
    ├── audio_utils.py
    ├── format_utils.py             # SRT/VTT time formatting
    ├── download_utils.py           # Model download helpers
    ├── dependency_check.py         # Optional dependency checks
    └── device_detection.py
```

## Environment Variables

- `CRISPASR_EXECUTABLE`: Path to crispasr binary
- `CRISPASR_N_GPU_LAYERS`: GPU layer offload count for LLM backends
- `CRISPASR_KV_ON_CPU`: Keep KV cache on CPU (set to `1`)
- `HF_TOKEN`: Hugging Face API token (diarization)
- `MISTRAL_API_KEY`: Mistral AI API key (Voxtral API)
- `CUDA_VISIBLE_DEVICES`: GPU selection
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO`: MPS memory optimization

## Testing

```bash
# Run all tests (257 tests)
python -m unittest discover -s tests -v

# Run unit tests only
python -m unittest discover -s tests/unit -v

# Run specific test suite
python -m unittest tests.unit.test_crispasr_params -v
python -m unittest tests.unit.test_tts_backends -v
python -m unittest tests.unit.test_crispasr_ffi -v
```

## License

MIT — see [LICENSE](LICENSE).

**Model licenses vary.** Most ASR models (Whisper, Parakeet, Canary, Voxtral, Qwen3-ASR) are permissive (MIT/Apache/CC-BY). TTS models: Piper (MIT), Kokoro (Apache 2.0), Chatterbox (MIT), SpeechT5 (MIT), Edge TTS (MS ToS). Check individual model cards on HuggingFace for exact terms before commercial deployment.
