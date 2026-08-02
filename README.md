# Susurrus: Audio Transcription, TTS & Translation Suite

Susurrus is a professional, modular audio suite providing transcription, text-to-speech, translation, and speech-to-speech through a unified GUI and CLI. Built with a clean architecture, it supports 48+ CrispASR ASR sub-backends, 50+ CrispASR TTS engines, multi-language translation, speaker diarization, EU AI Act compliance, and extensive customization options.

### Part of the Crisp ecosystem

| Project | Role |
|---|---|
| **Susurrus** | This repo — Python GUI + CLI with 48+ CrispASR ASR sub-backends, 50+ CrispASR TTS engines, translation, S2S |
| **[CrispASR](https://github.com/CrispStrobe/CrispASR)** | C++ ASR/TTS engine (v0.8.22) — 48+ ASR + 50+ TTS backends, ggml inference. Two integration paths: subprocess (binary) or FFI (libcrispasr Python bindings) |
| **[CrisperWeaver](https://github.com/CrispStrobe/CrisperWeaver)** | Flutter transcription app powered by CrispASR — desktop + mobile, fully offline |
| **[CrispTTS](https://github.com/CrispStrobe/CrispTTS)** | Python TTS suite — 20+ handlers, German focus |
| **[CrispEmbed](https://github.com/CrispStrobe/CrispEmbed)** | Text embedding engine (ggml) — XLM-R, Qwen3-Embed, Gemma3, dense + sparse + ColBERT |

## Features

### Transcription (48+ CrispASR ASR sub-backends)

- **CrispASR engine** (48+ ASR sub-backends): whisper, parakeet, canary, canary-ctc, canary-qwen, cohere, cohere-ar, qwen3, qwen3-1.7b, qwen3-ja-anime, mega-asr, voxtral, voxtral4b, granite, granite-4.1, granite-4.1-plus, granite-4.1-nar, moonshine, moonshine-streaming, kyutai-stt, kyutai-stt-2.6b, fastconformer-ctc, wav2vec2, hubert, data2vec, vibevoice, firered-asr, funasr, fun-asr-mlt-nano, paraformer, sensevoice, glm-asr, omniasr, omniasr-300m, omniasr-llm, omniasr-llm-1b, mimo-asr, moss-audio, moss-transcribe, moss-diarize, gemma4-e2b, gemma4-e4b, nemotron, mini-omni2, ark-asr, higgs-stt, parakeet-ctc-ja, reazonspeech, and more
- **CrispASR FFI** (`crispasr-ffi`): In-process inference via Python ctypes to libcrispasr — zero IPC overhead, persistent model sessions, native word-level timestamps and confidence scores
- **CrispASR subprocess** (`crispasr`): Binary execution with full parameter passthrough — works with just the binary, no shared library needed
- **Python backends**: mlx-whisper, faster-whisper (batched + sequenced), transformers, whisper.cpp, ctranslate2, whisper-jax, insanely-fast-whisper, OpenAI Whisper, Voxtral (local + API)
- **Flexible Input**: Local files, URLs, video audio extraction
- **Audio Format Support**: MP3, WAV, FLAC, M4A, AAC, OGG, OPUS, WebM, MP4, WMA
- **Language Detection**: Automatic or manual, multiple LID backends (whisper, silero, firered, ecapa)
- **Word-level Timestamps**: Native or CTC aligner-based
- **Performance Metrics**: Real-time factor (RTF) and words-per-second (WPS)
- **Backend Availability Probing**: Auto-detects which CrispASR backends are compiled in

### Text-to-Speech (50+ CrispASR engines)

- **CrispASR TTS** (50+ C++ backends): kokoro, orpheus, qwen3-tts, qwen3-tts-customvoice, qwen3-tts-1.7b-base, qwen3-tts-1.7b-customvoice, qwen3-tts-1.7b-voicedesign, chatterbox, chatterbox-turbo, kartoffelbox-turbo, lahgtna-chatterbox, vibevoice, vibevoice-1.5b, miotts, indextts, voxcpm2-tts, melotts, piper, bark, bark-tts, dia, dia-tts, zonos, zonos-tts, csm, csm-tts, sesame, cosyvoice3-tts, f5-tts, fastpitch, parler-tts, outetts, pocket-tts, speecht5, kugelaudio, lex-au-orpheus-de, kartoffel-orpheus-de-natural, kartoffel-orpheus-de-synthetic, lfm2-audio, mini-omni2, tada, tada-1b, tada-tts-1b, tada-3b-ml, dots-tts, bananamind-tts, bananamind-tts-de, voxtral-tts, omnivoice, omnivoice-singing, irodori-tts, irodori-tts-voicedesign, moss-tts, moss-tts-local
- **Python-native TTS** (5 backends): Edge TTS (cloud), Piper (MIT, ONNX), Kokoro ONNX (Apache 2.0), Chatterbox (MIT), SpeechT5 (MIT)
- **Voice cloning**: Reference audio support for applicable backends
- **Text extraction**: Load text from TXT, Markdown, HTML, PDF, EPUB files for synthesis
- **Voice selection**: Per-backend voice lists with configurable presets
- **Local playback**: `--tts-play` for direct speaker output

### Speech-to-Speech

- **Audio-in → Audio-out**: Supported by lfm2-audio and mini-omni2 backends
- **FFI support**: `speech_to_speech()` method for in-process S2S with optional intermediate transcript — this is the supported route, and it marks its output per EU AI Act Art. 50
- **Not a CLI flag**: there is no `susurrus --s2s`. The subprocess backend accepts `s2s=True` programmatically and passes `--s2s` to the binary, but that route is unverified — Susurrus does not inspect the binary's `--s2s-output`. Prefer the FFI route.

### Translation

- **CrispASR translation**: m2m100 (100 languages), m2m100-f16 (exact HF parity), MadLad (419 languages), Gemma4-E2B (140+ languages)
- **Bidirectional**: Any source → any target language pair

### Speaker Diarization

- **PyAnnote.audio**: State-of-the-art neural diarization (requires HF token)
- **CrispASR methods**: energy, xcorr, vad-turns, pyannote, sherpa, ecapa
- **Language-specific models**: English, German, Chinese, Spanish, Japanese
- **Configurable**: Min/max speaker counts, cluster thresholds

### Advanced CrispASR Features

- **VAD**: Silero, FireRed, configurable thresholds, plus VAD segment export/import for reproducible chunking
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

See **[COMPLIANCE.md](COMPLIANCE.md)** for the full obligations map — what the
software does for you, and what remains yours to do as provider or deployer.

- **Marking fails closed**: if no marking layer can be applied, the output is
  **deleted** and the run refused (exit 2) — Susurrus does not emit unmarked
  synthetic audio. WAV and MP3 always succeed (the declarative marker is pure
  standard library); exotic containers need C2PA or soundfile. A cheap
  preflight refuses before any model loads. The one way past it is
  `--accept-marking-responsibility`
- **Marking ON by default, on every TTS path**: no backend emits unmarked WAV
  or MP3
- **Marking is verified, not assumed**: on the CrispASR routes the binary does
  the marking, so Susurrus reads the finished file back and reports what is
  actually present — applying the declarative marker as a floor if nothing is
- **C2PA signing**: Content Credentials via
  [c2pa-python](https://pypi.org/project/c2pa-python/), included in the `tts`
  extra (`pip install 'susurrus[c2pa]'` on its own). A local signing identity
  is generated on first use; pass `--c2pa-cert`/`--c2pa-key` for your own
- **Dependency-free fallback marker**: a RIFF `LIST/INFO` chunk (WAV) or an
  ID3v2.4 tag (MP3) declaring AI generation is embedded even when every
  optional library is absent, so a default install still satisfies Art. 50(2)
- **Voice cloning gated everywhere**: an explicit rights attestation is
  required before cloning on *every* route — Python-native backends, both
  CrispASR routes, the in-process FFI route and speech-to-speech — refused
  before any model loads. Susurrus never sets the attestation on your behalf
- **Spoken disclosure on every backend**: cloned audio gets an audible,
  localized AI-disclosure prefix — CrispASR in-binary, Python-native backends
  by synthesizing the phrase with the same model, in the backend's own voice
  rather than the cloned one, in whatever container the output uses. If it
  cannot be delivered, the cloned audio is refused and deleted: Art. 50(4) is
  enforced separately from Art. 50(2), since a listener hears no metadata
- **In-sample watermark, always on**: a numpy-only spread-spectrum comb
  survives re-encoding where metadata does not, so no install is left with
  metadata as its only durable mark. `pip install 'susurrus[watermark]'`
  upgrades it to AudioSeal, which also resists deliberate removal
- **Art. 12 audit log**: every speaker enrollment *and* identification is
  recorded to a hash-chained append-only log. `susurrus --audit-log` prints it
  and verifies the chain; Tools → Biometric Audit Log in the GUI
- **`--accept-marking-responsibility`**: the explicit opt-out that produces
  unmarked audio, and the only thing that disarms the fail-closed gate. The
  narrower flags (`--no-watermark`, `--no-c2pa`, `--no-spoken-disclaimer`)
  each require it too, so reducing provenance is always a deliberate, attested
  act
- **`--detect-watermark`**: Standalone AI-content detection (confidence + verdict)
- **`--verify-c2pa`**: Check whether a file is marked as AI-generated —
  reports both the C2PA credentials and the declarative marker, exits 0 if
  either is present
- **Biometric warning**: using the speaker database without `--speaker-db-consent`
  warns about GDPR Art. 9 and possible Annex III(1)(a) high-risk classification
- **AI-literacy notice**: Help → About AI in Susurrus states the intended
  purpose, the known failure modes and what the system is not validated for
  (Art. 4), localized like the rest of the interface

### Intended Purpose & Limitations

Susurrus is a local-first tool for transcribing audio, synthesizing speech,
translating text, and separating speakers — for individuals and teams
processing their own or consented material.

**Output is a model prediction, not a record.** Transcription accuracy varies
sharply with accent, audio quality, background noise, domain vocabulary and
language; non-native accents and under-resourced languages typically fare
worse. Diarization guesses speaker boundaries and counts, and struggles with
overlapping or similar voices. Translation loses nuance and can invert meaning.
Review all output before relying on it.

**Not validated for** uses where an error carries legal or safety
consequences without human review — evidentiary transcripts, medical
documentation, employment or education decisions, law enforcement, or border
control. Several are Annex III high-risk areas whose obligations this project
does not implement. See [COMPLIANCE.md](COMPLIANCE.md).

**Speaker enrollment stores biometric data.** `--enroll-speaker` and
`--speaker-db` persist voice embeddings linked to named people — GDPR Art. 9
special-category data requiring a lawful basis.

### GUI

- **4-tab layout**: Transcription / Text-to-Speech / Translation / History
- **Segment list view**: Per-segment display with speaker color chips, confidence badges, inline editing
- **Waveform display**: PCM visualization with segment highlights, auto-loads on file selection
- **Live mic streaming**: "Stream Mic" button — real-time transcription from microphone
- **Watermark detection**: "Detect Watermark" button — check if audio is AI-generated
- **Batch queue**: Multi-file sequential processing with status tracking
- **History browser**: Search, load, delete past transcriptions (auto-saved)
- **Voice clone wizard**: 3-step guided dialog with a required EU AI Act consent
  checkbox — "Clone Voice" stays disabled until the attestation is given (Tools menu)
- **Server toggle**: Start/stop OpenAI-compatible HTTP server from Tools menu
- **Light/dark themes**: Toggle via Ctrl+T, persisted in QSettings
- **Log viewer**: Real-time log display with level filtering (View → Show Logs)
- **i18n**: full English + German interface (259 strings, no hardcoded text) —
  View → Language, persisted across sessions, defaults to the system language
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
│   │   ├── crispasr_tts_backend.py # CrispASR TTS (50+ engines)
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
    ├── c2pa_signing.py             # C2PA Content Credentials (c2pa-python)
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
