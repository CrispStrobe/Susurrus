# Susurrus v2.3.0 — CrispASR 0.8.7 Sync Plan

Syncing Susurrus with CrispASR v0.8.0 → v0.8.7 (HEAD). **All items implemented.**

## 1. New ASR Backends (config.py + BACKEND_MODEL_MAP)

- [x] `ark-asr` — ARK-ASR-3B (Whisper-RoPE + Qwen2.5-3B, 19 languages)
- [x] `higgs-stt` — Higgs Audio v3 (Whisper-large-v3 + Qwen3-1.7B)
- [x] `moss-transcribe` — MOSS-Transcribe-preview-2B
- [x] `gemma4-e4b` — Gemma4 E4B (larger 42L×2560 decoder)
- [x] `parakeet-ctc-ja` — Japanese FastConformer-CTC 1.1B
- [x] `reazonspeech` — Japanese FastConformer-RNNT 619M
- [x] `canary-ctc` — CTC-only variant of canary (fixes #195)
- [x] `qwen3-ja-anime` — Qwen3-ASR-1.7B Japanese anime/galgame fine-tune

## 2. New TTS Backends (config.py + TTS_BACKEND_MAP)

- [x] `tada` — TADA 1B/3B (Llama-3.2 + flow matching + codec)
- [x] `dots-tts` — rednote-hilab 2B continuous AR TTS, 48 kHz
- [x] `bananamind-tts` — Tacotron-lite + HiFi-GAN

## 3. New Translation Backends

- [x] `m2m100-f16` — M2M100 418M F16 (exact HF parity via faithful SP-BPE)

## 4–10: CLI Flags, FFI, Companions, Tests

All implemented — see git history for v2.3.0.

---

# Susurrus v2.4.0 — GUI Feature Parity Plan

Inspired by CrisperWeaver comparison. Priority order by impact/effort.

## P0 — Transcription History (persist + browse)

- [x] `utils/history_service.py` — JSON file-based history persistence
  - HistoryEntry: id, created_at, source_path, backend, model, language, segments, duration, speaker_names
  - Save dir: `~/.local/share/susurrus/history/` (XDG) or QStandardPaths
  - Auto-save on transcription completion
  - Load/list/delete/search (substring)
- [x] `gui/widgets/history_panel.py` — History browser widget
  - List view with metadata (date, file, backend, duration)
  - Click to load transcript into output
  - Delete button per entry
  - Search/filter bar
- [x] Wire into MainWindow as 4th tab or sidebar
- [x] Unit tests: save/load/delete/search round-trip
- [x] Live test: transcribe → verify history entry created

## P0 — Export Formats (SRT, VTT, JSON, CSV)

- [x] `utils/export_formats.py` — format converters
  - `export_srt(segments) → str`
  - `export_vtt(segments) → str`
  - `export_json(segments, metadata) → str`
  - `export_csv(segments) → str`
  - `export_txt(segments) → str`
- [x] GUI: replace plain "Save" with format picker dialog (dropdown: TXT/SRT/VTT/JSON/CSV)
- [x] CLI: `--output-format` flag for batch export
- [x] Unit tests: each format with edge cases (empty, unicode, long segments)

## P1 — Batch Queue (multi-file, sequential processing)

- [x] `workers/batch_queue.py` — BatchJob + BatchQueue
  - BatchJob: file_path, status (queued/running/done/error), progress, result
  - Sequential drain: process next on completion
  - Abort current + clear queue
- [x] GUI: batch panel (drag-drop files or multi-select)
  - Job list with status icons
  - Progress per job
  - Add/remove/retry controls
- [x] Auto-save results to history on completion
- [x] Unit tests: queue logic (enqueue, drain, abort, retry)

## P1 — Progress Callback (deterministic %)

- [x] Parse crispasr stderr progress lines (`progress: 0.45` or `[50%]`)
- [x] Update QProgressBar with actual 0–100% instead of indeterminate
- [x] Show RTF/WPS in progress area during transcription
- [x] Unit tests: progress line parsing

## P1 — Inline Segment Editing + Speaker Names

- [x] Make transcription output editable (QTextEdit or segment list widget)
- [x] Per-segment view with optional speaker label, timestamp, confidence
- [x] Speaker name remapping: "Speaker 1" → user-provided name
- [x] Edited flag on segments (for history persistence)
- [x] Unit tests: segment model, rename, edit

## P2 — Waveform Display

- [x] `gui/widgets/waveform_widget.py` — simple waveform from PCM/WAV
  - Load audio samples, downsample for display
  - Playback position indicator
  - Segment highlight regions (from timestamps)
- [x] Integrate below audio file input in transcription tab
- [x] Unit tests: sample loading, downsampling

## P2 — Live Streaming in GUI

- [x] Mic capture via sounddevice or PyAudio (16kHz mono)
- [x] Pipe to CrispASR `--stream --mic` subprocess
- [x] Real-time segment updates in output panel
- [x] Start/Stop recording button
- [x] Unit tests: mock stream, segment parsing

## P2 — Light Theme + Confidence Colors

- [x] `gui/themes.py` — light + dark theme definitions
  - Light: white bg, dark text, blue accents
  - Dark: current theme
  - Toggle in menu or settings
- [x] Speaker color palette (8 distinct colors, cycle)
- [x] Confidence color coding: >=0.8 green, >=0.6 orange, <0.6 red
- [x] Persist theme choice in QSettings
- [x] Unit tests: theme application, color mapping

## P3 — Voice Clone Wizard

- [x] 3-step dialog: capture/select audio → enter/transcribe ref text → hand off to TTS tab
- [x] Pre-populate TTS settings with reference audio + text
- [x] Unit tests: wizard state transitions

## P3 — i18n (German)

- [x] Extract all user-visible strings to a translations dict
- [x] German translation file
- [x] Language selector in settings or menu
- [x] Unit tests: string lookup, fallback to English

## P3 — Log Viewer

- [x] `gui/widgets/log_viewer.py` — real-time log display
  - Ring buffer (last 1000 entries)
  - Level filter (DEBUG/INFO/WARNING/ERROR)
  - Search bar
- [x] Accessible from Help menu or as a panel
- [x] Unit tests: buffer append, filter, search
