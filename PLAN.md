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

---

# Susurrus v2.6.0 — GUI Wiring & Polish Plan (DONE)

## W1 — Wire progress parser into TranscriptionThread (HIGH)

- [x] TranscriptionThread: read stderr, call `parse_progress_line()` on each line
- [x] New signal `progress_percent_signal = pyqtSignal(float)` (0.0–1.0)
- [x] MainWindow: connect signal → `QProgressBar.setValue(int(pct * 100))`
- [x] Switch `QProgressBar` from indeterminate (0,0) to determinate (0,100)
- [x] Display RTF/WPS in metrics panel when parsed
- [x] Unit test: mock stderr lines → verify signal emission
- [x] Fallback: stay indeterminate if no progress lines received

## W2 — Wire segment model into transcription output (HIGH)

- [x] Replace `_transcription_segments` list-of-tuples with `TranscriptionResult`
- [x] Parse speaker labels from `[Speaker 1]` prefixes in output lines
- [x] Use `TranscriptionResult` in save_transcription (all export formats)
- [x] Use `TranscriptionResult` in auto-save to history
- [x] Unit test: parsing output lines with speaker labels → correct Segments

## W3 — Place batch panel + waveform widget in transcription tab (HIGH)

- [x] Add `BatchPanel` below the output area (or as collapsible section)
- [x] Wire `BatchQueue` → `BatchPanel.set_queue()`
- [x] Add `WaveformWidget` below the audio file input row
- [x] Load waveform on file selection (`audio_input_path.textChanged`)
- [x] Highlight segment regions from `_transcription_segments`
- [x] Unit test: batch panel add/remove (mock queue)

## W4 — Pin dev tool versions to match CI (QUICK FIX)

- [x] `pyproject.toml [dev]`: pin `black==25.9.0`, `isort==6.1.0`, `ruff==0.15.7`, `bandit==1.9.4`
- [x] Update local dev env to match
- [x] Verify `python -m black --check .` passes with pinned version

## W5 — Persist settings across restarts (MEDIUM)

- [x] Save/restore theme choice ("dark"/"light") in QSettings
- [x] Save/restore last-used backend, model, language in QSettings
- [x] Save/restore last-used TTS backend, voice in QSettings
- [x] Load persisted values in `__init__` before UI setup
- [x] Unit test: QSettings round-trip (mock)

## W6 — Drag-and-drop into batch queue (MEDIUM)

- [x] Extend `dropEvent` to detect multi-file drops
- [x] First file → audio_input_path (existing behavior)
- [x] Additional files → batch queue
- [x] Visual feedback: highlight batch panel on drag hover
- [x] Unit test: drop event with multiple URLs

## W7 — Keyboard shortcuts (MEDIUM)

- [x] F5 = Transcribe (already wired in menu)
- [x] Ctrl+S = Save (already wired)
- [x] Ctrl+Shift+S = Save As (with format picker)
- [x] Ctrl+H = Switch to History tab
- [x] Ctrl+T = Toggle Light/Dark theme
- [x] Arrow Up/Down in segment list = navigate segments
- [x] Document shortcuts in Help → Keyboard Shortcuts dialog

## W8 — Server mode toggle in GUI (LOWER)

- [x] Settings dialog or Tools menu: "Start Server" toggle
- [x] Port field (default 8080)
- [x] Uses `CrispasrBackend.start_server()` in background thread
- [x] Status indicator: "Server running on :8080"
- [x] Stop button

## W9 — About dialog + README refresh (LOWER)

- [x] Update About dialog with current version, feature counts
- [x] Update README feature counts, architecture diagram
- [x] Add "What's New" section to README or link to releases

---

# Susurrus v2.7.0 — Real-World Usability & Testing Plan

## R1 — Run the GUI and fix what's broken (HIGHEST)

- [ ] Launch `python main.py`, check window renders without errors
- [ ] Fix any import errors, missing widgets, layout issues
- [ ] Test: select audio file → waveform loads
- [ ] Test: transcribe → progress bar works, output appears, segments stored
- [ ] Test: save → format picker dialog, each format produces valid file
- [ ] Test: History tab → entries appear, search works, load works
- [ ] Test: toggle theme → both themes render correctly
- [ ] Test: drag-drop file → input populated
- [ ] Test: View → Show Logs → log viewer dialog appears
- [ ] Fix all issues found; document any that require display-dependent fixes

## R2 — End-to-end CLI integration test (HIGH)

- [ ] `tests/integration/test_cli_e2e.py` — end-to-end CLI tests
- [ ] Test: `cli.py --backend crispasr --model auto:q5_0 --file jfk.wav` → output contains words
- [ ] Test: `cli.py --output-format srt --file jfk.wav` → valid SRT output
- [ ] Test: `cli.py --output-format json --file jfk.wav` → valid JSON with segments
- [ ] Test: `cli.py --mode align --text-file ref.txt --file jfk.wav` → runs or skips gracefully
- [ ] Test: `cli.py --list-backends` → lists backends without error
- [ ] All tests: ≤120s timeout, CPU-only, auto-skip without binary/audio
- [ ] Auto-skip if no crispasr binary or no test audio file

## R3 — Segment list view (replaces QPlainTextEdit) (HIGH)

- [ ] `gui/widgets/segment_list_widget.py` — custom widget
  - One row per segment: [speaker chip] [timestamp] [editable text] [confidence badge]
  - Speaker chip colored from gui/themes.speaker_color()
  - Confidence badge colored from gui/themes.confidence_color()
  - Double-click text to edit inline → sets segment.edited = True
  - Right-click context menu: rename speaker, copy text, delete segment
  - Arrow key navigation between segments
- [ ] Replace `self.transcription_output` (QPlainTextEdit) with SegmentListWidget
- [ ] Wire to TranscriptionResult for data, export_formats for save
- [ ] Fallback: show plain text if no structured segments available
- [ ] Unit test: add/edit/delete segments, speaker rename, keyboard nav

## R4 — CI integration test job (DONE)

- [x] `.github/workflows/ci.yml`: integration job added
- [x] continue-on-error: true (advisory)

---

# Susurrus v2.9.1+ — EU AI Act Compliance & Future Plan

## C1 — EU AI Act Provenance Flags (DONE — v2.9.1)

- [x] CLI: `--i-have-rights`, `--no-spoken-disclaimer`, `--watermark-model`,
      `--no-watermark`, `--detect-watermark`, `--c2pa-cert`, `--c2pa-key`
- [x] CLI `_build_crispasr_kwargs`: all 7 flags wired through
- [x] `--detect-watermark` as standalone verb (run and exit)
- [x] `--no-watermark` warning: logs EU AI Act Art. 50 responsibility shift
- [x] TTS backend: `no_watermark`, `c2pa_cert`, `c2pa_key` kwargs + warning
- [x] Unit test: full provenance flag set in PARAM_MAP
- [x] CrispASR defaults: watermark ON, C2PA signing ON (bundled cert)
- [x] Susurrus does NOT override these defaults — compliance by default

## C2 — c2pa-audio Python Integration (DONE — v2.9.2)

- [x] `utils/c2pa_signing.py` — sign/verify via c2pa-audio ctypes
- [x] `TTSBackend.sign_output()` — post-synthesis signing hook
- [x] `--verify-c2pa` CLI flag
- [x] Unit tests: 6 tests (import, fallback, sign_output, non-WAV skip)
- [x] Live tests: 5 provenance flag acceptance tests

See https://github.com/CrispStrobe/c2pa-audio (160 KB, Python ctypes)

## C3 — Watermark Detection in GUI (DONE — v2.10.0)

- [x] "Detect Watermark" button in transcription tab
- [x] Runs `--detect-watermark` on loaded audio, shows result dialog

## F1 — GUI: CrispASR advanced settings for provenance (DONE)

- [x] C2PA cert/key file pickers in TTS settings
- [x] "Disable watermark" checkbox with Art. 50 tooltip
- [x] "Voice Cloning Consent" checkbox (was already present)
- [x] "Skip AI Disclaimer" checkbox (was already present)

## F2 — Live streaming in GUI (DONE — v2.10.0)

- [x] "Stream Mic" button in transcription tab
- [x] Pipes to CrispASR `--stream --mic` subprocess
- [x] Real-time text output, Start/Stop toggle

## F3 — Voice clone wizard (DONE — v2.10.0)

- [x] 3-step dialog: select audio → enter ref text → confirm with consent
- [x] Pre-populates TTS tab with reference audio + i_have_rights
- [x] Accessible from Tools → Voice Clone Wizard

## F4 — i18n (DONE — v2.10.0)

- [x] `utils/i18n.py` — string lookup with English fallback
- [x] German translation (90+ strings)
- [x] 7 unit tests (locale switching, fallback, key parity, Art. 50)

## F5 — Server mode in GUI (DONE — v2.10.0)

- [x] Tools → Start/Stop Server (toggle)
- [x] Starts CrispASR `--server` on port 8080

## F6 — CrispEmbed integration (DONE — v2.10.0, stub)

- [x] `utils/semantic_search.py` — semantic_search() with substring fallback
- [x] Falls back to substring when CrispEmbed binary not available
- [x] 5 unit tests (import, empty, substring, title scoring)
