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

## C2 — c2pa-audio Python Integration (PARTIAL — see A1)

- [x] `utils/c2pa_signing.py` — sign/verify via c2pa-audio ctypes
- [x] `TTSBackend.sign_output()` — post-synthesis signing hook *defined*
- [x] `--verify-c2pa` CLI flag
- [x] Unit tests: 6 tests (import, fallback, sign_output, non-WAV skip)
- [x] Live tests: 5 provenance flag acceptance tests
- [!] **Never called.** The hook had zero production callers until A1; the
      tests only asserted the method existed. Fixed in A1.

See https://github.com/CrispStrobe/c2pa-audio (160 KB, Python ctypes)

## C3 — Watermark Detection in GUI (DONE — v2.10.0)

- [x] "Detect Watermark" button in transcription tab
- [x] Runs `--detect-watermark` on loaded audio, shows result dialog

## F1 — GUI: CrispASR advanced settings for provenance (PARTIAL — see A3)

- [x] C2PA cert/key file pickers in TTS settings
- [x] "Disable watermark" checkbox with Art. 50 tooltip
- [x] "Voice Cloning Consent" checkbox (was already present)
- [x] "Skip AI Disclaimer" checkbox (was already present)
- [!] **Widgets rendered but inert.** `TTSThread` forwarded only three keys,
      so the watermark checkbox and both cert pickers did nothing. Fixed in A3.

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

---

# Susurrus v2.11.0 — EU AI Act Compliance Remediation Plan

Findings from the 2026-08-01 compliance audit. C2 and F1 above were marked
DONE but shipped disconnected code — the hooks exist, nothing calls them.
This section supersedes those claims.

**Scope note.** The MIT licence does *not* exempt this project: Art. 2(12)
excludes free/open-source AI systems from most of the Regulation *except*
those falling under Art. 5 or Art. 50. A TTS suite with voice cloning is an
Art. 50 system, so the transparency obligations apply in full.

## A1 — Wire `sign_output()` into the synthesis paths (CRITICAL)

`workers/tts/backends/base.py:34` defines the C2PA hook; it has zero
production callers. edge-tts / piper / kokoro-onnx / speecht5 / chatterbox
therefore emit synthetic audio with no machine-readable marking (Art. 50(2)).

- [x] Renamed to `TTSBackend.apply_provenance()` — it now applies two marking
      layers, so `sign_output` no longer described what it did
- [x] `workers/tts_thread.py`: called after `synthesize()` on both branches
- [x] `cli.py:_run_tts`: same on both branches
- [x] Report marking status to the user (`_describe_marking`, `_report_marking`)
- [x] `CrispasrTTSBackend.apply_provenance()` overrides to a no-op — the binary
      marks its own output, re-signing would stack a second manifest
- [x] Replaced the `hasattr(TTSBackend, "sign_output")` test with
      `tests/unit/test_provenance_wiring.py`, which asserts the call happens

## A2 — Gate voice cloning on the Python path (CRITICAL)

`chatterbox_tts_backend.py:55` clones from an arbitrary WAV via
`audio_prompt_path` with no consent gate. `i_have_rights` is only consulted
on the `crispasr` branch (`tts_thread.py:43-59`); the Python branch drops it.

- [x] `TTSBackend.require_clone_consent()` in `base.py` — raises
      `PermissionError` when a reference audio is used without attestation
- [x] `ChatterboxTTSBackend.synthesize()`: gate moved to the *top* of the
      method, before the torch import and `from_pretrained()`. Both cloning
      routes (`voice=` and `reference_audio=`) converge there, so one gate
      covers both, and a refusal costs no model download
- [x] `cli.py` + `tts_thread.py`: pass `i_have_rights` / `reference_audio`
      through to Python-native backends
- [x] CLI exits 2 on refusal instead of surfacing a traceback; the GUI thread
      routes `PermissionError` to `error_signal` as a refusal, not a crash
- [x] `tests/unit/test_clone_consent.py` — 7 tests, including one asserting
      the gate fires before any model is fetched

## A3 — Wire the inert GUI provenance controls (CRITICAL)

`main_window.py:1118-1120` collects `no_watermark`, `c2pa_cert`, `c2pa_key`;
`tts_thread.py:53-59` forwards only three unrelated keys. The checkbox and
both file pickers are no-ops.

- [x] `tts_thread.py`: single `provenance` dict forwarded to *both* branches,
      so a new control can't be added to one path and forgotten on the other
- [x] Added "Disable C2PA" + "Accept marking responsibility" checkboxes to
      `gui/widgets/tts_settings.py` (were CLI-only)
- [x] `c2pa_signing._resolve_pem()` — the GUI/CLI supply cert *paths* while
      the c2pa-audio API wants PEM *text*; the pickers were mismatched even
      once wired
- [x] Python path honours `no_c2pa`; `no_watermark` warns via i18n
- [x] Unit test: every provenance widget reaches the backend kwargs

## A4 — Make wizard consent an actual attestation (HIGH)

`voice_clone_wizard.py:79-84` renders consent as a passive `QLabel`, then
`main_window.py:1294` sets `i_have_rights` automatically — the app asserts
the legal attestation on the user's behalf.

- [x] Replaced the label with a required `QCheckBox`; "Clone Voice" is
      disabled until it is ticked
- [x] Exposed `wizard.consent_given`; `i_have_rights` is set from it rather
      than unconditionally
- [x] Added a "Reference text" field to the TTS panel and wired
      `wizard.ref_text` to it — it was previously discarded into a log line
- [x] `ref_text` passed through `tts_thread` to the CrispASR backend
- [x] Verified headlessly: the wizard cannot be completed without consent

## A5 — Machine-readable AI marking that survives a missing library (HIGH)

C2PA signing degrades to a silent no-op when `c2pa-audio` is absent
(`utils/c2pa_signing.py:34-37`), which would leave output unmarked on a
default install. Art. 50(2) has no "unless a dependency is missing" clause.

- [x] `utils/ai_marking.py` — embeds a RIFF `LIST/INFO` chunk
      (`ISFT`/`ICMT`/`IENG`/`ITCH`), readable by any RIFF parser
- [x] Applied on every Python-path synthesis, alongside C2PA rather than
      instead of it — the two layers are complementary
- [x] `read_wav_ai_marker()` / `is_ai_marked()` for verification, surfaced via
      `--verify-c2pa`, which now reports both layers and exits 0 if either
      marks the file. **Behaviour change:** it previously exited 1 whenever
      `c2pa-audio` was absent, conflating "cannot check" with "not marked"
- [x] Verified with `ffprobe`: the marker reads as standard format tags in
      third-party tools, and the audio stream is byte-identical
- [x] Documented in COMPLIANCE.md that the *spoken* disclaimer is CrispASR-only
- [x] `tests/unit/test_ai_marking.py` — 9 tests: round-trip, idempotency,
      RIFF size-header correctness, word alignment on odd-length payloads,
      audio payload byte-identical after marking, no temp files left behind

## A6 — Surface the speaker-DB biometric path (HIGH)

`crispasr_backend.py:155-160` exposes `--speaker-db`, `--enroll-speaker`,
`--titanet-model`, `--speaker-db-consent`. Storing voice embeddings keyed to
named people is biometric identification — potentially Annex III(1)(a)
high-risk, and GDPR Art. 9 special-category data. `--speaker-db-consent` is
passed through as an unexplained boolean with no UI and no documentation.

- [x] `cli.py:_warn_speaker_biometrics()` — warns when `--speaker-db`,
      `--enroll-speaker` or `--expect-speakers` is used without consent
- [x] Help text now states what is being attested (GDPR Art. 9), not just
      "GDPR consent for persistent speaker database"
- [x] README + COMPLIANCE.md sections on the biometric path, Annex III(1)(a)
      classification, and deployer responsibilities
- [x] Unit tests: warning fires on each of the three flags, stays silent when
      attested or unused

## A7 — Write the missing Art. 13 / Art. 4 documentation (HIGH)

README has no statement of intended purpose, no limitations, no accuracy
disclosure, no human-oversight guidance, no AI-literacy material.

- [x] README "Intended Purpose & Limitations" section, including how accuracy
      degrades by accent, audio quality and language
- [x] Corrected the Python-native signing claim in the README compliance list
- [x] Noted that transcripts/translations are model predictions, not records
- [x] `COMPLIANCE.md` — obligations map, applicability dates, provider vs.
      deployer split, Art. 2(12) FOSS analysis, and an explicit "what Susurrus
      does not do for you" list
- [x] Corrected the C2/F1 claims above to reflect shipped reality

---

# Susurrus v2.12.0 — Closing the Remaining Compliance Gaps

The v2.11.0 work left four gaps recorded as "known limitations". This section
closes them. Order matters: B3 runs last so strings added by B1/B2 are
translated once, not twice.

## B4 — Art. 12 record-keeping for biometric events (HIGH)

Art. 12 requires high-risk systems to automatically record events over their
lifetime. A deployment using the speaker database may land in Annex III(1)(a),
and today Susurrus records nothing — a deployer cannot show who was enrolled,
when, or under what attestation.

- [x] `utils/audit_log.py` — append-only JSONL at
      `~/.local/share/susurrus/audit/biometric.jsonl` (XDG), one event per line
- [x] Records UTC timestamp, event type, speaker, database path, consent
      attestation, embedding model, Susurrus version
- [x] Logs enrollment *and* identification — Art. 12 covers use, not just setup
- [x] Never records the embedding or audio; a test greps the written file for
      biometric field names so a future edit cannot quietly add one
- [x] SHA-256 hash chain; detects modification, deletion, truncation *and*
      reordering
- [x] `--audit-log` CLI verb (exit 1 on a broken chain) + Tools → Biometric
      Audit Log in the GUI
- [x] Wired into `cli.py` alongside `_warn_speaker_biometrics`
- [x] `tests/unit/test_audit_log.py` — 19 tests
- [x] Fixed `__init__.py` version (was a stale 1.1.0 vs pyproject's 2.11.0) —
      it is recorded into every audit entry

## B1 — Spoken disclosure on Python-native TTS (HIGH)

Art. 50(4) disclosure is currently machine-readable only on the Python path;
CrispASR prepends an audible prefix in-binary. Close it by synthesizing the
disclosure with the same backend and concatenating.

- [x] `utils/spoken_disclosure.py` — synthesizes the phrase with the *same*
      backend/model, so sample rate and channel count align by construction
- [x] Recursion guard via a `_synthesizing_disclosure` flag
- [x] Refuses to concatenate mismatched WAV formats; a backend that cannot
      speak the disclosure loses the prefix, never the user's audio
- [x] Localized via `t("disclosure.spoken", locale=...)`
- [x] Cloning only (matching CrispASR), suppressible with
      `--no-spoken-disclaimer`
- [x] `tests/unit/test_spoken_disclosure.py` — 21 tests, including one that
      asserts the disclosure *precedes* the content (a trailing disclosure is
      not a disclosure)

## B2 — Neural watermark on Python-native TTS (HIGH)

The declarative marker satisfies Art. 50(2) machine-readability but is
metadata: strippable, and lost on re-encode. AudioSeal survives both.

- [x] `utils/audio_watermark.py` — AudioSeal embed/detect, lazily imported,
      with the failed-load result cached so a missing model is not re-fetched
      on every synthesis
- [x] Applied in `apply_provenance()` before C2PA signing; a test asserts the
      call order, since watermarking mutates samples that C2PA then hashes
- [x] Honours `no_watermark` / `accept_marking_responsibility`
- [x] `--detect-watermark` now falls back to the Python detector when no
      CrispASR binary is present, and exits 2 for "could not check" so it is
      not read as "not AI-generated"
- [x] `audioseal` added as a `pyproject.toml` extra, not a hard dependency
- [x] `tests/unit/test_audio_watermark.py` — 14 tests.
      **The live embed/detect round-trip is skipped here: `audioseal` is not
      installed in this environment, so that path is wired and unit-tested but
      not executed.**

## B3 — Complete the GUI internationalization (MEDIUM)

~440 user-visible strings are hardcoded English across 19 GUI files, so a
German user sees German only for the handful of consent strings added in
v2.11.0. Consent and compliance text especially must be in the user's language.

- [x] Split into `utils/translations/{en,de}.py`; `utils/i18n.py` keeps its
      public API and re-exports `TRANSLATIONS`
- [x] Migrated every user-visible string across 15 GUI files — 259 keys.
      Log messages stay English (developer-facing)
- [x] German translation for every key
- [x] View → Language selector, persisted in QSettings, defaulting to the
      system language via `detect_system_locale()`
- [x] `tests/unit/test_i18n.py` rewritten — 18 tests. Beyond key parity it
      asserts format placeholders match across locales (a dropped `{count}`
      would raise KeyError in German only), that compliance strings are
      actually translated rather than copied, and — via an AST scan of the
      whole `gui/` tree — that **no hardcoded user-visible string remains**.
      That last test is what found the 11 multi-line strings literal matching
      missed, and it is what stops the next new widget reintroducing English

## B5 — Fallout found while closing B1–B4

Not planned; recorded because each was a real defect the work surfaced.

- [x] `--detect-watermark` delegated to the CrispASR binary whenever one was
      *present*, and returned its exit code even when it crashed. A stale local
      build made the verb exit 250 with a dyld error instead of answering.
      It now falls through to the Python detector when the binary cannot
      produce a verdict, and distinguishes exit 1 ("not marked") from exit 2
      ("could not check") so an inconclusive result is never read as clean
- [x] The detector report named a single `detector` field covering two
      independent checks; split into `neural_detector` + `ai_marker`
- [!] **Known wart:** the two detector paths have different exit-code
      semantics. The binary path passes the binary's own code through
      (pre-existing behaviour); the Python fallback uses 0/1/2. Unifying them
      would mean parsing the binary's stdout for its verdict, which breaks on
      any output-format change — so the difference is documented in `--help`
      instead. Scripts that branch on the exit code should pin one path
- [x] `apply_provenance()` gained `voice`/`locale` parameters and two result
      keys, which broke four v2.11.0 tests. Fixed — noting it here because the
      signature is now implemented by four classes and two call sites

# Susurrus v2.13 — Audits C1–C6

Six audit rounds, each triggered by "is it done?" rather than by a plan. Every
one found real defects, and three of them found defects in code the previous
round had just written. Recorded in that order because the pattern is the
finding: reading the source of truth beats reasoning about it, and running the
thing beats reading it.

## C1 — Marking that did not depend on a guess

- [x] The declarative floor on the CrispASR routes was applied only when the
      detectors came up empty, which made an Art. 50(2) guarantee conditional
      on the in-sample detector being right. It was not: measured over 1500
      clips of unwatermarked speech, tones and noise it read ~12% of them as
      watermarked, and one false positive suppressed the floor and shipped
      unmarked audio under a printed "Marked as AI-generated". The floor is now
      unconditional — idempotent, dependency-free, does not touch the samples
- [x] Fixed the detector separately, because "we no longer depend on it" is not
      a reason to leave it wrong: weighted correlation instead of bare sign
      agreement, legacy band gated behind a stricter bar. FP 12% → 1.2%, TP
      93% → 98%, threshold 0.65 → 0.78
- [x] `--s2s` produced synthetic audio with no consent gate and no marking
      check — the last route whose only check lived in the binary
- [x] `susurrus --about-ai` renders the Art. 4 notice from the same localized
      source as the GUI dialog

## C2 — The three routes documented as uncovered

- [x] **Server mode is in the response path.** `utils/marking_proxy.py` binds
      the requested port, starts the binary on loopback, marks audio responses
      and refuses what it cannot mark. Non-audio streams through untouched
- [x] **The audit log is anchored.** Entry count and head hash mirrored to a
      sibling file, so tail truncation — which no hash chain can see — is
      caught. Still evidence, not proof: a test asserts that removing the
      anchor makes the truncated log verify again
- [x] **Translated text carries a disclosure**, on stderr and beside the GUI
      result box, never in the payload

## C3 — What testing against the real binary found

The proxy had been tested against a mock upstream only.

- [x] `--port` was emitted twice — once by `start_server` for the loopback
      port, once by `_append_params` from the operator's flag. Last one won, so
      the binary bound the **public** port unproxied while the proxy waited for
      an upstream that never came. Harmless while both sources agreed
- [x] The proxy stacked marks: it ran the full pipeline over audio the binary
      had already watermarked, ~41 dB SNR for a mark that already verified.
      `complete_marking()` verifies first and fills only gaps
- [x] The first cut of that fix marked *before* watermarking, and the
      watermarker's soundfile round-trip dropped the marker it had just
      written. Layer order restored; a test forces the embed path

## C4 — Every response format, not just the convenient one

- [x] `response_format: "f32"` returns `application/octet-stream`, which is not
      `audio/*`, so raw float32 synthetic audio walked straight out unmarked.
      Classification is now by **endpoint**, not Content-Type: a synthesis path
      returns synthetic audio whatever it labels it
- [x] Raw formats are marked rather than refused — no container, but the
      watermark rides on the samples and needs no header to detect
- [x] Concurrency measured: parity with direct access (the binary serialises on
      its model mutex; the proxy costs ~4%)

## C5 — Whose voice is a preset voice?

- [x] The Art. 50(4) disclosure keyed on "was reference audio supplied", and
      COMPLIANCE.md said outright that a stock voice is not a deepfake. Art.
      3(60) turns on the output *resembling* a person, not on how. Every
      stock-voice run on a real-person backend was an undisclosed deep fake
- [x] `utils/speaker_identity.py`: three values, `unknown` warns rather than
      assuming. GUI control added — the CLI could answer the question and the
      GUI could not

## C6 — What CrispASR's own table corrected

Reading `crispasr_speaker_identity_models.h` corrected four of C5's verdicts,
all of them mine, all from reasoning about a name instead of a model card.

- [x] `kartoffel-orpheus-de-synthetic` classified from the word "synthetic" in
      the filename. Not researched → `unknown`. Guessing synthetic is the error
      that silently *removes* a disclosure
- [x] `crispasr:fastpitch` inherited a sibling project's verdict for different
      weights (German NeMo vs NVIDIA English)
- [x] `crispasr:speecht5` takes its x-vector from the operator, so no
      backend-level verdict can be right. The Python-native backend is
      genuinely different: CMU ARCTIC speaker 7306 baked in as default
- [x] `crispasr:kokoro` was a blanket `synthetic`, wrong for the German HUI
      fine-tune. Resolution is now checkpoint-aware, as CrispASR's is
- [x] Streaming synthesis (`"stream": true`) would have been silently buffered
      by the proxy, turning a streaming endpoint into a non-streaming one.
      Refused with a 502 that names the fix
- [x] `wait_for_upstream` had a 60s limit, so any cold start needing a model
      download refused with "the server did not come up" while the download was
      running — a fail-closed gate firing on a healthy system, which is how a
      safety control gets switched off. Now waits while the process lives, with
      progress output. Found by trying to download a model to test something
      else, which is the only way this was ever going to surface
- [x] **The AudioSeal watermark had never applied.** `embed_watermark()` called
      `generator.get_watermarked_audio()`, which does not exist in the package;
      the AttributeError hit the fallback and the spread-spectrum comb marked
      the file instead. Output was never unmarked, so no Art. 50(2) hole — but
      the layer README and COMPLIANCE.md both called "resists deliberate
      removal" was inert on every install that actually had AudioSeal. The live
      test asserted only that the audio came back watermarked, which the
      fallback satisfied; it now asserts *which* layer did it, and pins the
      call shape. Round-trip verified: confidence 1.0, backend `audioseal`,
      20.8 dB SNR

## Remaining limitations

Structural first, environmental second.

- **25 of 59 TTS backends are unclassified for speaker identity.** They resolve
  to `unknown` and warn once each, which is honest rather than safe: a
  classification nobody researched is not a classification. Reading cards moved
  five of CrispASR's open questions (`parler-tts` → real_person, `dia` →
  synthetic, `bark`/`vibevoice`/`csm` → checked-and-still-unknown), and
  *downgraded* `bark` from an inherited `synthetic`. Do not read an
  unclassified backend as synthetic.
- **The German Kokoro verdict is half-settled.** The upstream card resolves the
  architecture — "This is a base model, not a voice", speaker-neutral over 51
  HUI speakers with a per-speaker cap so none dominates. It cannot resolve the
  voicepack, which is what a listener hears, and one ships as `df_eva` against
  a named HUI narrator. Now a per-voice question rather than a per-model one.
  `kokoro-de-hui-base` is in the local model cache, so this is live.
- **ASR streaming (SSE) cannot be exercised.** Only `qwen3_tts` declares
  `CAP_STREAMING` in the whole CrispASR codebase, so `/v1/audio/transcriptions`
  with `stream=true` always falls back to JSON. The proxy's passthrough relay
  is verified against a controlled upstream with timing assertions instead.
- **The AudioSeal live tests are slow, not skipped.** They were skipped when
  `audioseal` was absent, which is how the broken API call survived. With it
  installed the round-trip runs — and loading the detector costs ~25 s the
  first time in a process, which the marking proxy pays on its first audio
  response and the test suite pays once. Worth knowing before blaming a hang.

Organisational obligations (lawful basis, conformity assessment, disclosure to
your audience, registration) remain the provider's or deployer's and cannot be
discharged by code. COMPLIANCE.md states which are which.
