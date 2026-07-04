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

## 4. New CLI Flags → PARAM_MAP (crispasr_backend.py)

- [x] `--align-only` (bool) — standalone alignment mode
- [x] `--text-file` (str) — text/SRT input for align-only
- [x] `--align-output` (str) — alignment output path
- [x] `--align-format` (str) — alignment format (srt/json/plain)
- [x] `--make-ref` (bool) — create TADA voice reference GGUF
- [x] `--make-ref-output` (str) — voice reference output path
- [x] `--make-ref-encoder` (str) — TADA encoder GGUF path
- [x] `--make-ref-aligner` (str) — TADA aligner GGUF path
- [x] `--diarize-speakers` (bool) — shorthand for diarize + embedder auto
- [x] `--speaker-db-consent` (bool) — GDPR consent for persistent speaker DB
- [x] `--prefix-text` (str) — LLM initial prompt (granite keyword biasing)
- [x] `-odjson` (bool) — diarized JSON output format

## 5. New FFI Methods (crispasr_ffi_backend.py)

- [x] `set_top_k(k)` — top-K sampling
- [x] `set_do_sample(bool)` — multinomial sampling toggle
- [x] `set_tts_num_candidates(n)` — TTS candidate count (TADA)
- [x] `set_tts_noise_temp(float)` — diffusion noise temperature

## 6. New CLI Arguments (cli.py)

- [x] `--mode align` with `--text-file`, `--align-output`, `--align-format`
- [x] `--diarize-speakers`, `--speaker-db-consent`
- [x] `--prefix-text`

## 7. Companion Models (crispasr_utils.py + config.py)

- [x] `dots-tts` → CAM++ speaker encoder (`dots-tts-soar-spk`)
- [x] `tada` → encoder + aligner (`tada-encoder`, `tada-aligner`)

## 8. Server Response Format

- [x] `diarized_json` in PARAM_MAP output format flags (`-odjson`)

## 9. Bug Fix

- [x] `probe_backends()` — handle `{"backends": [...]}` dict-wrapper format

## 10. Tests — 117 total, all pass

### Unit tests (103 tests, 7 skipped)
- [x] PARAM_MAP entries for all new flags (TestCrispASR087Sync)
- [x] Config registry entries for all new backends (TestCrispASR087Registry)
- [x] Companion model resolution for dots-tts, tada (TestFFICompanionModels)
- [x] FFI session setter kwargs for top_k, do_sample, tts_num_candidates, tts_noise_temp
- [x] TTS backend map entries for tada, dots-tts, bananamind-tts

### Live integration tests (12 tests, 1 skipped)
- [x] Binary version check
- [x] `--list-backends-json` probe (handles dict-wrapper format)
- [x] Backend presence check (whisper, parakeet)
- [x] New 0.8.7 backends advisory check
- [x] `--dry-run-resolve` for whisper and parakeet
- [x] Transcribe with whisper tiny (CPU, 8GB safe)
- [x] `--align-only` flag acceptance
- [x] `--diarize-speakers` flag acceptance
- [x] `--prefix-text` flag acceptance
- [x] `--speaker-db-consent` flag acceptance
