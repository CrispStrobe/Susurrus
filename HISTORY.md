# Susurrus — completed work

What has shipped, and why it was done that way. Open work lives in
[PLAN.md](PLAN.md); this file is the record, not the backlog.

Kept in reverse chronological order, because the recent rounds are the ones a
new reader needs and the early ones are mostly backend plumbing.

**A note on version numbers.** The last git tag is `v2.11.0`. PLAN.md used
`v2.12.0` and `v2.13` as headers for work that landed in `main` but was never
tagged, so those labels described plans rather than releases. Everything from
the B round onward ships together as **2.12.0**. If you are trying to match a
section here to a release, use the tags, not the old headers.

---

## 2.12.0 — Audit round D (EU AI Act, fourth pass)

Triggered by "is this perfectly in line with the EU AI Act?", which it was not.
Six findings, two of them documentation that overstated coverage. The pattern
worth carrying forward: **the document drifts faster than the code**, and both
of the doc findings were in the direction that made Susurrus look better.

- [x] **D1 — the narrow opt-outs never took effect.** `--no-watermark`,
      `--no-c2pa` and `--no-spoken-disclaimer` each require
      `--accept-marking-responsibility`, and the attestation short-circuited
      `apply_provenance()` before any layer ran. So the documented "skips only
      the cryptographic layer" was unreachable: the combination the CLI *forces*
      you to pass produced completely unmarked audio. The CrispASR routes
      forward the flags to the binary individually and did behave as documented
      — the same backend-dependent split the attestation rule had been
      introduced to remove. The attestation is now the authorisation and the
      flag is the selection; a bare attestation is still a full opt-out.
- [x] **D2 — the unknown-speaker warning never reached a GUI operator.** The
      whole `unknown` policy rests on telling the deployer, and the telling was
      one `logger.warning`. `main.py` routes logging to a `StreamHandler`, which
      a packaged windowed app has no window for, and the log viewer attaches its
      handler lazily — only once the user opens Tools → Logs. Synthesizing
      before opening that dialog left no trace, under a status line reading
      "Marked as AI-generated". Now in the GUI status area, localized.
- [x] **D3 — two implementations of "is this cloning?"** The CLI preflight
      asked `os.path.isfile(--voice)` while the real gate also covered
      `--voice-dir` resolution and voice-bank selection. Nothing unmarked
      escaped, but the operator paid a full synthesis for a refusal knowable in
      milliseconds. `would_clone()` is now module-level and both callers use it.
- [x] **D4 — Art. 50(5) was absent from COMPLIANCE.md.** Not argued and
      dismissed — missing, which is worse, because a map that omits a provision
      reads as a map on which it does not appear. Timing is met structurally
      (the disclosure is the first thing in the file). Accessibility is a real
      gap: an audible-only disclosure reaches nobody who cannot hear it, and an
      audio file has nowhere to put a caption. `--disclosure-text` now prints
      the exact sentence for the caption or on-page notice.
- [x] **D5 — the Art. 50(4) artistic exception had no correctly-shaped exit.**
      The mitigation for evidently artistic or satirical work is about the
      *manner* of disclosure, not about marking. The only way to drop the spoken
      prefix also dropped Art. 50(2) marking, so the exit from an over-strict
      50(4) overshot into a 50(2) failure. Fixed by D1; documented.
- [x] **D6 — the marking dependencies were optional.** `soundfile` and `numpy`
      are what the in-sample watermark is built from and sat in `[tts]`. A bare
      install had metadata only — and *can* still drive the CrispASR binary, so
      the one configuration able to emit strippable-marking-only audio was the
      one that had never opted in. Now base dependencies. C2PA stays an extra
      deliberately: a compiled wheel that fails to install marks nothing at all.

Documentation corrections in the same round, recorded because they are the
recurring failure mode:

- [x] "40 classified, 19 unknown" read as 40 answered. 40 was the count of
      *table entries*, 24 of which record `unknown` as the answer. The real
      ratio is **16 answered, 43 of 59 open**.
- [x] `df_victoria` / `dm_martin` were documented `unknown` while the code
      classified them `synthetic` on the provider's card.
- [x] "A test enforces the asymmetry: no per-voice entry may be `synthetic`" —
      that test had been deliberately replaced by one requiring *evidence*,
      which is what separates the kikiri card from a guess.
- [x] "None of the backends is mixed … so that table is empty" outlived the
      change that filled it, in COMPLIANCE.md *and* in the comment directly
      above the table. `crispasr:kokoro` is exactly a mixed backend.
- [x] Bark was still listed `synthetic` in the disclosure table after the code
      downgraded it to `unknown`.
- [x] COMPLIANCE.md gained: model licences and upstream terms, personality
      rights in the cloned voice, data protection and the no-telemetry
      position, a provider-status section built on Art. 3(10)'s "commercial
      activity" test, and a header pinning version, review date and the fact
      that nothing in it is a certification.
- [x] `tests/unit/test_provenance_audit_v4.py` — 26 tests. The doc-sync group
      parses the counts out of COMPLIANCE.md, recomputes them from the registry,
      and fails the build if they diverge. Prose cannot be pinned; the numbers
      can, and those are what gets quoted.

## 2.12.0 — Audits C1–C6

Six rounds, each triggered by "is it done?" rather than by a plan. Every one
found real defects and three found defects in code the previous round had just
written. The pattern is the finding: reading the source of truth beats
reasoning about it, and running the thing beats reading it.

- [x] **C1 — marking that does not depend on a guess.** The declarative floor
      on the CrispASR routes was applied only when the detectors came up empty,
      making an Art. 50(2) guarantee conditional on the detector being right. It
      was not: over 1500 clips of unwatermarked speech, tones and noise it read
      ~12% as watermarked, and one false positive suppressed the floor and
      shipped unmarked audio under a printed "Marked as AI-generated". Floor is
      now unconditional. Detector fixed separately — weighted correlation,
      threshold 0.65 → 0.78, FP 12% → 1.2%, TP 93% → 98% — because "we no longer
      depend on it" is not a reason to leave it wrong. `--s2s` gained its
      consent gate and marking check. `--about-ai` added for CLI-only installs.
- [x] **C2 — the three routes documented as uncovered.** `utils/marking_proxy.py`
      puts Susurrus back in server mode's response path: binds the requested
      port, starts the binary on loopback, marks audio responses, refuses what
      it cannot mark, streams everything else untouched. The audit log gained a
      sibling anchor, because no hash chain can detect truncation of its own
      tail. Translated text carries a disclosure on stderr, never in the payload.
- [x] **C3 — what testing against the real binary found.** `--port` was emitted
      twice, so the binary bound the *public* port unproxied while the proxy
      waited for an upstream that never came — harmless only while both sources
      agreed. The proxy stacked marks over audio the binary had already
      watermarked (~41 dB SNR for a mark that already verified);
      `complete_marking()` verifies first and fills only gaps. The first fix for
      that marked *before* watermarking, and the watermarker's soundfile
      round-trip dropped the marker it had just written.
- [x] **C4 — every response format.** `response_format: "f32"` returns
      `application/octet-stream`, so raw float32 synthetic audio walked out
      unmarked. Classification is now by **endpoint**, not Content-Type.
      Concurrency measured at ~4% overhead.
- [x] **C5 — whose voice is a preset voice?** The disclosure keyed on "was
      reference audio supplied", and COMPLIANCE.md said outright that a stock
      voice is not a deepfake. Art. 3(60) turns on the output *resembling* a
      person, not on how. `utils/speaker_identity.py` added, with `unknown` as a
      question rather than an assumption.
- [x] **C6 — what CrispASR's own table corrected.** Four of C5's verdicts were
      wrong, all from reasoning about names instead of reading model cards.
      Also: streaming synthesis would have been silently buffered by the proxy
      (now a 502 naming the fix); `wait_for_upstream`'s 60 s limit made every
      cold start fail while a model download was still running — a fail-closed
      gate firing on a healthy system, which is how a safety control gets
      switched off. **AudioSeal had never once applied**:
      `generator.get_watermarked_audio()` does not exist in the package, so
      every install that actually had AudioSeal fell through to the
      spread-spectrum comb. Output was never unmarked, but the layer both docs
      called "resists deliberate removal" was inert. The live test asserted only
      that audio came back watermarked, which the fallback satisfied; it now
      asserts *which* layer did it.

## 2.12.0 — B round: closing the v2.11.0 known limitations

- [x] **B4 — Art. 12 record-keeping.** `utils/audit_log.py`, append-only JSONL,
      SHA-256 hash-chained, logging identification as well as enrollment
      (Art. 12 covers use, not just setup). Never records the embedding or
      audio; a test greps the written file for biometric field names so a future
      edit cannot quietly add one. `--audit-log` verb + GUI dialog. 19 tests.
- [x] **B1 — spoken disclosure on Python-native TTS.** Synthesized with the
      *same* backend so sample rate and channel count align by construction,
      with a recursion guard and a refusal to concatenate mismatched formats. A
      test asserts the disclosure *precedes* the content — a trailing disclosure
      is not a disclosure. 21 tests.
- [x] **B2 — neural watermark on Python-native TTS.** AudioSeal, lazily
      imported, failed-load result cached. Applied before C2PA signing, with a
      test on call order since watermarking mutates samples C2PA then hashes.
      `--detect-watermark` exits 2 for "could not check" so it is never read as
      "not AI-generated".
- [x] **B3 — complete GUI internationalization.** 259 keys across 15 GUI files,
      split into `utils/translations/{en,de}.py`. Beyond key parity the tests
      assert format placeholders match across locales, that compliance strings
      are genuinely translated rather than copied, and — via an AST scan of the
      whole `gui/` tree — that no hardcoded user-visible string remains. That
      last test found the 11 multi-line strings literal matching missed.
- [x] **B5 — fallout found while closing B1–B4.** `--detect-watermark`
      delegated to the binary whenever one was *present* and returned its exit
      code even when it crashed; a stale local build made the verb exit 250 with
      a dyld error instead of answering. The detector report's single `detector`
      field covered two independent checks; split into `neural_detector` +
      `ai_marker`.

## 2.11.0 — Compliance remediation (A1–A7)

Findings from the 2026-08-01 audit. C2 and F1 below had been marked DONE but
shipped disconnected code — the hooks existed and nothing called them.

- [x] **A1 — `sign_output()` had zero production callers.** Every Python-native
      backend emitted synthetic audio with no machine-readable marking. Renamed
      to `apply_provenance()` (it applies more than one layer) and wired into
      both CLI and GUI branches. The `hasattr(...)` test that had passed while
      the feature did nothing was replaced by one asserting the call happens.
- [x] **A2 — voice cloning ungated on the Python path.** Chatterbox cloned from
      an arbitrary WAV with no consent gate. `require_clone_consent()` added at
      the *top* of `synthesize()`, before the torch import, so a refusal costs
      no model download.
- [x] **A3 — the GUI provenance controls were inert.** The watermark checkbox
      and both cert pickers were no-ops because `TTSThread` forwarded three
      unrelated keys. One `provenance` dict now goes to both branches.
- [x] **A4 — the wizard asserted the attestation for the user.** Consent was a
      passive label and `i_have_rights` was set unconditionally. Now a required
      checkbox gating the button.
- [x] **A5 — marking that survives a missing library.** `utils/ai_marking.py`
      embeds a RIFF `LIST/INFO` chunk (WAV) readable by any parser. Verified
      with `ffprobe`; audio payload byte-identical.
- [x] **A6 — the speaker-DB biometric path surfaced.** Warning, help text
      naming GDPR Art. 9, and documentation of the Annex III(1)(a) question.
- [x] **A7 — the missing Art. 13 / Art. 4 documentation.** README intended
      purpose and limitations; COMPLIANCE.md created.

## 2.9.1–2.10.0 — Provenance flags and GUI features

- [x] C1 — CLI provenance flags (`--i-have-rights`, `--no-watermark`,
      `--detect-watermark`, `--c2pa-cert`/`--key`, …), all wired through
      `_build_crispasr_kwargs`. CrispASR defaults left untouched: marking on.
- [x] C2 — `utils/c2pa_signing.py`, `--verify-c2pa`. *(Hook defined but never
      called until A1 — see above.)*
- [x] C3 — "Detect Watermark" button in the transcription tab.
- [x] F1 — GUI provenance controls. *(Rendered but inert until A3.)*
- [x] F2 — live mic streaming in the GUI. F3 — voice clone wizard.
      F4 — i18n foundation. F5 — server toggle. F6 — `utils/semantic_search.py`
      with substring fallback.

## 2.7.0 — Usability and testing

- [x] R2 — `tests/integration/test_cli_e2e.py`, 9 end-to-end CLI tests, all
      auto-skipping without a binary or test audio.
- [x] R3 — `gui/widgets/segment_list_widget.py` replacing the plain text output:
      speaker chips, confidence badges, inline editing, keyboard navigation.
- [x] R4 — CI integration job (advisory, `continue-on-error`).

## 2.6.0 — GUI wiring and polish (W1–W9)

- [x] Progress parser wired into `TranscriptionThread` with a determinate
      progress bar and an indeterminate fallback; segment model wired through
      output, export and history; batch panel and waveform widget placed; dev
      tool versions pinned to CI; settings persisted; multi-file drag-and-drop;
      keyboard shortcuts; server toggle; About dialog and README refresh.

## 2.4.0 — GUI feature parity

- [x] History service and browser, export formats (SRT/VTT/JSON/CSV/TXT), batch
      queue, deterministic progress, inline segment editing and speaker
      renaming, waveform display, live streaming, light/dark themes with
      confidence colouring, voice clone wizard, German i18n, log viewer.

## 2.3.0 — CrispASR 0.8.7 sync

- [x] 8 ASR backends (`ark-asr`, `higgs-stt`, `moss-transcribe`, `gemma4-e4b`,
      `parakeet-ctc-ja`, `reazonspeech`, `canary-ctc`, `qwen3-ja-anime`),
      3 TTS backends (`tada`, `dots-tts`, `bananamind-tts`), `m2m100-f16`
      translation, plus CLI flags, FFI, companion resolution and tests.
