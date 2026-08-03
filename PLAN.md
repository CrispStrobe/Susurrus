# Susurrus — open work

Completed work has moved to [HISTORY.md](HISTORY.md). This file is the backlog
and nothing else: if an item here is finished, move it there rather than
ticking it in place, or this file becomes a changelog again.

**State as of 2026-08-03:** `main` is clean at 2.12.0, 580 tests pass, 46 skip.
The last git tag is `v2.11.0`, so 2.12.0 is unreleased and carries the B, C and
D audit rounds together.

---

## How to work on this repo

Three things the last four audit rounds established, all of them the hard way.

1. **Read the source of truth, don't reason about it.** Every wrong
   speaker-identity verdict so far came from inferring provenance from a name
   instead of reading a model card. Every documentation finding came from
   trusting the document over the code.
2. **A test that passes while the feature does nothing is worse than no test.**
   `hasattr(TTSBackend, "sign_output")` passed for a whole release while the
   hook had zero callers. AudioSeal's live test asserted "the audio came back
   watermarked", which the *fallback* satisfied, so a dead code path survived
   two audits. Assert which thing did the work.
3. **Run it.** `--port` emitted twice, the 60-second upstream timeout and the
   f32 response format were all invisible to code review and obvious within
   minutes of using the thing.

Before claiming a compliance obligation is met, check that the control is
reachable from **both** the CLI and the GUI. Four separate findings have been
"implemented, but the GUI path doesn't have it".

---

## P0 — Speaker-identity research (Art. 50(4))

**43 of 59 exposed TTS backends resolve to `unknown`.** This is the largest
substantive compliance gap in the project. `unknown` means no audible
disclosure is added; if one of those voices is an identifiable person, the
output is an undisclosed deep fake under Art. 3(60) and the Art. 50(4) duty is
live and unmet. The software warns, which makes the gap honest, not closed.

- [ ] Work through the 19 backends absent from `BACKEND_SPEAKER_IDENTITY`
      entirely: `dots-tts`, `irodori-tts`, `kartoffelbox-turbo`, `kugelaudio`,
      `lahgtna-chatterbox`, `lfm2-audio`, `mini-omni2`, `miotts`, `moss-tts`,
      `moss-tts-local`, `omnivoice`, `omnivoice-singing`, `pocket-tts`, `tada`,
      `tada-1b`, `tada-3b-ml`, `tada-tts-1b`, `voxcpm2-tts`, `voxtral-tts`
- [ ] Re-check the 24 recorded as `unknown`. Several were checked against cards
      that have since been updated; a dead end is worth revisiting once
- [ ] **Read the provider's card. Do not infer from the name.** Record the
      evidence string alongside the verdict, as `VOICE_SPEAKER_IDENTITY` already
      requires — a test enforces it there and should be extended to the backend
      table
- [ ] Guessing `synthetic` is the error that silently *removes* a disclosure.
      When the card is silent, the answer is `unknown`, not "probably fine"
- [ ] Update the counts in COMPLIANCE.md — `test_provenance_audit_v4.py` will
      fail the build until you do, which is the point

## P0 — Release 2.12.0

Unreleased work has accumulated across three audit rounds.

- [ ] Verify the release workflow still builds (`.github/workflows/`,
      `tests/unit/test_release_workflow.py`)
- [ ] Confirm `soundfile` + `numpy` as base dependencies do not break the
      PyInstaller bundle — they are new to the base install as of D6
- [ ] Tag `v2.12.0`, matching `pyproject.toml`, `setup.py`, `__init__.py` and
      the version COMPLIANCE.md pins itself to
- [ ] Update the COMPLIANCE.md "Last reviewed" date if anything compliance-
      relevant changes before the tag

## P1 — Run the GUI and fix what breaks (was R1)

Never completed, and the only item that survived from the v2.7.0 plan. It needs
a display and PyQt6, so it cannot be done in a headless agent environment —
this is a task for someone at a desk.

- [ ] `python main.py` — window renders without errors
- [ ] Select audio → waveform loads; transcribe → progress bar advances, output
      appears, segments stored
- [ ] Save → format picker produces a valid file in each of TXT/SRT/VTT/JSON/CSV
- [ ] History tab: entries appear, search works, load works
- [ ] Theme toggle renders both themes; drag-drop populates input;
      View → Show Logs opens
- [ ] **Synthesize with an `unknown` preset voice and confirm the D2 warning is
      visible in the status area**, not only in the log dialog. This is new in
      2.12.0 and has only been verified headlessly with a stubbed PyQt6
- [ ] Confirm the Art. 50(5) disclosure text is reachable — currently CLI-only
      via `--disclosure-text`; the GUI has no equivalent (see P2)

## P1 — C2PA as a base dependency

D6 promoted `soundfile` and `numpy` out of the extras. C2PA was deliberately
left behind: it is a compiled wheel and a hard dependency that fails to install
produces a tool that marks nothing at all. That was a judgement made without
data.

- [ ] Check `c2pa-python` wheel coverage across the platforms and Python
      versions Susurrus supports (`requires-python = ">=3.9"`)
- [ ] If coverage is broad, promote it to a base dependency and update the
      COMPLIANCE.md paragraph that explains why it is not
- [ ] If it is not, keep the current split and record the platforms that lack
      wheels, so this is not re-litigated from scratch

## P2 — Art. 50(5) accessibility

Timing is met structurally: the spoken disclosure is the first thing in the
file. Accessibility is not — an audible-only disclosure reaches nobody who
cannot hear it, and an audio file has nowhere to put a caption.
`--disclosure-text` hands the deployer the exact sentence; placement remains
theirs.

- [ ] GUI equivalent: show the disclosure text beside the output, copyable,
      whenever a run produced an audible disclosure
- [ ] Consider an opt-in sidecar (`output.wav` → `output.disclosure.txt`) so a
      publishing pipeline can pick it up automatically. **Opt-in**: writing an
      extra file next to every synthesis by default would surprise people
- [ ] Consider emitting a WebVTT cue rather than bare text, so it drops into a
      caption track without reformatting

## P2 — Model licences

COMPLIANCE.md now has a section saying to read each backend's licence and
explaining why there is no table. Building the table is real work and was
deliberately not attempted: inventing 59 licence verdicts would repeat exactly
the mistake the speaker-identity section documents.

- [ ] If you build it, read each licence. Same rule as speaker identity: record
      the evidence, and `unknown` when the terms are unclear
- [ ] Surface it at backend-selection time in the GUI, and in `--list-backends`
- [ ] Flag non-commercial and research-only checkpoints explicitly — those are
      the ones where a user is most likely to be caught out

## P2 — Delete the filename-matching fallback

`MODEL_RULES` matches on checkpoint filenames, against this project's own
"classify by provenance, not by filename" rule. It is used anyway because the
alternative is no answer, and because the failure is safe: a rename falls back
to `unknown` and warns.

- [ ] Track how widely CrispASR's GGUF `speaker_identity` stamp has spread
- [ ] Delete `MODEL_RULES` once the stamps cover the shipped checkpoints. It is
      the weakest link in the resolution chain and should be the first to go

## P3 — Regulatory watch

Neither of these can be settled from inside the repo, and both change what
"compliant" means.

- [ ] **The Commission's code of practice on marking and labelling AI-generated
      content.** Adherence is the presumption-of-conformity route under Art. 50.
      Susurrus's declarative RIFF/ID3 marker is a supplement to C2PA, not a
      standard — if the code of practice settles on a format, align to it
- [ ] **Applicability dates.** The table in COMPLIANCE.md is accurate as
      written and Art. 50 became applicable on 2 August 2026, but amendments
      have been under discussion. Re-check before relying on it

## P3 — Known warts

Recorded so they are not rediscovered as bugs.

- [ ] **`--detect-watermark` has two exit-code semantics.** The binary path
      passes the binary's own code through; the Python fallback uses 0/1/2.
      Unifying them means parsing the binary's stdout for its verdict, which
      breaks on any output-format change — documented in `--help` instead.
      Scripts that branch on the exit code should pin one path
- [ ] **ASR streaming (SSE) cannot be exercised.** Only `qwen3_tts` declares
      `CAP_STREAMING` in the whole CrispASR codebase, so
      `/v1/audio/transcriptions` with `stream=true` always falls back to JSON.
      The proxy's passthrough relay is verified against a controlled upstream
      with timing assertions instead
- [ ] **AudioSeal live tests are slow, not skipped.** Loading the detector costs
      ~25 s the first time in a process — paid once by the test suite and by the
      marking proxy on its first audio response. Worth knowing before blaming a
      hang. They were *skipped* when `audioseal` was absent, which is exactly
      how the broken API call survived two audits; do not re-skip them
- [ ] **The spoken disclosure for a real-person preset is spoken in that
      voice**, because a fixed-speaker model offers no second voice. For a
      *cloned* voice the reference is withheld so the disclaimer is never
      delivered by the impersonated speaker. No fix is available for the preset
      case short of shipping a separate disclosure voice

---

## Not code, and not deferrable

These fall on whoever publishes or deploys Susurrus and cannot be closed by a
commit. Recorded here because "the software does not do this" is easy to read
as "this does not need doing" — COMPLIANCE.md has the full treatment.

- Establishing whether you are a **provider** at all. Art. 3(10) turns on
  supply "in the course of a commercial activity"; publishing MIT source with
  no monetisation is a different act from shipping a product. Decide, and write
  the reasoning down
- Reading the licence of any model you deploy commercially
- A lawful basis for processing voice data, if you use the speaker database
- Telling **your** audience that content is AI-generated. Marking travels with
  the file; the duty to inform people does not
- Not overclaiming. COMPLIANCE.md is a representation people rely on, and a
  warranty disclaimer in a licence file is not obviously a defence for it.
  "Implements these measures", never "compliant" or "certified"
