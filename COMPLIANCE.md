# EU AI Act Compliance

This document maps Susurrus onto the obligations of Regulation (EU) 2024/1689
(the AI Act): what the software does for you, and what remains yours to do.

> **Not legal advice.** Compliance is a property of a *deployment*, not of a
> repository. This document describes the technical measures shipped in
> Susurrus so you can assess your own obligations.

## Does the AI Act apply to this project?

Yes, to the parts described below.

Susurrus is MIT-licensed, and Art. 2(12) exempts AI systems released under
free and open-source licences from most of the Regulation — **but not** those
falling under Art. 5 (prohibited practices) or **Art. 50** (transparency).
A text-to-speech suite with voice cloning is an Art. 50 system, so the
transparency obligations apply in full and the FOSS carve-out does not help.

## Applicability dates

| Provision | Applies from |
| --- | --- |
| Art. 5 prohibited practices, Art. 4 AI literacy | 2 February 2025 |
| GPAI obligations, governance, penalties | 2 August 2025 |
| Art. 50 transparency, Annex III high-risk | 2 August 2026 |
| Annex I product-embedded high-risk | 2 August 2027 |

Amendments to these dates have been under discussion. Verify the current
position before relying on the table.

## Who is who

- **Provider** — whoever places an AI system on the market or puts it into
  service under their own name. If you fork Susurrus, rebrand it, or ship it
  inside your product, that is you.
- **Deployer** — whoever uses the system under their own authority. If you run
  Susurrus to transcribe or synthesize, that is you.

Running Susurrus locally for purely personal, non-professional activity is
outside the Regulation's scope (Art. 2(10)) — but the Art. 50 marking still
applies to the *provider*, which is why marking is on by default and cannot be
disabled by accident.

## Art. 50(2) — marking synthetic audio

> Providers of AI systems generating synthetic audio, image, video or text
> shall ensure the outputs are marked in a machine-readable format and
> detectable as artificially generated or manipulated.

Susurrus applies marking on every TTS path that writes a WAV or an MP3.
Three kinds of layer, one of which has two tiers:

| Layer | Mechanism | Availability | Survives re-encoding |
| --- | --- | --- | --- |
| In-sample watermark (tier 1) | AudioSeal, learned | CrispASR binary, or `pip install 'susurrus[watermark]'` | Yes, incl. deliberate removal |
| In-sample watermark (tier 2) | Spread-spectrum comb — `utils/spread_spectrum.py` | Always — numpy only | Yes, for ordinary transcoding |
| Cryptographic | C2PA Content Credentials | CrispASR binary, or `pip install 'susurrus[c2pa]'` — included in `[tts]` | No (manifest is stripped) |
| Declarative | RIFF `LIST/INFO` (WAV) or ID3v2.4 (MP3) — `utils/ai_marking.py` | Always — no dependencies | No |

The layers are applied in a fixed order, and the order matters: the spoken
disclosure and the watermark change the samples, the declarative marker adds
metadata, and C2PA hashes the finished file — so C2PA must run last or its
manifest describes audio that no longer exists.

Art. 50(2) requires marking that is "effective, interoperable, robust and
reliable as far as technically feasible", and it has no "unless a dependency
is missing" clause. So neither layer that survives re-encoding is allowed to
be optional: when AudioSeal is absent, the spread-spectrum comb is embedded
instead, needing nothing beyond numpy. A default install therefore ships two
marks in the samples and one in the metadata, not metadata alone.

The declarative marker uses each container's own metadata format: any ordinary
parser reads it, parsers that don't care skip it, and the audio samples are
untouched by that layer.

Watermarking preserves the file's channel count and sample format. A marking
step that quietly downmixed stereo to mono, or requantised 24-bit to 16-bit,
would be damaging the audio it is supposed to be annotating.

**Containers other than WAV and MP3** (FLAC, M4A, Opus …) get C2PA if the
library is installed, but have no dependency-free fallback. Susurrus says so
rather than reporting success: the CLI prints a warning naming the container,
and the GUI shows the same. Prefer `.wav` or `.mp3` if marking matters to you.

**Marking is verified, not assumed.** On the CrispASR routes the binary
applies the layers itself, and its support depends on build options, engine
capability and version. Susurrus reads the finished file back — declarative
marker, C2PA manifest, and AudioSeal detection where available — and reports
what is actually there. If nothing is detectable it applies the declarative
marker as a floor. Earlier versions reported marking straight from the command
line flags, which meant a build without C2PA support still produced a
confident "Marked as AI-generated" over unmarked audio.

C2PA signing needs an X.509 credential. Pass `--c2pa-cert` / `--c2pa-key` to
use your own; otherwise Susurrus generates a local CA + end-entity chain once
and caches it under `~/.local/share/susurrus/c2pa/`. That identity is
self-issued: it makes the output tamper-evident, and it asserts nothing about
who you are. For a credential that carries identity, supply a CA-issued
certificate. Signing is offline by default; set `SUSURRUS_C2PA_TSA` to an
RFC 3161 timestamp authority URL to add a trusted timestamp.

Verify marking on a file — this reports **both** layers and exits 0 if the
file is marked by either:

```bash
susurrus --verify-c2pa output.wav
```

```json
{
  "c2pa": { "available": false },
  "ai_marker": {
    "ISFT": "Susurrus",
    "ICMT": "AI-GENERATED AUDIO. Synthesized by an AI system. Marked per EU AI Act Art. 50(2).",
    "ITCH": "EU-AI-Act-Art50-2",
    "IENG": "piper"
  },
  "marked_as_ai_generated": true
}
```

Third-party tools read the declarative marker without knowing anything about
Susurrus, which is what makes it machine-readable in the sense Art. 50(2)
means:

```console
$ ffprobe -v error -show_entries format_tags -of default output.wav
TAG:encoder=Susurrus
TAG:comment=AI-GENERATED AUDIO. Synthesized by an AI system. Marked per EU AI Act Art. 50(2).
TAG:encoded_by=EU-AI-Act-Art50-2
```

The declarative marker is metadata: it is trivially strippable and is not
robust to re-encoding. It establishes that Susurrus marked the output, not
that an arbitrary file you received is authentic. For tamper-evidence use
C2PA (`pip install 'susurrus[c2pa]'`, and included in the `tts` extra). For
survival through re-encoding the spread-spectrum watermark is always applied;
`pip install 'susurrus[watermark]'` upgrades that to AudioSeal, which also
resists deliberate removal.

Which layers your install can actually apply is reported per synthesis, and
`susurrus --verify-c2pa FILE` shows what a given file carries. Do not assume
a layer is active because it is documented — check.

**Opting out.** `--accept-marking-responsibility` produces completely unmarked
audio. `--no-c2pa` skips only the cryptographic layer, `--no-watermark` only
the neural one, `--no-spoken-disclaimer` only the audible prefix.

None of the three narrower flags takes effect on its own: each one requires
`--accept-marking-responsibility` alongside it, and Susurrus exits 2 if you
pass one without it. This matches what the CrispASR binary already enforced,
and replaces an inconsistency where the same flag meant different things
depending on which backend you chose.

Be precise about what the attestation does. Art. 50(2) binds the *provider* of
the system that generates the content, and no command-line flag can move a
statutory obligation from one party to another. What the flag records is that
you are shipping output this software did not mark, and are therefore acting
as the provider of that output — with the marking and disclosure duties that
follow. It is an attestation about your role, not a waiver.

## Art. 50(4) — deepfake disclosure

> Deployers of an AI system that generates or manipulates image, audio or
> video content constituting a deep fake shall disclose that the content has
> been artificially generated or manipulated.

Voice cloning is gated on an explicit rights attestation. Cloning is refused —
before any model is loaded — unless you pass `--i-have-rights` (CLI) or tick
the consent box (GUI). The attestation is yours to make: the application will
never set it on your behalf, including from the Voice Clone Wizard.

The gate is in Susurrus, on every route: the Python-native backends, both
CrispASR routes, the in-process FFI route, and speech-to-speech. The CrispASR
binary enforces `--i-have-rights` too, but relying on that alone put the only
check for those routes outside this codebase, where a version or build without
it would clone silently. A path-like `--voice` is treated as reference audio;
a preset voice name is not, so stock voices need no attestation.

What the attestation means: *this is my own voice, or the speaker consented to
having their voice cloned.*

Marking (above) satisfies the machine-readable half of disclosure. On top of
that, **every** backend prepends an audible spoken disclaimer to cloned audio —
CrispASR does it in-binary, and the Python-native backends synthesize the
phrase with the same model and concatenate it (`utils/spoken_disclosure.py`).
The phrase is localized, so a German user gets a German disclosure. Suppress it
with `--no-spoken-disclaimer` (which requires the attestation above);
machine-readable marking still applies.

The disclosure is spoken in the backend's **own** voice, never in the cloned
one. While it is being synthesized the cloning reference is withheld from the
backend, so a disclaimer cannot be delivered by the impersonated speaker —
which would produce exactly the confusion the disclosure exists to prevent.

The disclosure is added only when cloning from reference audio. Synthesis in a
stock voice is not a deepfake, so Art. 50(4) is not engaged — though Art. 50(2)
marking still applies to it, and does.

Disclosure to the people who see or hear the output remains a deployer
obligation that no library can discharge for you.

## Biometrics — speaker enrollment and identification

`--speaker-db`, `--enroll-speaker`, `--expect-speakers` and `--titanet-model`
store voice embeddings linked to named people and match new audio against them.

This has two consequences:

1. **GDPR Art. 9.** Voice embeddings used to identify a person are biometric
   data — special-category data. You need a lawful basis; consent is the usual
   one, and it must be freely given, specific and informed.
2. **Possible Annex III(1)(a) high-risk classification.** Remote biometric
   identification systems are high-risk under the AI Act. Whether your use
   qualifies depends on how you deploy it — one-to-one verification with the
   subject's active participation is treated differently from identifying
   people at a distance without their involvement.

Susurrus warns when the speaker database is used without `--speaker-db-consent`.
The flag is an attestation you make, not a technical control — it does not
create a lawful basis, it records that you believe you have one.

### Art. 12 record-keeping

Every enrollment and identification is written to an append-only log at
`~/.local/share/susurrus/audit/biometric.jsonl`, recording the UTC timestamp,
event type, speaker, database, whether consent was attested, and the embedding
model. Identification is logged as well as enrollment, because Art. 12 covers
use of the system and not only its setup.

The entry is written **after** the run completes, not when the flags are
parsed. A record derived from the command line alone documents an intention
rather than an event — it would assert that people were identified even when
the backend failed to start or the audio was unreadable. An audit trail that
overstates what happened is worse than a sparse one.

Entries are SHA-256 hash-chained, so modification, deletion, reordering and
truncation are all detectable:

```bash
susurrus --audit-log        # print the log and verify the chain; exit 1 if broken
```

Also available from the GUI under Tools → Biometric Audit Log.

The log deliberately never contains the voice embedding or any audio — a
record-keeping mechanism must not become a second copy of the special-category
data it documents. The hash chain is tamper-*evidence*, not tamper-proofing:
anyone who can write the file can rebuild it. If your obligations call for
stronger guarantees, ship the log to append-only storage.

If your deployment lands in Annex III, the remaining high-risk obligations
(risk management, data governance, human oversight, conformity assessment,
registration) fall on you as provider or deployer. Susurrus does not implement
them, and enabling these flags is not a conformity assessment.

## Art. 5 — prohibited practices

Susurrus ships no emotion-recognition capability, so the Art. 5(1)(f)
prohibition on inferring emotions in the workplace or in education is not
engaged by the software as distributed. Do not bolt one on for those contexts.

## Synthetic text — translation output

Art. 50(2) names synthetic *text* alongside audio, image and video. Susurrus
translates text with machine-translation models, and does **not** mark
translation output.

The reasoning: Art. 50(2) exempts systems performing "an assistive function
for standard editing" or not substantially altering the input. Translation
transforms text a user supplied rather than generating new content, and the
Art. 50(4) text-disclosure duty is scoped to text "published with the purpose
of informing the public on matters of public interest" — a judgement about
your publication, not about the tool.

This is a grey area, and it is stated here rather than left silent because a
compliance document that only lists the settled parts is not much use. If you
publish machine-translated text to inform the public, the disclosure duty is
yours and Susurrus does nothing toward it.

Transcription is not covered: it is a transformation of a real recording, not
synthetic content. But see the AI-literacy note below on what a transcript is
and is not.

## Art. 4 — AI literacy

Providers and deployers must ensure a sufficient level of AI literacy among
staff operating these systems. The GUI carries this as **Help → About AI in
Susurrus**: what the system is, its intended purpose, its known failure modes,
and what it is not validated for. It is localized along with the rest of the
interface. Practically, whoever runs Susurrus should understand:

- Transcription output is a **model prediction, not a record**. It contains
  errors, and error rates vary sharply by accent, audio quality, background
  noise, domain vocabulary and language. Non-native accents and
  under-resourced languages typically fare worse.
- Speaker diarization guesses speaker boundaries and counts. Overlapping
  speech, similar voices and short turns are common failure modes.
- Machine translation loses nuance and can invert meaning, especially with
  negation, idiom and ambiguous pronoun reference.
- Synthesized speech is not evidence of anything a real person said.

## Intended purpose and limitations

**Intended purpose.** A local-first desktop and command-line tool for
transcribing audio, synthesizing speech, translating text, and separating
speakers in recordings — for individuals and teams processing their own or
consented material.

**Not intended for**, and not validated for, any use where an error carries
legal or safety consequences without human review: evidentiary transcripts,
medical documentation, employment or education decisions, credit or benefit
eligibility, law enforcement, or migration and border control. Several of
these are Annex III high-risk areas with obligations this project does not
implement.

**Human oversight.** All output is intended to be reviewed by a person before
it is relied on. The GUI supports inline segment editing and speaker renaming
for exactly this reason.

## What Susurrus does not do for you

- Register anything with any authority.
- Perform a conformity assessment or produce technical documentation
  under Art. 11 / Annex IV.
- Establish a lawful basis for processing voice data.
- Disclose to your audience that content is AI-generated.
- Assess whether *your* deployment is high-risk.
- Guarantee the audit log's integrity against someone with write access to it.

## Reporting

Found a compliance gap? Open an issue at
<https://github.com/CrispStrobe/Susurrus/issues>. Gaps in shipped marking or
consent gating are treated as defects, not feature requests.
