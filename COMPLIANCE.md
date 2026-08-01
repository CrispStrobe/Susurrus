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

Susurrus applies marking on **every** TTS path. Two layers:

| Layer | Mechanism | Availability | Survives re-encoding |
| --- | --- | --- | --- |
| Neural watermark | AudioSeal | CrispASR binary, or `pip install audioseal` | Yes |
| Cryptographic | C2PA Content Credentials | CrispASR binary, or `pip install c2pa-audio` | No (manifest is stripped) |
| Declarative | RIFF `LIST/INFO` chunk (`utils/ai_marking.py`) | Always — no dependencies | No |

The layers are applied in a fixed order, and the order matters: the spoken
disclosure and the neural watermark change the samples, the declarative marker
adds metadata, and C2PA hashes the finished file — so C2PA must run last or its
manifest describes audio that no longer exists.

The declarative layer exists because C2PA signing degrades to a no-op when the
optional `c2pa-audio` library is absent. Art. 50(2) has no "unless a dependency
is missing" clause, so a default install still emits marked audio. The marker
is a standard RIFF chunk: any parser can read it, parsers that don't care skip
it, and the audio samples are untouched.

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
C2PA (install `c2pa-audio`), and for survival through re-encoding use a
CrispASR backend, which adds an AudioSeal neural watermark.

**Opting out.** `--accept-marking-responsibility` produces completely unmarked
audio. This is deliberately a single explicit flag, not a side effect of any
other option: using it transfers the Art. 50 marking obligation to you as the
operator. `--no-c2pa` skips only the cryptographic layer and `--no-watermark`
only the neural one; the declarative marker still applies in both cases.

## Art. 50(4) — deepfake disclosure

> Deployers of an AI system that generates or manipulates image, audio or
> video content constituting a deep fake shall disclose that the content has
> been artificially generated or manipulated.

Voice cloning is gated on an explicit rights attestation. Cloning is refused —
before any model is loaded — unless you pass `--i-have-rights` (CLI) or tick
the consent box (GUI). The attestation is yours to make: the application will
never set it on your behalf, including from the Voice Clone Wizard.

What the attestation means: *this is my own voice, or the speaker consented to
having their voice cloned.*

Marking (above) satisfies the machine-readable half of disclosure. On top of
that, **every** backend prepends an audible spoken disclaimer to cloned audio —
CrispASR does it in-binary, and the Python-native backends synthesize the
phrase with the same model and concatenate it (`utils/spoken_disclosure.py`).
The phrase is localized, so a German user gets a German disclosure. Suppress it
with `--no-spoken-disclaimer`; machine-readable marking still applies.

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

## Art. 4 — AI literacy

Providers and deployers must ensure a sufficient level of AI literacy among
staff operating these systems. Practically, whoever runs Susurrus should
understand:

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
