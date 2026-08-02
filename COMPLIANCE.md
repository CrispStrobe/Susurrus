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
| In-sample watermark (tier 1) | AudioSeal, learned | CrispASR binary, or `pip install 'susurrus[watermark]'` | Yes, incl. resampling and deliberate removal |
| In-sample watermark (tier 2) | Spread-spectrum comb — `utils/spread_spectrum.py` | Needs numpy + soundfile, both in `[tts]` | Yes for transcoding, **no for resampling** |
| Cryptographic | C2PA Content Credentials | CrispASR binary, or `pip install 'susurrus[c2pa]'` — included in `[tts]` | No (manifest is stripped) |
| Declarative | RIFF `LIST/INFO` (WAV) or ID3v2.4 (MP3) — `utils/ai_marking.py` | Always — no dependencies | No |

Two limits of tier 2 are worth stating plainly, because earlier versions of
this document overstated both.

It is **not dependency-free.** `utils/spread_spectrum.py` needs only numpy, but
the code that reads and rewrites the audio around it needs soundfile, and
soundfile ships in the `[tts]` extra rather than the base install. A bare
`pip install susurrus` therefore has the declarative marker and nothing else.
That install cannot run the Python TTS backends either, so in practice it
matters for the CrispASR-binary route, where marking is the binary's job and
the declarative marker is the floor Susurrus can add.

It does **not survive resampling.** The comb rides on fixed bin indices of a
fixed-size FFT, so it is tied to the sample rate it was embedded at. An MP3
round trip at the same rate is fine; 24 kHz → 16 kHz drops detection to chance.
Restoring the original rate restores detection, which is why a
44.1k → 16k → 44.1k round trip verifies and a plain 44.1k → 16k does not.
Compensating by sweeping candidate rate ratios was tried and rejected — a max
over ~12 hypotheses drove the false-positive rate to 100%. Use AudioSeal where
resampling is expected.

The layers are applied in a fixed order, and the order matters: the spoken
disclosure and the watermark change the samples, the declarative marker adds
metadata, and C2PA hashes the finished file — so C2PA must run last or its
manifest describes audio that no longer exists.

Art. 50(2) requires marking that is "effective, interoperable, robust and
reliable as far as technically feasible", and it has no "unless a dependency
is missing" clause. So neither layer that survives re-encoding is allowed to
be optional: when AudioSeal is absent, the spread-spectrum comb is embedded
instead. A `[tts]` install therefore ships two marks in the samples and one in
the metadata, not metadata alone.

### Marking fails closed

**If Susurrus cannot mark the output, it does not produce the output.** When no
machine-readable layer lands, or a cloning run cannot deliver its audible
disclosure, the file is **deleted** and the run exits 2 with a refusal naming
what was missing and how to satisfy it.

Deleting rather than warning is the point. Earlier versions logged a warning
and left the audio on disk under the name you asked for, which is not a
control — the unmarked file still exists and can still be shipped by anyone
who does not read stderr. An obligation that the software announces but does
not enforce is one the software has not implemented.

Two consequences worth being clear about:

- **A minimal install still works for WAV and MP3.** The declarative marker is
  pure standard library, so it always succeeds for those two containers. The
  gate bites on exotic containers (FLAC, Opus, M4A) with no C2PA and no
  soundfile, and on cloning to a non-WAV container without soundfile.
- **The check runs twice.** A cheap preflight refuses before any model loads,
  so an unmarkable request costs milliseconds rather than a full synthesis
  that is then thrown away. The authoritative check runs after synthesis,
  because only the finished file can say what actually landed.

The single way past the gate is `--accept-marking-responsibility`, which is
the same attestation described under "Opting out" below — one rule, not a
second switch. Either the software marked the output, or a named human said
they are taking the Art. 50 duty on.

The declarative marker uses each container's own metadata format: any ordinary
parser reads it, parsers that don't care skip it, and the audio samples are
untouched by that layer.

Watermarking preserves the file's channel count and sample format. A marking
step that quietly downmixed stereo to mono, or requantised 24-bit to 16-bit,
would be damaging the audio it is supposed to be annotating.

**Containers other than WAV and MP3** (FLAC, M4A, Opus …) get C2PA if the
library is installed, and the in-sample watermark wherever soundfile can
round-trip the container, but they have no *declarative* fallback. If neither
is available, synthesis to those containers is refused rather than performed
unmarked. Prefer `.wav` or `.mp3` if marking matters to you — those two never
depend on an optional package.

**Server mode is covered too, by a marking proxy.** `--mode server` used to
hand the socket to the CrispASR binary, which made an HTTP endpoint the one
route emitting synthetic audio Susurrus never saw — and an endpoint reaches
people who will never read a warning printed on the operator's terminal.

Susurrus now binds the port you asked for and starts the binary on loopback,
so it is back in the response path:

- `audio/*` responses are buffered and run through the same marking pipeline a
  local synthesis uses, then forwarded with the marked bytes and an
  `X-Susurrus-AI-Marked: EU-AI-Act-Art50-2` header.
- Audio that **cannot** be marked is answered with a 502 instead of being
  served unmarked. The gate works the same way on the wire as it does on disk.
- Everything else — transcription JSON, chat completions, Server-Sent Events —
  is streamed through untouched. Buffering an SSE stream would turn a working
  endpoint into one that appears to hang, so the relay is deliberately split:
  audio is the only thing worth holding.
- Protocol upgrades (WebSocket) are refused with a 501 naming the reason. They
  cannot be forwarded through this proxy, and half-forwarding one produces a
  connection that dies later for reasons nobody can trace.
- **Streaming synthesis** (`"stream": true` on `/v1/audio/speech`, which pushes
  audio per sentence) is refused with a 502. Marking needs the finished
  samples, so the proxy would have to buffer the whole stream — silently
  turning a streaming endpoint into a non-streaming one, for reasons invisible
  from the client. Set `stream=false`, or take the attestation and bypass the
  proxy.

If the proxy cannot be established — the port is taken, the backend does not
come up — the run is **refused** rather than falling back to serving directly.
Silently degrading to the behaviour the proxy exists to prevent is how a
control becomes decorative. The GUI's Tools → server toggle behaves the same
way and refuses to start if the proxy does not bind.

The one way to run unproxied is `--accept-marking-responsibility`, the same
attestation as everywhere else, and it prints the old warning: marking is then
the binary's alone. Note the proxy marks what the *endpoint returns*; if you
put a CDN or another proxy in front that re-encodes audio, you are back to
being the provider of whatever comes out the far end.

**Marking is verified, not assumed.** On the CrispASR routes the binary
applies the layers itself, and its support depends on build options, engine
capability and version. Susurrus reads the finished file back — declarative
marker, C2PA manifest, and in-sample detection where available — and reports
what is actually there. If even the floor fails the output is deleted and the
run refused. Earlier versions reported marking straight from the command line
flags, which meant a build without C2PA support still produced a confident
"Marked as AI-generated" over unmarked audio.

**The declarative floor is applied unconditionally on those routes**, not only
when the detectors come up empty. Gating it on "nothing was detected" made an
Art. 50(2) guarantee depend on a detector being right, and the spread-spectrum
detector was not reliable enough to carry that: measured over 1500 clips of
unwatermarked speech, harmonic stacks, tones and noise, it read ~12% of them as
watermarked. One false positive suppressed the floor and the gate then passed
on the phantom reading, releasing genuinely unmarked audio under a green status
line. The marker is idempotent, needs no dependency and does not touch the
samples, so there is no cost to applying it every time — and a marking
obligation should not turn on a coin flip. The detector statistic was fixed in
the same change (weighted correlation, threshold 0.65 → 0.78, false positives
12% → 1.2%, true detection 93% → 98%), but the enforcement no longer depends on
it being right.

That residual 1.2% still matters for the *verification* verbs.
`susurrus --detect-watermark FILE` reports the in-sample tier's own reading, so
roughly one unwatermarked file in eighty will come back
`detected_as_ai_generated: true` on the spread-spectrum tier alone. The report
names the detector that ran and gives the confidence — read both. A positive
from this tier is evidence, not proof; C2PA is the layer that answers
"is this file authentic" rather than "does this look marked".

C2PA signing needs an X.509 credential. Pass `--c2pa-cert` / `--c2pa-key` to
use your own; otherwise Susurrus generates a local CA + end-entity chain once
and caches it under `~/.local/share/susurrus/c2pa/`. That identity is
self-issued: it makes the output tamper-evident, and it asserts nothing about
who you are. For a credential that carries identity, supply a CA-issued
certificate. Signing is offline by default; set `SUSURRUS_C2PA_TSA` to an
RFC 3161 timestamp authority URL to add a trusted timestamp.

Verify marking on a file — this reports **both** layers and exits 0 if the
file is marked by either. It dispatches on the container, so an MP3 marked
with ID3 verifies as readily as a WAV marked with a RIFF chunk:

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

`--accept-marking-responsibility` is also the only thing that disarms the
fail-closed gate above. That is deliberate: an install that cannot mark and an
operator who has chosen not to mark are the same situation as far as the
output is concerned, and both should require the same explicit statement.
There is no separate "allow unmarked output" switch, and no configuration file
setting that turns the gate off quietly.

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

The disclosure is prepended in whatever container the output uses, not only
WAV. That matters because the Python-native backend that clones (chatterbox)
picks its encoder from the output extension, and the GUI's save dialog offers
`.mp3` — so restricting the disclosure to WAV would have left the one route
where Art. 50(4) actually applies producing undisclosed deepfakes. Non-WAV
concatenation needs soundfile, which ships in the `tts` extra.

Art. 50(2) and Art. 50(4) are tracked **separately**, because they fail
separately. If a run clones a voice and the audible disclosure does not land,
that is a refusal in its own right — the output is deleted even when the
machine-readable marking succeeded, because a listener hears no metadata and
no manifest. Reporting a marking success that is true of only one half was how
this used to behave, and it let an undisclosed deepfake through under a green
status line.

The disclosure is spoken in the backend's **own** voice, never in the cloned
one. While it is being synthesized the cloning reference is withheld from the
backend, so a disclaimer cannot be delivered by the impersonated speaker —
which would produce exactly the confusion the disclosure exists to prevent.

**A preset voice can be a deepfake too.** This document used to say that
synthesis in a stock voice is not one, and that the disclosure is owed only
when cloning from reference audio. That was wrong, and it was wrong about a
large share of the voices actually shipped.

Art. 3(60) defines a deep fake as AI-generated content "resembling existing
persons … that would falsely appear to a person to be authentic". It says
nothing about *how* the resemblance was obtained. A Piper voice trained on one
speaker's corpus resembles that speaker whether or not you passed a WAV; so
does a SpeechT5 voice conditioned on a CMU Arctic speaker embedding. Of the 27
TTS models CrispTTS classified against their providers' own model cards, 13
turned out to be real people.

So the disclosure now turns on two questions, not one:

| | Art. 50(4) disclosure |
| --- | --- |
| Cloned from reference audio | Always |
| Preset voice, `real_person` (piper, speecht5, crispasr:orpheus-de …) | Yes |
| Preset voice, `synthetic` (kokoro, bark …) | No |
| Preset voice, `unknown` (edge-tts, melotts …) | No — but warns once |

`unknown` is deliberately not a synonym for `synthetic`. It means nobody has
checked, which is a question for the deployer rather than a default to assume
away: forcing a disclosure on a guess would prepend a sentence to every stock
voice, and assuming "synthetic" would silently drop the duty for voices that
turn out to be people. The warning names the backend and says what to pass.

Override the shipped classification with `--speaker-identity`
(`real_person` | `synthetic` | `unknown`), or the **Preset voice is:** control
in the GUI's TTS settings, when you know better than the table — for a voice
pack you added, or a multi-voice backend where one voice differs from the rest.

**Coverage is partial, and the gap is loud rather than silent.** Susurrus
exposes 59 TTS backends; 20 are classified from provider documentation and the
other 39 resolve to `unknown` and warn once each. That is the honest state: a
classification nobody researched is not a classification, and the warning names
the backend so the answer can be supplied rather than assumed. Do not read an
unclassified backend as safe.

Where a single backend mixes real and designed voices — SauerkrautTTS ships two
studio-recorded people and two synthetic voices — a per-voice entry overrides
the backend-level answer. None of the backends Susurrus currently exposes is
mixed in that way, so that table is empty; it exists because the alternative
for a mixed model is classifying it by its riskiest voice and prepending a
disclosure to the rest.

**The verdict can depend on the checkpoint, not just the backend.** One CrispASR
backend serves several models with different answers: `crispasr:orpheus` runs
Canopy's base model (undocumented speakers, `unknown`) *and* Kartoffel's German
fine-tune (`real_person`), and `crispasr:kokoro` runs both hexgrad's English
voicepacks (`synthetic`) and a German fine-tune whose backbone is a corpus of
named narrators. Those are resolved by matching the loaded model's name.

Matching on a filename is against this project's own "classify by provenance,
not by filename" rule. It is used anyway because the alternative is no answer
at all, and because the failure is safe: a renamed checkpoint matches nothing,
falls back to the backend verdict or to `unknown`, and warns. A rename can turn
a known answer back into a question; it cannot turn `real_person` into
`synthetic`.

Four classifications in the first cut of this table were wrong, all from
reasoning about names rather than reading model cards, and are recorded here
because the same mistakes are easy to repeat:

- `crispasr:kartoffel-orpheus-de-synthetic` was marked `synthetic` **because
  the name says so**. It has not been researched. Guessing "synthetic" is the
  error that silently removes a disclosure, so it is now `unknown`.
- `crispasr:fastpitch` inherited a sibling project's `real_person` verdict for
  a German NeMo model. CrispASR ships NVIDIA's *English* FastPitch — different
  weights, so the verdict does not port.
- `crispasr:speecht5` takes its speaker x-vector from the operator, so no
  backend-level verdict can be right. The Python-native `speecht5` backend is
  genuinely different: it bakes in CMU ARCTIC speaker 7306 as its default.
- `crispasr:kokoro` was a blanket `synthetic`, which is wrong for the German
  HUI fine-tune.

The German Kokoro case is an open **conflict**, not a settled answer: CrispTTS
classifies Kokoro `synthetic` and is right about the English voicepacks, while
the German backbone is trained on the same named-narrator corpus that both
projects cite when marking FastPitch German `real_person`. It is held at
`unknown` rather than inheriting either neighbouring verdict.

Art. 50(2) marking applies to all of these regardless, and does.

One honest limit: when the disclosure is owed for a *preset* real-person voice,
it is spoken in that same voice, because there is no second voice to use. For a
cloned voice Susurrus withholds the reference so the disclaimer is never
delivered by the impersonated speaker; a fixed-speaker model offers no such
choice. An audible disclosure in the person's voice is still better than none.

**Speech-to-speech is gated more strictly than synthesis.** `--s2s` re-voices a
real recording of a real person, so the rights attestation is required whatever
the target voice is — unlike `--tts`, where a stock voice needs none. A preset
target still produces a recording of someone saying something in a voice that
is not theirs. `--s2s` also requires an explicit `--s2s-output`: audio at a path
Susurrus cannot name is audio it cannot mark, verify or delete. Both routes into
speech-to-speech — the subprocess one and the in-process FFI one — apply the
Art. 50 gate in Susurrus rather than leaving it to the binary. The subprocess
route previously did neither, and was the last synthesis path whose only check
lived outside this codebase.

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

Entries are SHA-256 hash-chained, so modification, deletion from the middle,
and reordering are all detectable:

```bash
susurrus --audit-log        # print the log and verify the chain; exit 1 if broken
```

Also available from the GUI under Tools → Biometric Audit Log.

**Truncation of the tail is caught by an anchor, not by the chain.** Deleting
the most recent *n* entries leaves a shorter chain that still verifies, because
every remaining entry's `prev_hash` still matches its predecessor. No hash chain
can detect this on its own — it needs a reference held outside itself. So after
every append, the entry count and head hash are mirrored into a sibling file,
`biometric.jsonl.anchor`, and `--audit-log` compares the two. Removing entries
from the end now contradicts the anchor and reports as a failure naming how many
are missing. Replacing the final entry is caught the same way.

The anchor is a sibling rather than a section of the log, because an anchor
stored inside the thing it anchors would be truncated by the same edit it is
supposed to detect.

**This raises the bar; it does not make the log tamper-proof.** Anyone who can
write the log can write the anchor beside it, and a log with no anchor at all
verifies as before — that case has to be tolerated, because logs written before
anchoring existed have none, and flagging every upgraded deployment as tampered
would train operators to ignore the check. Deletion from the *middle*,
modification of any entry, and reordering are all still detected by the chain
itself. If your obligations call for guarantees that survive an attacker with
write access, ship the log to append-only storage; nothing in a file the
attacker controls can give you that.

The log deliberately never contains the voice embedding or any audio — a
record-keeping mechanism must not become a second copy of the special-category
data it documents. The anchor holds only a count, a hash and a timestamp, for
the same reason.

If your deployment lands in Annex III, the remaining high-risk obligations
(risk management, data governance, human oversight, conformity assessment,
registration) fall on you as provider or deployer. Susurrus does not implement
them, and enabling these flags is not a conformity assessment.

## Provisions that do not apply, and why

A map that lists only the engaged obligations cannot be checked. These were
considered and found not to bite; if your fork changes that, they are yours.

**Art. 50(1) — informing people they are interacting with an AI system.** This
binds systems "intended to interact directly with natural persons". Susurrus
has no conversational surface: it processes files and text you hand it, and
never addresses a person as an interlocutor. Not engaged.

**Art. 50(3) — emotion recognition and biometric categorisation.** Susurrus
ships neither. Speaker enrollment and matching is biometric *identification*,
which is a different thing from *categorisation* (inferring attributes such as
sex, age or ethnicity) — so Art. 50(3)'s duty to inform exposed persons is not
engaged, while the Annex III(1)(a) analysis below is. Do not add emotion
inference and leave this paragraph standing.

**Art. 53 / 55 — general-purpose AI models.** Susurrus is not a GPAI provider.
It downloads and runs third-party models; it does not train, fine-tune or place
a model on the market under its own name. The GPAI obligations fall on whoever
provides the models you point it at. If you fine-tune a model and ship it as
part of a fork, you may become a provider of that model, and the fact that
Susurrus is MIT-licensed does nothing for you there.

**Art. 5 — prohibited practices** is treated below rather than here, because it
needs a statement about deployment and not only about the code.

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
compliance document that only lists the settled parts is not much use.

**The exemption covers marking, not silence.** Whatever the right reading of
Art. 50(2) is for translation, the output is still machine-generated, and the
Art. 50(4) duty on text published to inform the public on a matter of public
interest still lands on whoever publishes it. So Susurrus says so: the CLI
prints a disclosure after every translation, and the GUI shows one beside the
result box.

The disclosure goes on **stderr**, never stdout. `--mode translate` exists to
be piped and redirected, and a notice mixed into the payload would corrupt
every downstream use of it — a disclosure that makes the output unusable is one
that gets suppressed, and a suppressed disclosure discloses nothing. The GUI
follows the same principle: the notice sits beside the result box rather than
inside it, so text copied out of the box is the translation and nothing else.

If you publish machine-translated text to inform the public, the disclosure
duty is still yours. Susurrus now tells you that; it cannot do it for you.

Transcription is not covered: it is a transformation of a real recording, not
synthetic content. But see the AI-literacy note below on what a transcript is
and is not.

## Art. 4 — AI literacy

Providers and deployers must ensure a sufficient level of AI literacy among
staff operating these systems. Susurrus carries this as **Help → About AI in
Susurrus** in the GUI and **`susurrus --about-ai`** on the command line: what
the system is, its intended purpose, its known failure modes, and what it is
not validated for. Both render the same localized source, so they cannot drift
apart — and a CLI-only deployment, which is most server deployments, is no
longer the one that gets nothing. Practically, whoever runs Susurrus should
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

- Mark output you produced with `--accept-marking-responsibility`. That flag
  is the point at which the software stops and you start.
- Register anything with any authority.
- Perform a conformity assessment or produce technical documentation
  under Art. 11 / Annex IV.
- Establish a lawful basis for processing voice data.
- Disclose to your audience that content is AI-generated. Marking travels with
  the file and the marking proxy covers your own endpoint, but the duty to tell
  the people who see or hear the output is a deployer obligation that no
  library can discharge.
- Assess whether *your* deployment is high-risk.
- Guarantee the audit log's integrity against someone with write access to it.
  The anchor makes tail truncation *evident*; it does not prevent it.
- Mark synthetic audio that leaves your endpoint through something other than
  the marking proxy — a CDN, a re-encoding gateway, or the binary running
  unproxied under `--accept-marking-responsibility`.

## Reporting

Found a compliance gap? Open an issue at
<https://github.com/CrispStrobe/Susurrus/issues>. Gaps in shipped marking or
consent gating are treated as defects, not feature requests.
