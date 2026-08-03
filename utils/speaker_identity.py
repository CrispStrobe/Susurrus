"""Whose voice is this? — the Art. 50(4) question a preset voice also asks.

Susurrus used to decide the audible AI disclosure on one thing: whether the
caller supplied reference audio. Cloning got a disclosure, a preset voice did
not, and COMPLIANCE.md said in as many words that "synthesis in a stock voice
is not a deepfake".

That is wrong for a large share of the presets actually shipped. Art. 3(60)
defines a deep fake as AI-generated content "resembling existing persons …
that would falsely appear to a person to be authentic" — it says nothing about
*how* the resemblance was obtained. A Piper voice trained on one speaker's
corpus resembles that speaker whether or not the user passed a WAV. Of the 27
TTS models CrispTTS classified against their upstream model cards, 13 turned
out to be real people; the researched values are reused here rather than
re-guessed, and this module deliberately mirrors CrispTTS's API so the two
projects can be compared line by line.

Three values, and the third is the important one:

``real_person``  the preset is an identifiable individual — a named donor, or
                 a corpus speaker such as VCTK's p225. Art. 50(4) engages.
``synthetic``    a designed or blended voice that is nobody in particular.
``unknown``      provenance not established. **Not** a synonym for synthetic:
                 it is a question the deployer has to answer, so it warns once
                 and does not force a disclosure. Forcing one would prepend a
                 sentence to every stock voice on a guess; silently assuming
                 "synthetic" would be the costly error in the other direction.
"""

import logging

logger = logging.getLogger(__name__)

#: Permitted values. Anything else resolves to ``unknown`` rather than being
#: trusted — a typo must not silently disable a disclosure.
SPEAKER_IDENTITY_VALUES = frozenset({"real_person", "synthetic", "unknown"})

#: Per-backend classification, keyed by Susurrus TTS backend name.
#:
#: Values come from CrispTTS's reading of each provider's own model card (see
#: its commit "read the upstream model cards; six more voices are real
#: people"), not from inspection of this repo. Where a backend is absent it
#: resolves to ``unknown`` and warns, which is the honest default: a missing
#: entry means nobody has checked, and that is exactly what the operator needs
#: told.
#:
#: These are *backend-level* answers. A multi-voice backend can mix real and
#: designed voices, so a per-voice override is the escape hatch — see
#: :func:`resolve_speaker_identity`.
BACKEND_SPEAKER_IDENTITY = {
    # -- real people -------------------------------------------------------
    # Piper's German voices are named for their donors — thorsten, eva_k,
    # karlsson, kerstin, ramona — who are the HUI-Audio-Corpus-German and
    # Thorsten-Voice narrators. Named individuals who published recordings.
    "piper": "real_person",
    "crispasr:piper": "real_person",
    # The *Python-native* SpeechT5 backend loads Matthijs/cmu-arctic-xvectors
    # and defaults to speaker_idx 7306 — a CMU ARCTIC x-vector baked in, so the
    # voice you hear without doing anything is one of bdl, slt, jmk, awb, rms,
    # clb, ksp: identifiable recorded people, pseudonymous exactly as VCTK's
    # p225 is. This verdict does NOT port to crispasr:speecht5, which takes the
    # x-vector from the operator — see the unknowns below.
    "speecht5": "real_person",
    # Kartoffel-Orpheus "natural" is the provider's own word: fine-tuned
    # "primarily on natural human speech recordings" — permissive podcasts,
    # lectures, OER — with its 19 speakers extracted from those recordings.
    # Real people who spoke in public.
    "crispasr:kartoffel-orpheus-de-natural": "real_person",
    # The card names its speakers: "trained on 34 speakers, characterized by
    # name (e.g. Jon, Lea, Gary, Jenna, Mike, Laura)", over mls_eng and
    # libritts_r_filtered — LibriTTS-R derives from LibriVox, whose narrators
    # are real people. Named speakers from real recordings.
    "crispasr:parler-tts": "real_person",
    # -- designed voices ---------------------------------------------------
    # hexgrad/Kokoro-82M's voicepacks are style vectors, documented upstream as
    # designed/blended rather than any one person. The German HUI fine-tune is
    # the exception and is handled by MODEL_RULES below, not here.
    "kokoro-onnx": "synthetic",
    "crispasr:kokoro": "synthetic",
    # The card answers the Art. 3(60) question directly: "the model was not
    # fine-tuned on a specific voice. Hence, you will get different voices
    # every time you run the model." A voice that is different on every run
    # cannot resemble one existing person.
    "crispasr:dia": "synthetic",
    "crispasr:dia-tts": "synthetic",
    # -- checked, genuinely undocumented -----------------------------------
    # Recorded so the same dead ends are not re-searched. Each of these was
    # looked for and not found; none is a shrug.
    #
    # Microsoft's TTS transparency note defines "voice talent" only for custom
    # neural voice. On prebuilt voices it is silent, and whoever de-DE-Katja
    # and friends were modelled on is not publicly identified.
    "edge-tts": "unknown",
    # Documents its architecture lineage (VITS2/Bert-VITS2), not its speakers.
    "crispasr:melotts": "unknown",
    # Canopy Labs discloses 100k+ hours of "permissive/non-copyrighted" audio
    # and nothing about the origin of tara, leah, jess, leo, dan, mia, zac,
    # zoe. Checked the HF card, the GitHub repo and the web.
    "crispasr:orpheus": "unknown",
    "crispasr:lex-au-orpheus-de": "unknown",
    # Banaxi-Tech's card, found since: en-us on LJSpeech (Linda Johnson again),
    # de-de on ThorstenVoice credited "Voice: Thorsten Müller" — the same donor
    # as piper's de_DE-thorsten, reached by a second route.
    "crispasr:bananamind-tts": "real_person",
    "crispasr:bananamind-tts-de": "real_person",
    # Was "synthetic" here, inherited from a sibling project. Checked the card
    # at huggingface.co/suno/bark: it says *nothing* about where the 100+
    # speaker presets came from — not that they are designed, not that they are
    # actors. An undocumented voice is a question, not a synthetic one, and
    # inheriting a verdict is not the same as having evidence for it.
    "crispasr:bark": "unknown",
    "crispasr:bark-tts": "unknown",
    # Card checked: technical architecture and usage restrictions, nothing at
    # all about where the speaker voices came from.
    "crispasr:vibevoice": "unknown",
    "crispasr:vibevoice-1.5b": "unknown",
    # Zero-shot models: the voice comes from a reference the operator supplies,
    # so there is no preset whose provenance could be classified. Both cards
    # were read and neither documents the training voices either. The cloning
    # gate is what matters on these routes, not this table.
    "crispasr:f5-tts": "unknown",
    "crispasr:indextts": "unknown",
    # Zonos: "trained on more than 200k hours of varied multilingual speech",
    # and nothing about whose. Clones from a reference clip, which is the
    # gated path.
    "crispasr:zonos": "unknown",
    "crispasr:zonos-tts": "unknown",
    # OuteTTS: "~60k hours of audio", no sourcing or consent detail. Ships
    # named default profiles (EN-FEMALE-1-NEUTRAL) whose origin is undocumented.
    "crispasr:outetts": "unknown",
    # CosyVoice: zero-shot cloning, and on training data only "Some examples
    # are sourced from the internet. If any content infringes on your rights,
    # please contact us to request its removal" — which is a takedown notice,
    # not provenance. Its voice *bank* is separately gated as cloning.
    "crispasr:cosyvoice3-tts": "unknown",
    # sesame: "a base generation model ... has not been fine-tuned on any
    # specific voice". That is the provider answering the Art. 3(60) question
    # for the preset path directly, so it is taken as evidence rather than held
    # at unknown — the wording is weaker than Dia's "different voices every
    # time", but it says the thing that matters. The intended route is
    # prompting with audio context, which is cloning and discloses anyway.
    "crispasr:csm": "synthetic",
    "crispasr:csm-tts": "synthetic",
    "crispasr:sesame": "synthetic",
    # Qwen3-TTS ships named preset personas (Cherry — "a sunny, positive,
    # friendly and natural young woman" — Ethan, Chelsie, Serena …) over 5M+
    # hours of multilingual speech, and documents neither who recorded them nor
    # that nobody did. Commercial preset voices of this kind are commonly voice
    # actors, which would make them real people whose identity is simply not
    # published — so "synthetic" would be a guess in the direction that removes
    # a disclosure.
    "crispasr:qwen3-tts": "unknown",
    "crispasr:qwen3-tts-1.7b-base": "unknown",
    # CustomVoice takes the speaker from the operator, and VoiceDesign builds
    # one from a text description. Neither has a fixed preset whose provenance
    # could be classified; the cloning route already forces the disclosure.
    "crispasr:qwen3-tts-customvoice": "unknown",
    "crispasr:qwen3-tts-1.7b-customvoice": "unknown",
    "crispasr:qwen3-tts-1.7b-voicedesign": "unknown",
    "crispasr:irodori-tts-voicedesign": "unknown",
    # Now read rather than inferred: the card says "trained on synthetic German
    # speech". It was held at unknown while only the repo name said so — the
    # name was never the evidence, and it still isn't; the card is.
    "crispasr:kartoffel-orpheus-de-synthetic": "synthetic",
    # Card read since: nvidia/tts_en_fastpitch is "trained on LJSpeech" —
    # 13,100 clips of one LibriVox narrator, Linda Johnson. Susurrus held this
    # at unknown while the hypothesis was only conventional wisdom; the card
    # settles it.
    "crispasr:fastpitch": "real_person",
    # microsoft/speecht5_tts takes its 512-d speaker x-vector from the operator
    # via --voice, so the identity is per-invocation and no backend-level
    # verdict can be right. Distinct from the Python-native speecht5 above,
    # which bakes in a CMU ARCTIC default.
    "crispasr:speecht5": "unknown",
    # -- cloning backends ---------------------------------------------------
    # Identity comes from the reference audio, and is_cloning already forces
    # the disclosure for that. Listed so they read as considered, not missed.
    "chatterbox": "unknown",
    "crispasr:chatterbox": "unknown",
    "crispasr:chatterbox-turbo": "unknown",
}

#: Per-voice overrides, keyed by ``(backend, voice)`` with both lowercased.
#:
#: A backend-level answer is wrong for a model that ships some real voices and
#: some designed ones — SauerkrautTTS is the known example, where Tom and Anna
#: are studio recordings of people and Max and Lena are not. None of the
#: backends Susurrus currently exposes is mixed in that way (the Kartoffel
#: natural/synthetic split is two separate backends, and every Piper voice is
#: a donor while every Kokoro voice is a blend), so this is empty. It exists
#: because the alternative for a mixed model is classifying it by its riskiest
#: voice and prepending a disclosure to the rest.
#: ``(backend, voice) -> (identity, evidence)``.
#:
#: The evidence travels with the verdict rather than in a comment beside it,
#: because the rule this table has to obey is about *what the verdict rests
#: on*, not about which value it takes. Classifying a voice ``synthetic`` from
#: its name silently removes a disclosure; classifying it ``synthetic`` because
#: the provider says the base was "trained entirely on synthetic (TTS-generated)
#: audio" is a documented fact. Only the second is allowed, and a test enforces
#: it by requiring every entry to cite something.
#:
#: This is where the kokoro-de-hui question actually lives. The checkpoint is
#: documented speaker-neutral — "This is a base model, not a voice" — so the
#: voice a listener hears is the pack's, and the pack is what gets classified.
VOICE_SPEAKER_IDENTITY = {
    ("crispasr:kokoro", "df_eva"): (
        "real_person",
        "per-speaker pack from HUI-Audio-Corpus-German carrying the narrator's "
        "own name; HUI is built from librivox.org recordings",
    ),
    ("crispasr:kokoro", "dm_bernd"): (
        "real_person",
        "per-speaker pack from HUI-Audio-Corpus-German carrying the narrator's "
        "own name; HUI is built from librivox.org recordings",
    ),
    # Person-shaped names, and the answer is still synthetic — because the
    # evidence is the base's training data, not the name. Susurrus held these
    # at unknown while only the name was known, which was right then and is
    # superseded now.
    ("crispasr:kokoro", "df_victoria"): (
        "synthetic",
        "kikiri fine-tune over kikiri-german-base-51speakers-synthetic, "
        '"trained entirely on synthetic (TTS-generated) audio"',
    ),
    ("crispasr:kokoro", "dm_martin"): (
        "synthetic",
        "kikiri fine-tune over kikiri-german-base-51speakers-synthetic, "
        '"trained entirely on synthetic (TTS-generated) audio"',
    ),
    ("crispasr:kokoro", "af_heart"): (
        "synthetic",
        "hexgrad/Kokoro-82M voicepack; upstream documents these as designed "
        "style vectors rather than any one person",
    ),
    ("crispasr:kokoro", "ef_dora"): (
        "synthetic",
        "hexgrad/Kokoro-82M voicepack; upstream documents these as designed "
        "style vectors rather than any one person",
    ),
    ("crispasr:kokoro", "ff_siwis"): (
        "synthetic",
        "hexgrad/Kokoro-82M voicepack; upstream documents these as designed "
        "style vectors rather than any one person",
    ),
}

# The Python-native kokoro-onnx backend serves the same packs by the same
# names, so it answers identically rather than maintaining a second list.
VOICE_SPEAKER_IDENTITY.update(
    {
        ("kokoro-onnx", voice): value
        for (_backend, voice), value in list(VOICE_SPEAKER_IDENTITY.items())
    }
)

#: ``(backend, model-name substring) -> identity``, checked before the
#: backend-level table.
#:
#: One CrispASR backend serves many checkpoints, and they do not share an
#: answer: ``crispasr:kokoro`` runs both hexgrad's English voicepacks and a
#: German fine-tune whose backbone is a corpus of named narrators. A
#: backend-level verdict is simply not expressible for those.
#:
#: Matching on a file name is against this project's own "classify by
#: provenance, not by filename" rule, and is used anyway because the
#: alternative is no answer at all — and because the failure is *safe*: a
#: renamed checkpoint matches nothing, falls through to the backend table or to
#: unknown, and warns. A rename can turn a known answer back into a question;
#: it cannot turn ``real_person`` into ``synthetic``.
MODEL_RULES = (
    # Held at unknown, but for a narrower reason than before. The upstream card
    # (huggingface.co/cstr/kokoro-de-hui-base-GGUF) settles half of it: "This is
    # a base model, not a voice" — a speaker-neutral Stage-1 multispeaker base
    # over the HUI corpus's 51 speakers, with a per-speaker duration cap
    # specifically so no one of them dominates. It does not reproduce an
    # individual on its own.
    #
    # What it cannot settle is the half that reaches a listener: the base
    # produces nothing without a voicepack, and the voice you hear is the
    # voicepack's. Those are separate artifacts with their own provenance, and
    # at least one shipped name (``df_eva``) matches a named HUI narrator. So
    # the pairing stays a question until the packs are researched — which is a
    # per-voice answer, not a per-model one. See VOICE_SPEAKER_IDENTITY.
    ("crispasr:kokoro", "hui", "unknown"),
    ("kokoro-onnx", "hui", "unknown"),
    # One backend, several checkpoints, different answers.
    ("crispasr:orpheus", "kartoffel-orpheus-de-natural", "real_person"),
)


#: GGUF keys CrispASR stamps into a checkpoint, so the answer travels with the
#: weights instead of with the filename.
STAMP_KEY = "crispasr.voice.speaker_identity"
STAMP_EVIDENCE_KEY = "crispasr.voice.speaker_identity_evidence"

_stamp_cache = {}


def identity_from_stamp(model):
    """Return ``(identity, evidence)`` stamped into *model*, or ``(None, None)``.

    This is the answer to prefer whenever it exists: it is written by whoever
    converted the checkpoint, it survives a rename, and it does not depend on
    Susurrus recognising a filename. MODEL_RULES stays as the fallback for the
    checkpoints published before stamping existed — which is most of them.

    Cached on (path, mtime, size) because the caller asks once per synthesis
    and the answer only changes if the file does.
    """
    if not model:
        return None, None

    import os

    try:
        stat = os.stat(model)
        cache_key = (model, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None, None

    if cache_key in _stamp_cache:
        return _stamp_cache[cache_key]

    from utils.gguf_metadata import read_string_keys

    kv = read_string_keys(model, (STAMP_KEY, STAMP_EVIDENCE_KEY))
    value = (kv.get(STAMP_KEY) or "").strip().lower()
    evidence = kv.get(STAMP_EVIDENCE_KEY) or ""

    if value and value not in SPEAKER_IDENTITY_VALUES:
        logger.warning(
            "Checkpoint %s stamps an unrecognised speaker identity %r; ignoring it.",
            model,
            value,
        )
        value = ""

    result = (value or None, evidence or None)
    _stamp_cache[cache_key] = result
    return result


def identity_for_model(backend, model):
    """Return a model-specific verdict for *backend*, or None if no rule hits."""
    if not backend or not model:
        return None
    key = str(backend).strip().lower()
    haystack = str(model).strip().lower()
    for rule_backend, needle, identity in MODEL_RULES:
        if key == rule_backend and needle in haystack:
            return identity
    return None


_warned = set()


def resolve_speaker_identity(backend=None, override=None, voice=None, model=None):
    """Resolve whose voice a preset produces.

    Precedence: an explicit override, then a per-voice entry, then the
    backend's classification, then ``unknown``.

    Args:
        backend: TTS backend name, e.g. ``"piper"`` or ``"crispasr:kokoro"``.
        override: Operator's ``--speaker-identity`` value, if any.
        voice: Voice id, for backends whose voices differ from each other.
        model: Loaded checkpoint name or path. Only the tail is inspected, and
            only to distinguish checkpoints that a single backend serves.

    Returns:
        One of :data:`SPEAKER_IDENTITY_VALUES`.
    """
    if override:
        value = str(override).strip().lower()
        if value in SPEAKER_IDENTITY_VALUES:
            return value
        logger.warning(
            "Unrecognised speaker identity %r; treating as 'unknown'. Expected one of: %s",
            override,
            ", ".join(sorted(SPEAKER_IDENTITY_VALUES)),
        )
        return "unknown"

    key = str(backend).strip().lower() if backend else None

    if key and voice:
        entry = VOICE_SPEAKER_IDENTITY.get((key, str(voice).strip().lower()))
        per_voice = entry[0] if isinstance(entry, tuple) else entry
        if per_voice in SPEAKER_IDENTITY_VALUES:
            return per_voice

    # A stamp inside the checkpoint beats every guess about it: it is written
    # by whoever converted the weights and survives a rename. It sits below the
    # per-voice answer because a stamp describes the model, and for a
    # base-plus-voicepack architecture the pack is what a listener hears.
    stamped, _evidence = identity_from_stamp(model)
    if stamped in SPEAKER_IDENTITY_VALUES:
        return stamped

    # Before the backend table: a checkpoint-specific answer is more precise
    # than a blanket one, and for backends that serve several models it is the
    # only answer that can be right.
    per_model = identity_for_model(key, model)
    if per_model in SPEAKER_IDENTITY_VALUES:
        return per_model

    if not key:
        return "unknown"

    declared = BACKEND_SPEAKER_IDENTITY.get(key)
    if declared in SPEAKER_IDENTITY_VALUES:
        return declared
    return "unknown"


def requires_spoken_disclosure(is_cloning, speaker_identity, backend=None):
    """Whether this output owes an audible Art. 50(4) disclosure.

    True when the voice is cloned from a reference recording, or when the
    preset voice belongs to an identifiable person. ``unknown`` warns once per
    backend and returns False — see the module docstring for why that is not
    the same as deciding the voice is synthetic.
    """
    if is_cloning:
        return True
    if speaker_identity == "real_person":
        return True
    if speaker_identity == "unknown":
        warn_unknown_once(backend)
    return False


def warn_unknown_once(backend):
    """Tell the operator, once per backend, that the question is unanswered."""
    key = str(backend or "<unknown-backend>")
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(
        "Backend '%s' does not record whether its preset voice belongs to a "
        "real person, so no spoken AI disclosure was added. If the voice is an "
        "identifiable individual, the output is a deep fake under EU AI Act "
        "Art. 3(60) and the Art. 50(4) duty to disclose it is yours. Pass "
        "--speaker-identity real_person to have Susurrus prepend the "
        "disclosure, or --speaker-identity synthetic to silence this.",
        key,
    )


def _reset_warnings_for_tests():
    """Clear the warn-once state (test helper)."""
    _warned.clear()
