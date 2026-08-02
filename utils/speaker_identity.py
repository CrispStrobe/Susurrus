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
    # Trained on a single speaker's corpus (lessac, thorsten, …).
    "piper": "real_person",
    "crispasr:piper": "real_person",
    # SpeechT5 conditions on CMU Arctic speaker embeddings — real speakers.
    "speecht5": "real_person",
    "crispasr:orpheus-de": "real_person",
    # Kokoro's voices are blended rather than any one person.
    "kokoro-onnx": "synthetic",
    "crispasr:kokoro": "synthetic",
    "crispasr:bark": "synthetic",
    # Cloning backends: identity comes from the reference audio, and the
    # is_cloning path already forces a disclosure for that.
    "chatterbox": "unknown",
    # Microsoft does not document whose recordings these came from.
    "edge-tts": "unknown",
    "crispasr:melotts": "unknown",
    "crispasr:orpheus": "unknown",
}

_warned = set()


def resolve_speaker_identity(backend=None, override=None, voice=None):
    """Resolve whose voice a preset produces.

    Precedence: an explicit override, then a per-voice entry, then the
    backend's classification, then ``unknown``.

    Args:
        backend: TTS backend name, e.g. ``"piper"`` or ``"crispasr:kokoro"``.
        override: Operator's ``--speaker-identity`` value, if any.
        voice: Voice id, for future per-voice entries. Accepted now so callers
            need not change when one is added.

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

    if not backend:
        return "unknown"

    declared = BACKEND_SPEAKER_IDENTITY.get(str(backend).strip().lower())
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
