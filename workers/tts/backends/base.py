# workers/tts/backends/base.py
"""Abstract base class for TTS backends."""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

#: Backends whose ``--voice`` selects an entry from a baked voice *bank* rather
#: than naming a preset.
#:
#: cosyvoice3 and kugelaudio keep their voices inside a bundle discovered
#: alongside the model, and ``--voice`` picks one **by name**. So the gate below
#: received a bare string that resolved to no file on disk, concluded "preset",
#: and let a zero-shot voice clone through with no attestation and no Art. 50(4)
#: disclosure. ``--voice victim.wav`` on the same backend *was* gated, which is
#: exactly why this looked covered. CrispASR found it in its own gate and calls
#: these bundles "baked voice-clone bundles" in the backend's own header.
#:
#: Susurrus cannot open the bundle to tell a cloned entry from a designed one,
#: so it treats every selection from one as cloning. That over-gates a bank
#: entry that happens to be synthetic; the alternative under-gates a real
#: person's cloned voice, and only one of those is a compliance failure.
VOICE_BANK_BACKENDS = frozenset(
    {
        "crispasr:cosyvoice3-tts",
        "crispasr:kugelaudio",
    }
)

#: GGUF key a CrispASR voice baker stamps into a pack derived from someone's
#: recording. Its absence means "preset", not "cannot say" — see
#: :func:`TTSBackend.pack_is_a_clone`.
CLONED_FROM_RECORDING_KEY = "crispasr.voice.cloned_from_recording"

#: Raised when voice cloning is attempted without a rights attestation.
CLONE_CONSENT_ERROR = (
    "Voice cloning requires a rights attestation. Pass --i-have-rights (CLI) "
    "or tick 'I have rights to clone this voice' (GUI) to confirm the cloned "
    "speaker consented, or that this is your own voice. See EU AI Act Art. 50."
)


def pack_is_a_clone(path):
    """Is this voice file a clone of someone, or a shipped preset?

    A recording is always a clone: passing ``victim.wav`` is the plainest case
    there is. A **voice pack** is not, by default — kokoro, qwen3-tts, vibevoice
    and miotts all ship synthetic or upstream-licensed presets as ``.gguf``, and
    a pack baked from a recording is indistinguishable from one of those by
    suffix alone. Susurrus used to treat every existing file as a clone, which
    meant ``--voice kokoro-voice-af_heart.gguf`` demanded a speaker-consent
    attestation for a designed voice: a refusal nobody can satisfy honestly, on
    a documented example, which teaches operators to pass --i-have-rights
    reflexively. An attestation that is always required means nothing.

    So packs are presets unless they say otherwise. CrispASR's bakers stamp
    ``crispasr.voice.cloned_from_recording`` into packs derived from a user
    recording, and this reads it back.

    The honest limitation, inherited: a pack baked before that stamp existed
    carries no provenance and reads as a preset. Re-bake it to gate it. The
    cases where provenance is *knowable without asking the file* — a recording
    passed directly, a bank selection, a voice-dir resolution — do not depend on
    the stamp at all.
    """
    import os

    if os.path.splitext(str(path))[1].lower() != ".gguf":
        return True  # a recording, or anything else handed over directly

    from utils.gguf_metadata import read_string_keys

    stamped = read_string_keys(path, (CLONED_FROM_RECORDING_KEY,)).get(
        CLONED_FROM_RECORDING_KEY, ""
    )
    return str(stamped).strip().lower() in {"true", "1", "yes"}


def resolve_voice_dir_reference(voice_dir, candidates):
    """Resolve a bare voice name against *voice_dir*, or return None.

    Unlike the voice-bank case this can be answered exactly, by asking the
    filesystem, so it gates on a file that genuinely exists rather than on the
    backend's identity. A name that resolves to nothing stays a preset.
    """
    import os

    if not voice_dir:
        return None

    for candidate in candidates:
        if not candidate or os.sep in str(candidate) or os.path.splitext(str(candidate))[1]:
            continue
        for extension in (".wav", ".gguf"):
            path = os.path.join(str(voice_dir), f"{candidate}{extension}")
            if os.path.isfile(path):
                return path
    return None


def would_clone(backend_name=None, voices=(), voice_dir=None):
    """Return the reference this synthesis would clone from, or None.

    The one implementation of "is this cloning?", callable without a backend
    instance so the CLI preflight can ask the *same* question the authoritative
    gate asks. It previously asked a narrower one — ``os.path.isfile(voice)``
    alone — which missed both documented no-path cloning routes: a bare
    ``--voice`` name resolved against ``--voice-dir``, and a selection from a
    baked voice bank. The gate after synthesis still caught those, so nothing
    unmarked escaped; what the operator got instead was a full model load and a
    synthesis before a refusal that was knowable in milliseconds.

    Three ways to clone, in the order they can be answered cheaply:

    1. a path handed over directly, unless it is an unstamped voice pack;
    2. a bare name that resolves to a file under ``--voice-dir``;
    3. any selection on a backend whose voices live in a bank.
    """
    import os

    candidates = [v for v in voices if v]

    for candidate in candidates:
        if os.path.isfile(candidate) and pack_is_a_clone(candidate):
            return candidate

    resolved = resolve_voice_dir_reference(voice_dir, candidates)
    if resolved:
        return resolved

    if backend_name and str(backend_name).strip().lower() in VOICE_BANK_BACKENDS:
        return str(candidates[0]) if candidates else None

    return None


class TTSBackend(ABC):
    """Base class for all TTS backends.

    Each backend implements ``synthesize()`` which converts text to audio
    and writes the result to an output file. Optionally, backends can
    expose ``list_voices()`` to enumerate available voice options.
    """

    def __init__(self, model_id=None, device="cpu", language=None, **kwargs):
        self.model_id = model_id
        self.device = device
        self.language = language
        self.kwargs = kwargs

    @abstractmethod
    def synthesize(self, text, output_path, voice=None):
        """Synthesize *text* to audio and write to *output_path*.

        Args:
            text: The text to synthesize.
            output_path: Path for the output audio file.
            voice: Optional voice ID override.

        Returns:
            The path to the written audio file.
        """

    def require_clone_consent(self, reference_audio):
        """Refuse to clone a voice without an explicit rights attestation.

        Call this from ``synthesize()`` at the point where the effective
        reference audio is resolved, so every route into cloning passes
        through one gate.

        Args:
            reference_audio: The resolved reference audio path, or a falsy
                value when no cloning is requested.

        Raises:
            PermissionError: If a reference audio is given without consent.
        """
        if not reference_audio:
            return
        if self.kwargs.get("i_have_rights"):
            return
        raise PermissionError(CLONE_CONSENT_ERROR)

    def resolve_reference_audio(self, voice=None):
        """Return the reference audio this synthesis would clone from, or None.

        Every candidate is checked, not just the first truthy one. The earlier
        ``voice or kwargs["voice"] or kwargs["reference_audio"]`` short-circuit
        meant a voice *name* (``"af_sarah"``) masked a real reference-audio
        path, so cloning went undetected and the Art. 50(4) audible disclosure
        was silently skipped.

        Returns None while a spoken disclosure is being synthesized: the
        disclosure must be spoken in the backend's own voice, never in the
        cloned one — announcing "this audio is AI-generated" *as* the person
        being impersonated is the confusion the disclosure exists to prevent.

        Delegates to :func:`would_clone` so the CLI preflight and this gate
        cannot answer the same question differently. Every candidate is offered
        to every test, which gates slightly more than the old per-stage tuples
        did — a ``tts_voice`` that is itself a path now counts. That is the
        direction this module errs in deliberately: over-gating a designed
        voice costs an attestation, under-gating a real person's voice costs a
        disclosure.
        """
        if getattr(self, "_synthesizing_disclosure", False):
            return None

        return would_clone(
            backend_name=self.speaker_backend_name(),
            voices=(
                voice,
                self.kwargs.get("voice"),
                self.kwargs.get("reference_audio"),
                self.kwargs.get("tts_voice"),
            ),
            voice_dir=self.kwargs.get("voice_dir"),
        )

    #: Kept as methods so existing callers and tests keep working; the logic
    #: lives at module level so the CLI preflight can reach it without
    #: constructing a backend.
    file_is_a_clone = staticmethod(pack_is_a_clone)

    def voice_dir_reference(self, voice=None):
        """Resolve a bare voice name against ``--voice-dir``, or return None."""
        if getattr(self, "_synthesizing_disclosure", False):
            return None

        return resolve_voice_dir_reference(
            self.kwargs.get("voice_dir"),
            (voice, self.kwargs.get("voice"), self.kwargs.get("tts_voice")),
        )

    def voice_bank_selection(self, voice=None):
        """Return the bank entry this synthesis would clone, or None.

        See :data:`VOICE_BANK_BACKENDS` for why a bare name counts as cloning
        on these backends.
        """
        if getattr(self, "_synthesizing_disclosure", False):
            return None
        if self.speaker_backend_name() not in VOICE_BANK_BACKENDS:
            return None
        for candidate in (voice, self.kwargs.get("voice")):
            if candidate:
                return str(candidate)
        return None

    def is_cloning(self, voice=None):
        """Return True if this synthesis clones a voice from reference audio."""
        return self.resolve_reference_audio(voice) is not None

    def apply_provenance(self, output_path, model=None, voice=None, locale=None):
        """Apply the EU AI Act Art. 50 obligations to synthesized audio.

        Delegates to :func:`utils.provenance.apply_provenance`, which is the
        single implementation shared with the routes that are not
        ``TTSBackend`` subclasses. See that module for the layer ordering and
        why it matters.

        CrispASR-based backends mark inside the binary and override this to
        *verify* the result rather than re-apply it.

        Args:
            output_path: Path to the synthesized audio.
            model: Optional model identifier recorded in the marker.
            voice: The voice used, to decide whether this was a cloning run.
            locale: Language for the spoken disclosure.

        Returns:
            dict with keys ``spoken``, ``watermark``, ``marker``, ``c2pa``,
            ``opted_out`` and ``unsupported_format`` (all bool).
        """
        from utils.provenance import apply_provenance as _apply

        return _apply(
            output_path,
            options=self.kwargs,
            model=model or self.model_id,
            backend=self,
            is_cloning=self.is_cloning(voice),
            locale=locale,
            speaker_backend=self.speaker_backend_name(),
        )

    def speaker_backend_name(self):
        """Name used to look this backend up in the speaker-identity table.

        Defaults to the kwarg the CLI and GUI already thread through, so a
        backend only overrides this if its registered name differs from the one
        it is constructed with.
        """
        return self.kwargs.get("tts_backend_name")

    def list_voices(self):
        """Return a list of available voice IDs for this backend."""
        return []

    def cleanup(self):
        """Release any resources held by this backend."""
