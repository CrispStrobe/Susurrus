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

#: Raised when voice cloning is attempted without a rights attestation.
CLONE_CONSENT_ERROR = (
    "Voice cloning requires a rights attestation. Pass --i-have-rights (CLI) "
    "or tick 'I have rights to clone this voice' (GUI) to confirm the cloned "
    "speaker consented, or that this is your own voice. See EU AI Act Art. 50."
)


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
        """
        import os

        if getattr(self, "_synthesizing_disclosure", False):
            return None

        for candidate in (
            voice,
            self.kwargs.get("voice"),
            self.kwargs.get("reference_audio"),
        ):
            if candidate and os.path.isfile(candidate):
                return candidate

        # A voice-bank selection is a clone that never touches the filesystem,
        # so ``isfile`` cannot see it. Returning the name keeps one code path:
        # the consent gate, is_cloning() and the Art. 50(4) disclosure all read
        # this one answer, and none of them needs to learn about banks.
        return self.voice_bank_selection(voice)

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
