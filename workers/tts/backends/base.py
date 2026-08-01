# workers/tts/backends/base.py
"""Abstract base class for TTS backends."""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

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

    def is_cloning(self, voice=None):
        """Return True if this synthesis clones a voice from reference audio."""
        import os

        reference = voice or self.kwargs.get("voice") or self.kwargs.get("reference_audio")
        return bool(reference and os.path.isfile(reference))

    def apply_provenance(self, output_path, model=None, voice=None, locale=None):
        """Apply the EU AI Act Art. 50 obligations to synthesized audio.

        Four layers, applied in this order — **the order is load-bearing**:

        1. **Spoken disclosure** (Art. 50(4)) — prepends audible speech, so it
           must run before anything that depends on the sample data.
        2. **Neural watermark** (AudioSeal) — mutates samples; survives
           re-encoding. Optional dependency.
        3. **RIFF INFO marker** — declarative metadata, dependency-free, so a
           default install still emits marked audio.
        4. **C2PA Content Credentials** — hashes the finished file, so it must
           be last or the manifest describes audio that no longer exists.

        CrispASR-based backends do all of this inside the binary and override
        this method to a no-op.

        Args:
            output_path: Path to the synthesized audio.
            model: Optional model identifier recorded in the marker.
            voice: The voice used, to decide whether this was a cloning run.
            locale: Language for the spoken disclosure.

        Returns:
            dict with keys ``spoken``, ``watermark``, ``marker``, ``c2pa``
            (all bool) and ``opted_out`` (bool).
        """
        result = {
            "spoken": False,
            "watermark": False,
            "marker": False,
            "c2pa": False,
            "opted_out": False,
        }

        if not output_path or not output_path.lower().endswith(".wav"):
            return result

        if self.kwargs.get("accept_marking_responsibility"):
            logger.warning(
                "AI-content marking skipped (--accept-marking-responsibility). "
                "Responsibility for marking this output rests with the "
                "operator per EU AI Act Art. 50."
            )
            result["opted_out"] = True
            return result

        # 1. Spoken disclosure — cloning only, matching CrispASR's behaviour.
        if self.is_cloning(voice) and not self.kwargs.get("no_spoken_disclaimer"):
            try:
                from utils.spoken_disclosure import prepend_spoken_disclosure

                result["spoken"] = prepend_spoken_disclosure(self, output_path, locale=locale)
            except ImportError:
                pass

        # 2. Neural watermark — robust to re-encoding, optional dependency.
        if not self.kwargs.get("no_watermark"):
            try:
                from utils.audio_watermark import embed_watermark

                result["watermark"] = embed_watermark(output_path)
            except ImportError:
                pass

        # 3. Declarative marker — the layer that is always available.
        try:
            from utils.ai_marking import embed_wav_ai_marker

            result["marker"] = embed_wav_ai_marker(output_path, model=model or self.model_id)
        except ImportError:
            pass

        # 4. C2PA last: it hashes the final bytes.
        if not self.kwargs.get("no_c2pa"):
            try:
                from utils.c2pa_signing import sign_wav_file

                result["c2pa"] = sign_wav_file(
                    output_path,
                    cert_pem=self.kwargs.get("c2pa_cert"),
                    key_pem=self.kwargs.get("c2pa_key"),
                )
            except ImportError:
                pass

        if not result["c2pa"] and not result["marker"] and not result["watermark"]:
            logger.warning(
                "Could not mark %s as AI-generated. EU AI Act Art. 50(2) "
                "requires machine-readable marking of synthetic audio.",
                output_path,
            )

        return result

    def list_voices(self):
        """Return a list of available voice IDs for this backend."""
        return []

    def cleanup(self):
        """Release any resources held by this backend."""
