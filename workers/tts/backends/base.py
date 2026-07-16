# workers/tts/backends/base.py
"""Abstract base class for TTS backends."""

from abc import ABC, abstractmethod


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

    def sign_output(self, output_path):
        """Sign the output audio with C2PA Content Credentials.

        Called automatically after synthesis for Python-native backends.
        CrispASR-based backends are already signed by the binary.

        Returns True if signed, False if c2pa-audio not available.
        """
        if output_path and output_path.lower().endswith(".wav"):
            try:
                from utils.c2pa_signing import sign_wav_file

                return sign_wav_file(output_path)
            except ImportError:
                pass
        return False

    def list_voices(self):
        """Return a list of available voice IDs for this backend."""
        return []

    def cleanup(self):
        """Release any resources held by this backend."""
