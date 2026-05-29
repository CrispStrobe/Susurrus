# workers/translation/backends/base.py
"""Abstract base class for translation backends."""

from abc import ABC, abstractmethod


class TranslationBackend(ABC):
    """Base class for all translation backends."""

    def __init__(self, model_id=None, device="cpu", **kwargs):
        self.model_id = model_id
        self.device = device
        self.kwargs = kwargs

    @abstractmethod
    def translate(self, text, source_lang, target_lang):
        """Translate text from source_lang to target_lang.

        Returns the translated text as a string.
        """

    def list_languages(self):
        """Return a list of supported language codes."""
        return []

    def cleanup(self):
        """Release any resources held by this backend."""
