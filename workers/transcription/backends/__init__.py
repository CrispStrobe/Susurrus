# workers/transcription/backends/__init__.py
"""Transcription backends"""

from .base import TranscriptionBackend
from .crispasr_backend import CrispasrBackend
from .crispasr_ffi_backend import CrispasrFFIBackend
from .ctranslate2_backend import CTranslate2Backend
from .faster_whisper_backend import FasterWhisperBatchedBackend, FasterWhisperSequencedBackend
from .insanely_fast_backend import InsanelyFastBackend
from .mlx_backend import MLXBackend
from .openai_whisper_backend import OpenAIWhisperBackend
from .transformers_backend import TransformersBackend
from .voxtral_backend import VoxtralAPIBackend, VoxtralLocalBackend
from .whisper_cpp_backend import WhisperCppBackend
from .whisper_jax_backend import WhisperJaxBackend

__all__ = [
    "TranscriptionBackend",
    "MLXBackend",
    "FasterWhisperBatchedBackend",
    "FasterWhisperSequencedBackend",
    "TransformersBackend",
    "WhisperCppBackend",
    "CTranslate2Backend",
    "WhisperJaxBackend",
    "InsanelyFastBackend",
    "OpenAIWhisperBackend",
    "VoxtralLocalBackend",
    "VoxtralAPIBackend",
    "CrispasrBackend",
    "CrispasrFFIBackend",
    "get_backend",
]

_BACKEND_CLASSES = {
    "mlx-whisper": MLXBackend,
    "faster-batched": FasterWhisperBatchedBackend,
    "faster-sequenced": FasterWhisperSequencedBackend,
    "transformers": TransformersBackend,
    "whisper.cpp": WhisperCppBackend,
    "ctranslate2": CTranslate2Backend,
    "whisper-jax": WhisperJaxBackend,
    "insanely-fast-whisper": InsanelyFastBackend,
    "openai whisper": OpenAIWhisperBackend,
    "voxtral-local": VoxtralLocalBackend,
    "voxtral-api": VoxtralAPIBackend,
    "crispasr": CrispasrBackend,
    "crispasr-ffi": CrispasrFFIBackend,
}


def get_backend(backend_name, **kwargs):
    """Get backend instance by name.

    Supports ``crispasr:<subbackend>`` notation — e.g.
    ``crispasr:parakeet`` will create a CrispasrBackend with
    ``crispasr_backend="parakeet"``.
    """
    name = backend_name.lower().strip()

    # Handle crispasr-ffi:<subbackend> notation
    if name.startswith("crispasr-ffi:"):
        sub = name.split(":", 1)[1]
        kwargs.setdefault("crispasr_backend", sub)
        name = "crispasr-ffi"

    # Handle crispasr:<subbackend> notation
    elif name.startswith("crispasr:"):
        sub = name.split(":", 1)[1]
        kwargs.setdefault("crispasr_backend", sub)
        name = "crispasr"

    backend_class = _BACKEND_CLASSES.get(name)
    if not backend_class:
        raise ValueError(
            f"Unknown backend: {backend_name}. "
            f"Available: {', '.join(sorted(_BACKEND_CLASSES))}"
        )

    return backend_class(**kwargs)
