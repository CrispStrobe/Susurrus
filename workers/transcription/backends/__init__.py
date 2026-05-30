# workers/transcription/backends/__init__.py
"""Transcription backends"""

import importlib

from .base import TranscriptionBackend

__all__ = [
    "TranscriptionBackend",
    "get_backend",
]

# Lazy registry: maps backend name to (module_name, class_name)
_BACKEND_REGISTRY = {
    "mlx-whisper": ("mlx_backend", "MLXBackend"),
    "faster-batched": ("faster_whisper_backend", "FasterWhisperBatchedBackend"),
    "faster-sequenced": ("faster_whisper_backend", "FasterWhisperSequencedBackend"),
    "transformers": ("transformers_backend", "TransformersBackend"),
    "whisper.cpp": ("whisper_cpp_backend", "WhisperCppBackend"),
    "ctranslate2": ("ctranslate2_backend", "CTranslate2Backend"),
    "whisper-jax": ("whisper_jax_backend", "WhisperJaxBackend"),
    "insanely-fast-whisper": ("insanely_fast_backend", "InsanelyFastBackend"),
    "openai whisper": ("openai_whisper_backend", "OpenAIWhisperBackend"),
    "voxtral-local": ("voxtral_backend", "VoxtralLocalBackend"),
    "voxtral-api": ("voxtral_backend", "VoxtralAPIBackend"),
    "crispasr": ("crispasr_backend", "CrispasrBackend"),
    "crispasr-ffi": ("crispasr_ffi_backend", "CrispasrFFIBackend"),
}


def _load_backend_class(module_name, class_name):
    """Import a backend class on demand."""
    mod = importlib.import_module(f".{module_name}", __package__)
    return getattr(mod, class_name)


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

    entry = _BACKEND_REGISTRY.get(name)
    if not entry:
        raise ValueError(
            f"Unknown backend: {backend_name}. "
            f"Available: {', '.join(sorted(_BACKEND_REGISTRY))}"
        )

    backend_class = _load_backend_class(*entry)
    return backend_class(**kwargs)
