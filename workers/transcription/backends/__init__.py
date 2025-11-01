# workers/transcription/backends/__init__.py
"""Transcription backends"""
from .base import TranscriptionBackend
from .mlx_backend import MLXBackend
from .faster_whisper_backend import FasterWhisperBatchedBackend, FasterWhisperSequencedBackend
from .transformers_backend import TransformersBackend
from .whisper_cpp_backend import WhisperCppBackend
from .ctranslate2_backend import CTranslate2Backend
from .whisper_jax_backend import WhisperJaxBackend
from .insanely_fast_backend import InsanelyFastBackend
from .openai_whisper_backend import OpenAIWhisperBackend
from .voxtral_backend import VoxtralLocalBackend, VoxtralAPIBackend

def get_backend(backend_name, **kwargs):
    """Get backend instance by name"""
    backends = {
        'mlx-whisper': MLXBackend,
        'faster-batched': FasterWhisperBatchedBackend,
        'faster-sequenced': FasterWhisperSequencedBackend,
        'transformers': TransformersBackend,
        'whisper.cpp': WhisperCppBackend,
        'ctranslate2': CTranslate2Backend,
        'whisper-jax': WhisperJaxBackend,
        'insanely-fast-whisper': InsanelyFastBackend,
        'openai whisper': OpenAIWhisperBackend,
        'voxtral-local': VoxtralLocalBackend,
        'voxtral-api': VoxtralAPIBackend,
    }
    
    backend_class = backends.get(backend_name.lower())
    if not backend_class:
        raise ValueError(f"Unknown backend: {backend_name}")
    
    return backend_class(**kwargs)