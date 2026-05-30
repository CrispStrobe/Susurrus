# workers/tts/backends/__init__.py
"""TTS backend registry."""

from .base import TTSBackend

__all__ = ["TTSBackend", "get_tts_backend"]


def get_tts_backend(backend_name, **kwargs):
    """Get a TTS backend instance by name.

    Supports ``crispasr:<tts-backend>`` notation for CrispASR-based TTS.
    """
    name = backend_name.lower().strip()

    # Lazy imports to avoid pulling in heavy dependencies at module load
    _backends = {}

    def _load(key, mod, cls):
        if key not in _backends:
            import importlib

            m = importlib.import_module(f".{mod}", package=__package__)
            _backends[key] = getattr(m, cls)
        return _backends[key]

    # CrispASR TTS (binary-based)
    if name.startswith("crispasr"):
        cls = _load("crispasr-tts", "crispasr_tts_backend", "CrispasrTTSBackend")
        if ":" in name:
            kwargs.setdefault("crispasr_backend", name.split(":", 1)[1])
        return cls(**kwargs)

    registry = {
        "edge-tts": ("edge_tts_backend", "EdgeTTSBackend"),
        "piper": ("piper_tts_backend", "PiperTTSBackend"),
        "kokoro-onnx": ("kokoro_onnx_tts_backend", "KokoroOnnxTTSBackend"),
        "chatterbox": ("chatterbox_tts_backend", "ChatterboxTTSBackend"),
        "speecht5": ("speecht5_tts_backend", "SpeechT5TTSBackend"),
    }

    if name not in registry:
        raise ValueError(
            f"Unknown TTS backend: {backend_name}. "
            f"Available: {', '.join(sorted(registry))} + crispasr:<engine>"
        )

    mod, cls_name = registry[name]
    cls = _load(name, mod, cls_name)
    return cls(**kwargs)
