# workers/translation/backends/__init__.py
"""Translation backend registry."""

from .base import TranslationBackend

__all__ = ["TranslationBackend", "get_translation_backend"]


def get_translation_backend(backend_name, **kwargs):
    """Get a translation backend instance by name."""
    name = backend_name.lower().strip()

    if name.startswith("crispasr"):
        from .crispasr_translation_backend import CrispasrTranslationBackend

        if ":" in name:
            kwargs.setdefault("crispasr_backend", name.split(":", 1)[1])
        return CrispasrTranslationBackend(**kwargs)

    raise ValueError(
        f"Unknown translation backend: {backend_name}. "
        f"Available: crispasr, crispasr:m2m100, crispasr:madlad, crispasr:gemma4-e2b"
    )
