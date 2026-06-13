"""Shared test fixtures and skip markers."""

import os
import unittest


def crispasr_available():
    """Check if the crispasr binary is discoverable."""
    from utils.crispasr_utils import find_crispasr

    return find_crispasr() is not None


def pyqt6_available():
    """Check if PyQt6 is importable."""
    try:
        import PyQt6  # noqa: F401

        return True
    except ImportError:
        return False


def diarization_importable():
    """Check if the diarization backend (and pyannote.audio) can be imported.

    This goes further than ``importlib.util.find_spec`` — it actually imports
    the package so torchaudio/pyannote compatibility shims are exercised. This
    is the import that issue #12 reported as broken.
    """
    try:
        import backends.diarization  # noqa: F401

        return True
    except Exception:
        return False


def hf_token_available():
    """Check if a Hugging Face token is configured (required for live diarization)."""
    return bool(os.environ.get("HF_TOKEN"))


def live_tests_enabled():
    """Live tests hit the network / download models; opt in with RUN_LIVE_TESTS=1."""
    return os.environ.get("RUN_LIVE_TESTS") == "1"


skip_no_crispasr = unittest.skipUnless(crispasr_available(), "crispasr binary not available")
skip_no_pyqt6 = unittest.skipUnless(pyqt6_available(), "PyQt6 not installed")
skip_no_diarization = unittest.skipUnless(
    diarization_importable(), "diarization backend (pyannote.audio) not importable"
)
skip_no_live = unittest.skipUnless(
    live_tests_enabled() and hf_token_available(),
    "live test — set RUN_LIVE_TESTS=1 and HF_TOKEN to run",
)
