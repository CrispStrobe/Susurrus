"""Shared test fixtures and skip markers.

Every availability probe here is evaluated *lazily*, at the point the
decorated test runs, never at import time. ``diarization_importable()``
pulls in pyannote.audio and torch; doing that while pytest is still
collecting made every run — including ``pytest tests/unit/test_ai_marking.py``
— pay a multi-minute import before a single test executed.
"""

import functools
import os
import unittest


def lazy_skip_unless(predicate, reason):
    """``unittest.skipUnless`` that evaluates *predicate* at call time.

    ``unittest.skipUnless`` takes a boolean, so writing
    ``skipUnless(expensive_probe(), ...)`` at module level runs the probe
    during collection. This defers it to the test itself, and caches the
    result so a probe runs at most once per session.
    """
    cached = functools.lru_cache(maxsize=1)(predicate)

    def decorator(test_item):
        if isinstance(test_item, type):
            # Class decorator: defer to each test method.
            for name in list(vars(test_item)):
                if name.startswith("test"):
                    attr = getattr(test_item, name)
                    if callable(attr):
                        setattr(test_item, name, decorator(attr))
            return test_item

        @functools.wraps(test_item)
        def wrapper(*args, **kwargs):
            if not cached():
                raise unittest.SkipTest(reason)
            return test_item(*args, **kwargs)

        return wrapper

    return decorator


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
    """Check if the full diarization API (pyannote.audio etc.) is available.

    The package imports even without the heavy optional deps (it degrades
    gracefully), so we check that ``DiarizationManager`` actually resolved —
    that is the API issue #12 reported as missing/broken.
    """
    try:
        import backends.diarization as diarization

        return diarization.DiarizationManager is not None
    except Exception:
        return False


def hf_token_available():
    """Check if a Hugging Face token is configured (required for live diarization)."""
    return bool(os.environ.get("HF_TOKEN"))


def live_tests_enabled():
    """Live tests hit the network / download models; opt in with RUN_LIVE_TESTS=1."""
    return os.environ.get("RUN_LIVE_TESTS") == "1"


skip_no_crispasr = lazy_skip_unless(crispasr_available, "crispasr binary not available")
skip_no_pyqt6 = lazy_skip_unless(pyqt6_available, "PyQt6 not installed")
skip_no_diarization = lazy_skip_unless(
    diarization_importable, "diarization backend (pyannote.audio) not importable"
)
skip_no_live = lazy_skip_unless(
    lambda: live_tests_enabled() and hf_token_available(),
    "live test — set RUN_LIVE_TESTS=1 and HF_TOKEN to run",
)
