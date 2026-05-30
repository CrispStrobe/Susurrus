"""Shared test fixtures and skip markers."""

import shutil
import unittest


def crispasr_available():
    """Check if the crispasr binary is on PATH or set via env."""
    import os

    if os.environ.get("CRISPASR_EXECUTABLE"):
        return os.path.isfile(os.environ["CRISPASR_EXECUTABLE"])
    return shutil.which("crispasr") is not None


def pyqt6_available():
    """Check if PyQt6 is importable."""
    try:
        import PyQt6  # noqa: F401

        return True
    except ImportError:
        return False


# For unittest-based tests: decorators
skip_no_crispasr = unittest.skipUnless(crispasr_available(), "crispasr binary not available")
skip_no_pyqt6 = unittest.skipUnless(pyqt6_available(), "PyQt6 not installed")
