"""Shared test fixtures and skip markers."""

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


skip_no_crispasr = unittest.skipUnless(crispasr_available(), "crispasr binary not available")
skip_no_pyqt6 = unittest.skipUnless(pyqt6_available(), "PyQt6 not installed")
