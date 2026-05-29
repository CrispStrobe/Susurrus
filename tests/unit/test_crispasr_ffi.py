"""Test CrispASR FFI backend."""

import unittest


class TestFFIBackendRegistration(unittest.TestCase):
    """Test that the FFI backend is registered and discoverable."""

    def test_ffi_in_registry(self):
        from workers.transcription.backends import get_backend
        b = get_backend("crispasr-ffi", model_id="test.gguf", device="cpu")
        self.assertEqual(b.__class__.__name__, "CrispasrFFIBackend")

    def test_ffi_subbackend_notation(self):
        from workers.transcription.backends import get_backend
        b = get_backend("crispasr-ffi:parakeet", model_id="auto", device="cpu")
        self.assertEqual(b.__class__.__name__, "CrispasrFFIBackend")
        self.assertEqual(b.extra_kwargs.get("crispasr_backend"), "parakeet")

    def test_ffi_available_check(self):
        from workers.transcription.backends.crispasr_ffi_backend import _ffi_available
        # Should return True/False without crashing
        result = _ffi_available()
        self.assertIsInstance(result, bool)

    def test_ffi_available_backends_static(self):
        from workers.transcription.backends.crispasr_ffi_backend import CrispasrFFIBackend
        backends = CrispasrFFIBackend.available_backends()
        self.assertIsInstance(backends, list)

    def test_ffi_kwargs_propagated(self):
        from workers.transcription.backends.crispasr_ffi_backend import CrispasrFFIBackend
        b = CrispasrFFIBackend(
            model_id="test.gguf", device="cpu",
            temperature=0.5, beam_size=5, vad=True,
            grammar="root ::= 'yes' | 'no'",
        )
        self.assertEqual(b.extra_kwargs["temperature"], 0.5)
        self.assertEqual(b.extra_kwargs["beam_size"], 5)
        self.assertEqual(b.extra_kwargs["vad"], True)
        self.assertIn("root", b.extra_kwargs["grammar"])

    def test_cleanup_idempotent(self):
        from workers.transcription.backends.crispasr_ffi_backend import CrispasrFFIBackend
        b = CrispasrFFIBackend(model_id="test.gguf", device="cpu")
        b.cleanup()  # No session was opened
        b.cleanup()  # Should not crash


class TestFFIConfig(unittest.TestCase):
    """Test FFI entries in config."""

    def test_ffi_in_backend_model_map(self):
        from config import BACKEND_MODEL_MAP
        self.assertIn("crispasr-ffi", BACKEND_MODEL_MAP)

    def test_ffi_default_model(self):
        from config import get_default_model_for_backend
        self.assertEqual(get_default_model_for_backend("crispasr-ffi"), "auto")
        self.assertEqual(get_default_model_for_backend("crispasr-ffi:parakeet"), "auto")


if __name__ == "__main__":
    unittest.main()
