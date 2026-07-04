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
            model_id="test.gguf",
            device="cpu",
            temperature=0.5,
            beam_size=5,
            vad=True,
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


class TestFFIS2SMethod(unittest.TestCase):
    """Test that the FFI backend exposes S2S and error methods."""

    def test_speech_to_speech_method_exists(self):
        from workers.transcription.backends.crispasr_ffi_backend import CrispasrFFIBackend

        self.assertTrue(hasattr(CrispasrFFIBackend, "speech_to_speech"))

    def test_last_synth_error_method_exists(self):
        from workers.transcription.backends.crispasr_ffi_backend import CrispasrFFIBackend

        self.assertTrue(hasattr(CrispasrFFIBackend, "last_synth_error"))

    def test_last_synth_error_no_session(self):
        from workers.transcription.backends.crispasr_ffi_backend import CrispasrFFIBackend

        b = CrispasrFFIBackend(model_id="test.gguf", device="cpu")
        # No session opened — should return None, not crash
        self.assertIsNone(b.last_synth_error())


class TestFFI087SessionSetters(unittest.TestCase):
    """Test that 0.8.7 FFI session setters are wired in _apply_session_state."""

    def test_top_k_wired(self):
        from workers.transcription.backends.crispasr_ffi_backend import CrispasrFFIBackend

        b = CrispasrFFIBackend(model_id="test.gguf", device="cpu", top_k=50)
        self.assertEqual(b.extra_kwargs["top_k"], 50)

    def test_do_sample_wired(self):
        from workers.transcription.backends.crispasr_ffi_backend import CrispasrFFIBackend

        b = CrispasrFFIBackend(model_id="test.gguf", device="cpu", do_sample=True)
        self.assertTrue(b.extra_kwargs["do_sample"])

    def test_tts_num_candidates_wired(self):
        from workers.transcription.backends.crispasr_ffi_backend import CrispasrFFIBackend

        b = CrispasrFFIBackend(model_id="test.gguf", device="cpu", tts_num_candidates=8)
        self.assertEqual(b.extra_kwargs["tts_num_candidates"], 8)

    def test_tts_noise_temp_wired(self):
        from workers.transcription.backends.crispasr_ffi_backend import CrispasrFFIBackend

        b = CrispasrFFIBackend(model_id="test.gguf", device="cpu", tts_noise_temp=0.7)
        self.assertAlmostEqual(b.extra_kwargs["tts_noise_temp"], 0.7)


class TestFFICompanionModels(unittest.TestCase):
    """Test companion model resolution for new backends."""

    def test_mini_omni2_companion(self):
        from utils.crispasr_utils import resolve_companions

        companions = resolve_companions("mini-omni2")
        self.assertIn("codec", companions)
        self.assertEqual(companions["codec"]["name"], "snac-24khz")

    def test_chatterbox_companion(self):
        from utils.crispasr_utils import resolve_companions

        companions = resolve_companions("chatterbox")
        self.assertIn("codec", companions)
        self.assertEqual(companions["codec"]["name"], "chatterbox-s3gen")

    def test_orpheus_companion(self):
        from utils.crispasr_utils import resolve_companions

        companions = resolve_companions("orpheus")
        self.assertIn("codec", companions)

    def test_dots_tts_companion(self):
        from utils.crispasr_utils import resolve_companions

        companions = resolve_companions("dots-tts")
        self.assertIn("speaker", companions)
        self.assertEqual(companions["speaker"]["name"], "dots-tts-soar-spk")

    def test_tada_companions(self):
        from utils.crispasr_utils import resolve_companions

        companions = resolve_companions("tada")
        self.assertIn("encoder", companions)
        self.assertIn("aligner", companions)
        self.assertEqual(companions["encoder"]["name"], "tada-encoder")
        self.assertEqual(companions["aligner"]["name"], "tada-aligner")

    def test_unknown_backend_no_companion(self):
        from utils.crispasr_utils import resolve_companions

        companions = resolve_companions("nonexistent-backend")
        self.assertEqual(companions, {})


if __name__ == "__main__":
    unittest.main()
