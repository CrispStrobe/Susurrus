"""Test C2PA signing utility (graceful fallback when library not installed)."""

import unittest


class TestC2paSigningAvailability(unittest.TestCase):
    """Test that c2pa_signing module imports and degrades gracefully."""

    def test_import(self):
        from utils.c2pa_signing import is_available

        # Should not crash regardless of library availability
        self.assertIsInstance(is_available(), bool)

    def test_sign_returns_false_without_lib(self):
        from utils.c2pa_signing import sign_wav_file

        # Signing a nonexistent file should return False (not crash)
        result = sign_wav_file("/nonexistent/file.wav")
        self.assertFalse(result)

    def test_verify_returns_none_without_lib(self):
        from utils.c2pa_signing import verify_wav_file

        # Verification should return None when lib not available
        result = verify_wav_file("/nonexistent/file.wav")
        # Either None (lib not available) or exception-caught None
        self.assertIsNone(result)


class TestTTSBaseSignOutput(unittest.TestCase):
    """Test that TTSBackend.sign_output exists and degrades gracefully."""

    def test_sign_output_method_exists(self):
        from workers.tts.backends.base import TTSBackend

        self.assertTrue(hasattr(TTSBackend, "sign_output"))

    def test_sign_output_returns_false_for_nonexistent(self):
        """sign_output on a nonexistent file should not crash."""

        from workers.tts.backends.base import TTSBackend

        class DummyTTS(TTSBackend):
            def synthesize(self, text, output_path, voice=None):
                return output_path

        b = DummyTTS()
        result = b.sign_output("/nonexistent/output.wav")
        self.assertFalse(result)

    def test_sign_output_skips_non_wav(self):
        from workers.tts.backends.base import TTSBackend

        class DummyTTS(TTSBackend):
            def synthesize(self, text, output_path, voice=None):
                return output_path

        b = DummyTTS()
        result = b.sign_output("output.mp3")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
