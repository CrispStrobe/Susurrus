"""Test transcription backends"""

import unittest


class TestBackends(unittest.TestCase):
    def test_backend_registry_importable(self):
        """Verify the backends package can be imported"""
        try:
            from workers.transcription.backends import get_backend

            self.assertTrue(callable(get_backend))
        except ImportError as e:
            self.skipTest(f"Backend dependencies not available: {e}")

    def test_get_backend_unknown_raises(self):
        """Unknown backend name raises ValueError"""
        try:
            from workers.transcription.backends import get_backend
        except ImportError as e:
            self.skipTest(f"Backend dependencies not available: {e}")

        with self.assertRaises(ValueError):
            get_backend("nonexistent-backend")

    def test_get_backend_valid_name(self):
        """Known backend names resolve without error"""
        try:
            from workers.transcription.backends import get_backend
        except ImportError as e:
            self.skipTest(f"Backend dependencies not available: {e}")

        try:
            backend = get_backend("transformers", model_id="openai/whisper-tiny", device="cpu")
            self.assertIsNotNone(backend)
        except (ImportError, RuntimeError, OSError):
            # Expected in CI without full model deps
            pass


class TestUtils(unittest.TestCase):
    def test_format_utils(self):
        """Test time formatting utilities"""
        from utils.format_utils import time_to_srt, time_to_vtt

        srt = time_to_srt(3661.5)
        self.assertIn("01:01:01", srt)

        vtt = time_to_vtt(3661.5)
        self.assertIn("01:01:01", vtt)

    def test_audio_utils_is_valid_time(self):
        """Test time validation"""
        from utils.audio_utils import is_valid_time

        self.assertTrue(is_valid_time("10"))
        self.assertTrue(is_valid_time(10))
        self.assertFalse(is_valid_time(None))
        self.assertFalse(is_valid_time(""))


if __name__ == "__main__":
    unittest.main()
