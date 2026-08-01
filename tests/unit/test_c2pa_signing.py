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


class TestPemResolution(unittest.TestCase):
    """The CLI and GUI supply cert/key *paths*; c2pa-audio wants PEM *text*."""

    def test_inline_pem_passed_through(self):
        from utils.c2pa_signing import _resolve_pem

        pem = "-----BEGIN CERTIFICATE-----\nabc\n-----END CERTIFICATE-----"
        self.assertEqual(_resolve_pem(pem), pem)

    def test_path_is_read_from_disk(self):
        import os
        import tempfile

        from utils.c2pa_signing import _resolve_pem

        pem = "-----BEGIN CERTIFICATE-----\nxyz\n-----END CERTIFICATE-----"
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cert.pem")
            with open(path, "w", encoding="utf-8") as f:
                f.write(pem)
            self.assertEqual(_resolve_pem(path), pem)

    def test_none_and_unreadable_path(self):
        from utils.c2pa_signing import _resolve_pem

        self.assertIsNone(_resolve_pem(None))
        self.assertIsNone(_resolve_pem(""))
        self.assertIsNone(_resolve_pem("/nonexistent/cert.pem"))


class TestTTSBaseProvenance(unittest.TestCase):
    """TTSBackend.apply_provenance degrades gracefully.

    Note: whether it is *called* is covered by test_provenance_wiring.py —
    asserting the method merely exists is what let the v2.10.0 gap ship.
    """

    def _dummy(self, **kwargs):
        from workers.tts.backends.base import TTSBackend

        class DummyTTS(TTSBackend):
            def synthesize(self, text, output_path, voice=None):
                return output_path

        return DummyTTS(**kwargs)

    def test_nonexistent_file_does_not_crash(self):
        result = self._dummy().apply_provenance("/nonexistent/output.wav")
        self.assertFalse(result["c2pa"])
        self.assertFalse(result["marker"])

    def test_skips_non_wav(self):
        result = self._dummy().apply_provenance("output.mp3")
        self.assertFalse(any(result.values()))

    def test_returns_all_expected_keys(self):
        result = self._dummy().apply_provenance("output.mp3")
        self.assertEqual(set(result), {"spoken", "watermark", "marker", "c2pa", "opted_out"})

    def test_crispasr_backend_does_not_double_sign(self):
        """The binary signs its own output; re-signing would stack manifests."""
        from workers.tts.backends.crispasr_tts_backend import CrispasrTTSBackend

        b = CrispasrTTSBackend(model_id="auto")
        result = b.apply_provenance("out.wav")
        self.assertTrue(result["c2pa"])
        self.assertTrue(result["marker"])
        self.assertTrue(result["watermark"])
        self.assertFalse(result["opted_out"])


if __name__ == "__main__":
    unittest.main()
