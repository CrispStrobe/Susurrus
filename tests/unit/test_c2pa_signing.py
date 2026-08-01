"""Test C2PA signing utility (graceful fallback when library not installed)."""

import math
import shutil
import struct
import unittest
import wave


def _write_wav(path, frames=8000):
    """Write a small valid WAV fixture and return its path."""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(
            b"".join(struct.pack("<h", int(3000 * math.sin(i / 20))) for i in range(frames))
        )
    return path


def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None


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

    def test_returns_all_expected_keys(self):
        result = self._dummy().apply_provenance("/nonexistent/output.mp3")
        self.assertEqual(
            set(result),
            {"spoken", "watermark", "marker", "c2pa", "opted_out", "unsupported_format"},
        )

    def test_mp3_is_a_supported_container(self):
        """MP3 gets an ID3 marker — it used to be silently left unmarked."""
        import os
        import subprocess
        import tempfile

        from utils.ai_marking import read_ai_marker

        if not _ffmpeg_available():
            self.skipTest("ffmpeg not available to build an MP3 fixture")

        with tempfile.TemporaryDirectory() as d:
            wav = _write_wav(os.path.join(d, "in.wav"))
            mp3 = os.path.join(d, "out.mp3")
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", wav, "-codec:a", "libmp3lame", mp3],
                check=True,
            )
            result = self._dummy().apply_provenance(mp3)
            self.assertTrue(result["marker"])
            self.assertFalse(result["unsupported_format"])
            self.assertIsNotNone(read_ai_marker(mp3))

    def test_unsupported_container_is_reported_not_claimed(self):
        """An unmarkable container must say so rather than report success."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = _write_wav(os.path.join(d, "audio.opus"))
            result = self._dummy(no_c2pa=True).apply_provenance(path)
            self.assertTrue(result["unsupported_format"])
            self.assertFalse(result["marker"])

    def test_crispasr_backend_verifies_rather_than_asserts(self):
        """The binary marks its own output — but that must be *checked*.

        Reporting c2pa/watermark/marker straight from the flags meant a build
        without C2PA support, or an engine that cannot watermark, still
        produced a confident "Marked as AI-generated" over unmarked audio.
        """
        import os
        import tempfile

        from workers.tts.backends.crispasr_tts_backend import CrispasrTTSBackend

        b = CrispasrTTSBackend(model_id="auto")
        with tempfile.TemporaryDirectory() as d:
            # A file the binary did *not* mark: nothing may be claimed except
            # the declarative marker this method applies as a floor.
            path = _write_wav(os.path.join(d, "out.wav"))
            result = b.apply_provenance(path)
            self.assertFalse(result["opted_out"])
            self.assertTrue(result["marker"], "should apply a marker floor")

            from utils.ai_marking import read_ai_marker

            self.assertIsNotNone(read_ai_marker(path))

    def test_crispasr_backend_does_not_double_sign(self):
        """Already-marked output is not marked a second time."""
        import os
        import tempfile

        from utils.ai_marking import embed_ai_marker, read_ai_marker
        from workers.tts.backends.crispasr_tts_backend import CrispasrTTSBackend

        b = CrispasrTTSBackend(model_id="auto")
        with tempfile.TemporaryDirectory() as d:
            path = _write_wav(os.path.join(d, "out.wav"))
            embed_ai_marker(path, model="binary")
            size_before = os.path.getsize(path)

            result = b.apply_provenance(path)
            self.assertTrue(result["marker"])
            self.assertEqual(size_before, os.path.getsize(path))
            self.assertEqual(read_ai_marker(path)["IENG"], "binary")


if __name__ == "__main__":
    unittest.main()
