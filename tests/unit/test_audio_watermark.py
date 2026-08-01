"""Test the AudioSeal neural watermark layer.

``audioseal`` is an optional dependency. These tests cover the wiring and the
degradation path unconditionally; the live embed/detect round-trip runs only
when the library is actually installed.
"""

import os
import tempfile
import unittest
import wave
from unittest import mock

from utils import audio_watermark
from workers.tts.backends.base import TTSBackend


def _audioseal_installed():
    try:
        import audioseal  # noqa: F401

        return True
    except ImportError:
        return False


def _write_wav(path, frames=16000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x01\x00" * frames)


class Dummy(TTSBackend):
    def synthesize(self, text, output_path, voice=None):
        return output_path


class TestGracefulDegradation(unittest.TestCase):
    """Absent audioseal, watermarking is a no-op — never a crash."""

    def setUp(self):
        audio_watermark._reset_cache_for_tests()
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "out.wav")
        _write_wav(self.path)

    def tearDown(self):
        audio_watermark._reset_cache_for_tests()
        self._dir.cleanup()

    def test_embed_returns_false_without_library(self):
        with mock.patch.dict("sys.modules", {"audioseal": None}):
            self.assertFalse(audio_watermark.embed_watermark(self.path))

    def test_detect_returns_none_without_library(self):
        with mock.patch.dict("sys.modules", {"audioseal": None}):
            self.assertIsNone(audio_watermark.detect_watermark(self.path))

    def test_is_available_reports_honestly(self):
        self.assertIsInstance(audio_watermark.is_available(), bool)

    def test_audio_untouched_when_unavailable(self):
        before = open(self.path, "rb").read()
        with mock.patch.dict("sys.modules", {"audioseal": None}):
            audio_watermark.embed_watermark(self.path)
        self.assertEqual(open(self.path, "rb").read(), before)

    def test_model_load_failure_degrades(self):
        """A failed model download must not break synthesis."""
        fake = mock.MagicMock()
        fake.AudioSeal.load_generator.side_effect = RuntimeError("no network")
        with mock.patch.dict("sys.modules", {"audioseal": fake}):
            self.assertFalse(audio_watermark.embed_watermark(self.path))

    def test_load_failure_is_not_retried(self):
        """Repeated failures shouldn't re-attempt a download per synthesis."""
        fake = mock.MagicMock()
        fake.AudioSeal.load_generator.side_effect = RuntimeError("no network")
        with mock.patch.dict("sys.modules", {"audioseal": fake}):
            audio_watermark.embed_watermark(self.path)
            audio_watermark.embed_watermark(self.path)
        self.assertEqual(fake.AudioSeal.load_generator.call_count, 1)


class TestProvenanceLayerOrdering(unittest.TestCase):
    """Watermarking mutates samples; C2PA hashes them. Order is load-bearing."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "out.wav")
        _write_wav(self.path)

    def tearDown(self):
        self._dir.cleanup()

    def test_watermark_runs_before_c2pa_signing(self):
        calls = []

        def fake_watermark(path):
            calls.append("watermark")
            return True

        def fake_sign(path, cert_pem=None, key_pem=None):
            calls.append("c2pa")
            return True

        with mock.patch("utils.audio_watermark.embed_watermark", fake_watermark):
            with mock.patch("utils.c2pa_signing.sign_wav_file", fake_sign):
                Dummy().apply_provenance(self.path)

        self.assertEqual(calls, ["watermark", "c2pa"])

    def test_marker_runs_before_c2pa_signing(self):
        calls = []
        with mock.patch(
            "utils.ai_marking.embed_wav_ai_marker",
            side_effect=lambda p, model=None: calls.append("marker") or True,
        ):
            with mock.patch(
                "utils.c2pa_signing.sign_wav_file",
                side_effect=lambda p, cert_pem=None, key_pem=None: calls.append("c2pa") or True,
            ):
                Dummy().apply_provenance(self.path)
        self.assertEqual(calls, ["marker", "c2pa"])

    def test_no_watermark_skips_the_neural_layer_only(self):
        with mock.patch("utils.audio_watermark.embed_watermark") as wm:
            result = Dummy(no_watermark=True).apply_provenance(self.path)
        wm.assert_not_called()
        self.assertFalse(result["watermark"])
        self.assertTrue(result["marker"], "declarative marking must still apply")

    def test_accept_responsibility_skips_the_neural_layer(self):
        with mock.patch("utils.audio_watermark.embed_watermark") as wm:
            result = Dummy(accept_marking_responsibility=True).apply_provenance(self.path)
        wm.assert_not_called()
        self.assertTrue(result["opted_out"])

    def test_watermark_result_is_reported(self):
        with mock.patch("utils.audio_watermark.embed_watermark", return_value=True):
            self.assertTrue(Dummy().apply_provenance(self.path)["watermark"])

    def test_result_exposes_all_four_layers(self):
        result = Dummy().apply_provenance(self.path)
        self.assertEqual(set(result), {"spoken", "watermark", "marker", "c2pa", "opted_out"})


class TestDetectWatermarkVerb(unittest.TestCase):
    """`--detect-watermark` must never turn "cannot check" into "clean"."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "out.wav")
        _write_wav(self.path)

    def tearDown(self):
        self._dir.cleanup()

    def _run(self):
        import argparse

        import cli

        args = argparse.Namespace(detect_watermark=self.path)
        return cli._run_detect_watermark(args)

    def test_inconclusive_exits_two_not_one(self):
        """Exit 1 means 'not marked'; exit 2 means 'could not check'."""
        with mock.patch("cli._detect_watermark_via_binary", return_value=None):
            with mock.patch("utils.audio_watermark.detect_watermark", return_value=None):
                self.assertEqual(self._run(), 2)

    def test_declarative_marker_alone_is_a_positive(self):
        from utils.ai_marking import embed_wav_ai_marker

        embed_wav_ai_marker(self.path, model="piper")
        with mock.patch("cli._detect_watermark_via_binary", return_value=None):
            with mock.patch("utils.audio_watermark.detect_watermark", return_value=None):
                self.assertEqual(self._run(), 0)

    def test_neural_detection_is_a_positive(self):
        detection = {"watermarked": True, "confidence": 0.99, "threshold": 0.5}
        with mock.patch("cli._detect_watermark_via_binary", return_value=None):
            with mock.patch("utils.audio_watermark.detect_watermark", return_value=detection):
                self.assertEqual(self._run(), 0)

    def test_clean_audio_with_a_working_detector_exits_one(self):
        detection = {"watermarked": False, "confidence": 0.01, "threshold": 0.5}
        with mock.patch("cli._detect_watermark_via_binary", return_value=None):
            with mock.patch("utils.audio_watermark.detect_watermark", return_value=detection):
                self.assertEqual(self._run(), 1)

    def test_binary_result_wins_when_it_succeeds(self):
        with mock.patch("cli._detect_watermark_via_binary", return_value=0) as binary:
            self.assertEqual(self._run(), 0)
        binary.assert_called_once()

    def test_falls_back_when_binary_crashes(self):
        """A stale or broken binary must not leave the question unanswered."""
        import subprocess

        import cli

        crashed = subprocess.CompletedProcess([], returncode=-6, stdout="", stderr="dyld: boom")
        with mock.patch.object(cli, "_build_crispasr_kwargs", return_value={}):
            with mock.patch("utils.crispasr_utils.find_crispasr", return_value="/bin/crispasr"):
                with mock.patch("subprocess.run", return_value=crashed):
                    result = cli._detect_watermark_via_binary(mock.Mock(), self.path)
        self.assertIsNone(result, "a crashed binary must fall through to Python")

    def test_binary_verdict_is_kept_even_on_nonzero_exit(self):
        """Exit 1 with output is a real 'not watermarked' answer, not a crash."""
        import subprocess

        import cli

        answered = subprocess.CompletedProcess([], returncode=1, stdout="no watermark", stderr="")
        with mock.patch.object(cli, "_build_crispasr_kwargs", return_value={}):
            with mock.patch("utils.crispasr_utils.find_crispasr", return_value="/bin/crispasr"):
                with mock.patch("subprocess.run", return_value=answered):
                    result = cli._detect_watermark_via_binary(mock.Mock(), self.path)
        self.assertEqual(result, 1)

    def test_no_binary_falls_through(self):
        import cli

        with mock.patch("utils.crispasr_utils.find_crispasr", return_value=None):
            self.assertIsNone(cli._detect_watermark_via_binary(mock.Mock(), self.path))


@unittest.skipUnless(_audioseal_installed(), "audioseal not installed")
class TestLiveRoundTrip(unittest.TestCase):
    """Live embed → detect. Skipped unless audioseal is actually present."""

    def setUp(self):
        audio_watermark._reset_cache_for_tests()
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "out.wav")
        _write_wav(self.path, frames=16000)

    def tearDown(self):
        audio_watermark._reset_cache_for_tests()
        self._dir.cleanup()

    def test_embed_then_detect(self):
        self.assertTrue(audio_watermark.embed_watermark(self.path))
        result = audio_watermark.detect_watermark(self.path)
        self.assertIsNotNone(result)
        self.assertTrue(result["watermarked"])

    def test_clean_audio_is_not_detected(self):
        clean = os.path.join(self._dir.name, "clean.wav")
        _write_wav(clean, frames=16000)
        result = audio_watermark.detect_watermark(clean)
        self.assertIsNotNone(result)
        self.assertFalse(result["watermarked"])


if __name__ == "__main__":
    unittest.main()
