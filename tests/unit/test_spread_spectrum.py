"""Tests for the dependency-free spread-spectrum watermark.

The point of this layer is that it works when nothing optional is installed,
so these tests deliberately avoid torch, audioseal and any network.
"""

import os
import unittest

import numpy as np

from utils import spread_spectrum as ss


def _speech_like(seconds=4.0, sample_rate=24000):
    """A tonal signal with speech-like structure in the watermark band."""
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False, dtype=np.float32)
    return (
        0.30 * np.sin(2 * np.pi * 180 * t)
        + 0.12 * np.sin(2 * np.pi * 420 * t)
        + 0.05 * np.sin(2 * np.pi * 1800 * t)
    ).astype(np.float32)


class TestCrossProjectConstants(unittest.TestCase):
    """These values are an interop contract, not free parameters.

    CrispASR, CrispTTS and Susurrus must agree, or none can detect the
    others' watermarks. CrispTTS and CrispASR diverged on the comb placement
    once already, and neither could read the other until it was fixed.
    """

    def test_key_and_geometry_match_crispasr(self):
        self.assertEqual(ss.WATERMARK_KEY, 0x437269737041535F)
        self.assertEqual(ss.WATERMARK_NBINS, 32)
        self.assertEqual(ss.FFT_SIZE, 1024)
        self.assertEqual(ss.HOP, 512)

    def test_default_band_is_the_speech_band(self):
        lo, hi, alpha = ss.wm_params(1024)
        self.assertEqual((lo, hi), (1024 // 16, 1024 // 5))
        self.assertAlmostEqual(alpha, 0.05)

    def test_legacy_band_available_for_older_audio(self):
        lo, hi, alpha = ss.wm_params(1024, legacy=True)
        self.assertEqual((lo, hi), (1024 // 16, 1024 // 2 - 1))
        self.assertAlmostEqual(alpha, 0.08)

    def test_legacy_env_selects_the_old_band(self):
        previous = os.environ.pop("CRISPASR_WATERMARK_LEGACY", None)
        os.environ["CRISPASR_WATERMARK_LEGACY"] = "1"
        try:
            self.assertEqual(ss.wm_params(1024)[1], 1024 // 2 - 1)
        finally:
            os.environ.pop("CRISPASR_WATERMARK_LEGACY", None)
            if previous is not None:
                os.environ["CRISPASR_WATERMARK_LEGACY"] = previous

    def test_prng_is_deterministic(self):
        a, b = ss._Prng(42), ss._Prng(42)
        self.assertEqual([a.next() for _ in range(50)], [b.next() for _ in range(50)])

    def test_bin_pattern_is_deterministic_and_in_band(self):
        lo, hi, _ = ss.wm_params(1024)
        bins = ss.generate_bin_pattern(ss.WATERMARK_KEY, 1024, 32, lo, hi)
        self.assertEqual(bins, ss.generate_bin_pattern(ss.WATERMARK_KEY, 1024, 32, lo, hi))
        self.assertEqual(len(bins), 32)
        for idx, sign in bins:
            self.assertTrue(lo <= idx < hi)
            self.assertIn(sign, (-1, 1))


class TestEmbedDetect(unittest.TestCase):
    def test_roundtrip_is_detected(self):
        marked = ss.embed(_speech_like())
        self.assertGreater(ss.detect(marked), ss.DETECTION_THRESHOLD)

    def test_unwatermarked_audio_is_not_detected(self):
        self.assertLess(ss.detect(_speech_like()), ss.DETECTION_THRESHOLD)

    def test_silence_is_not_detected(self):
        self.assertLess(ss.detect(np.zeros(48000, dtype=np.float32)), ss.DETECTION_THRESHOLD)

    def test_legacy_marked_audio_still_detected(self):
        """Detection sweeps both bands, so older audio keeps verifying."""
        previous = os.environ.pop("CRISPASR_WATERMARK_LEGACY", None)
        os.environ["CRISPASR_WATERMARK_LEGACY"] = "1"
        try:
            marked = ss.embed(_speech_like())
        finally:
            os.environ.pop("CRISPASR_WATERMARK_LEGACY", None)
            if previous is not None:
                os.environ["CRISPASR_WATERMARK_LEGACY"] = previous
        self.assertGreater(ss.detect(marked), ss.DETECTION_THRESHOLD)

    def _snr(self, pcm, **kwargs):
        noise = ss.embed(pcm, **kwargs) - pcm
        return float(10 * np.log10(np.mean(pcm**2) / np.mean(noise**2)))

    def test_embed_is_quiet_at_low_strength(self):
        """Absolute SNR is signal-dependent, so pin it at a fixed alpha.

        The comb's weight is relative to the mean bin magnitude, so a peaky
        synthetic tone stack and a real broadband recording give very
        different numbers for the same alpha — measured 39.5 dB on 20 s of
        real speech versus ~15 dB on the tone stack below. Asserting a
        headline figure here would be testing the fixture, not the code.
        """
        self.assertGreater(self._snr(_speech_like(), alpha=0.005), 20.0)

    def test_snr_decreases_monotonically_with_alpha(self):
        pcm = _speech_like()
        snrs = [self._snr(pcm, alpha=a) for a in (0.005, 0.02, 0.05, 0.08)]
        self.assertEqual(
            snrs, sorted(snrs, reverse=True), f"louder alpha must mean lower SNR, got {snrs}"
        )

    def test_survives_resampling(self):
        """A plain resample used to destroy the mark on the old wide band."""
        marked = ss.embed(_speech_like(seconds=6.0))
        # Crude decimate/interpolate, enough to break a fragile watermark.
        down = np.interp(np.arange(0, len(marked), 1.5), np.arange(len(marked)), marked)
        up = np.interp(
            np.linspace(0, len(down) - 1, len(marked)), np.arange(len(down)), down
        ).astype(np.float32)
        self.assertGreater(ss.detect(up), ss.DETECTION_THRESHOLD)

    def test_short_audio_is_a_noop(self):
        pcm = np.zeros(500, dtype=np.float32)
        np.testing.assert_array_equal(ss.embed(pcm), pcm)
        self.assertEqual(ss.detect(pcm), 0.0)

    def test_zero_alpha_leaves_samples_untouched(self):
        pcm = _speech_like()
        np.testing.assert_allclose(ss.embed(pcm, alpha=0.0), pcm, atol=1e-6)

    def test_alpha_none_matches_negative(self):
        pcm = _speech_like()
        np.testing.assert_allclose(ss.embed(pcm), ss.embed(pcm, alpha=-1.0), atol=1e-6)


class TestFallbackWiring(unittest.TestCase):
    """audio_watermark must reach this layer when AudioSeal is absent."""

    def setUp(self):
        import tempfile

        import soundfile as sf

        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "t.wav")
        sf.write(self.path, _speech_like(), 24000, subtype="PCM_16")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_embed_falls_back_when_audioseal_missing(self):
        from unittest import mock

        from utils import audio_watermark

        with mock.patch.object(audio_watermark, "_load", return_value=None):
            self.assertTrue(audio_watermark.embed_watermark(self.path))
            result = audio_watermark.detect_watermark(self.path)
        self.assertIsNotNone(result, "detection must not return None without AudioSeal")
        self.assertTrue(result["watermarked"])
        self.assertEqual(result["backend"], "spread-spectrum")

    def test_detect_reports_unmarked_audio_honestly(self):
        from unittest import mock

        from utils import audio_watermark

        with mock.patch.object(audio_watermark, "_load", return_value=None):
            result = audio_watermark.detect_watermark(self.path)
        self.assertFalse(result["watermarked"])


if __name__ == "__main__":
    unittest.main()
