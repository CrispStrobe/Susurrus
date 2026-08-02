"""Regressions for the third EU AI Act audit.

Each test here corresponds to a finding, and names the failure it prevents
rather than the code it touches — a compliance test that only pins an
implementation detail stops protecting the obligation the moment the
implementation moves.
"""

import os
import shutil
import tempfile
import unittest
import wave
from unittest import mock

try:
    import numpy as np

    from utils import spread_spectrum as ss

    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover - minimal installs
    np = None
    ss = None
    _HAVE_NUMPY = False

requires_numpy = unittest.skipUnless(_HAVE_NUMPY, "numpy not installed")


# ---------------------------------------------------------------------------
# Finding 1 — the detector called ~12% of unwatermarked audio watermarked, and
# a false positive there suppressed the Art. 50(2) declarative floor.
# ---------------------------------------------------------------------------


@requires_numpy
class TestDetectorFalsePositives(unittest.TestCase):
    """The in-sample detector must not call ordinary audio AI-generated.

    The old statistic counted bare sign agreement across 32 bins and took the
    better of two band placements, so a lucky run of coin flips cleared the
    threshold. Measured false-positive rate was ~12%.
    """

    @staticmethod
    def _signal(kind, seed, seconds=4.0, rate=24000):
        r = np.random.default_rng(seed)
        n = int(seconds * rate)
        if kind == "speech":
            x = np.convolve(r.standard_normal(n), np.ones(24) / 24, mode="same")
            x = x * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * np.arange(n) / rate) ** 2)
        elif kind == "tone":
            t = np.arange(n) / rate
            x = sum(np.sin(2 * np.pi * f * t) for f in r.uniform(200, 4000, 6))
        else:
            x = r.standard_normal(n)
        return (x / max(abs(x).max(), 1e-9) * 0.3).astype(np.float32)

    def test_false_positive_rate_stays_low(self):
        """Over a mixed corpus, well under 2% of clean audio may read as marked.

        Deliberately a rate and not a single-clip assertion: the failure this
        guards against is statistical, and one fixture passing says nothing
        about the distribution.
        """
        scores = [
            ss.detect(self._signal(kind, seed))
            for kind in ("speech", "tone", "noise")
            for seed in range(40)
        ]
        rate = sum(s >= ss.DETECTION_THRESHOLD for s in scores) / len(scores)
        self.assertLess(
            rate,
            0.02,
            f"{rate:.1%} of unwatermarked clips read as watermarked "
            f"(n={len(scores)}); Art. 50(2) marking must not be a coin flip",
        )

    def test_watermarked_audio_still_detected(self):
        """The false-positive fix must not have been bought with recall."""
        scores = [ss.detect(ss.embed(self._signal("speech", seed))) for seed in range(30)]
        rate = sum(s >= ss.DETECTION_THRESHOLD for s in scores) / len(scores)
        self.assertGreater(rate, 0.95, f"only {rate:.0%} of marked clips detected")

    def test_threshold_was_not_quietly_lowered(self):
        """0.65 is the value the measured ~12% false-positive rate came from."""
        self.assertGreaterEqual(ss.DETECTION_THRESHOLD, 0.78)
        self.assertGreater(ss.LEGACY_DETECTION_THRESHOLD, ss.DETECTION_THRESHOLD)


@requires_numpy
class TestResamplingIsNotSurvived(unittest.TestCase):
    """Pin the limitation, so nobody documents robustness we do not have.

    The comb rides on fixed FFT bin indices, so it is tied to its sample rate.
    This is a known and accepted limit — AudioSeal is the answer where
    resampling matters — but COMPLIANCE.md claimed otherwise for a while, and a
    documented limitation with no test is a limitation that gets re-forgotten.
    """

    def test_rate_change_loses_the_mark_but_restoring_the_rate_recovers_it(self):
        source = np.convolve(
            np.random.default_rng(7).standard_normal(24000 * 6), np.ones(24) / 24, mode="same"
        ).astype(np.float32)
        source = (source / abs(source).max() * 0.3).astype(np.float32)
        marked = ss.embed(source)

        def resample(x, factor):
            n_out = int(len(x) * factor)
            return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(
                np.float32
            )

        downsampled = resample(marked, 16000 / 24000)
        self.assertLess(
            ss.detect(downsampled),
            ss.DETECTION_THRESHOLD,
            "if this now passes, the detector became rate-invariant — update "
            "COMPLIANCE.md and utils/spread_spectrum.py rather than this test",
        )

        restored = resample(downsampled, 24000 / 16000)
        self.assertGreater(ss.detect(restored), ss.DETECTION_THRESHOLD)


class TestDeclarativeFloorIsUnconditional(unittest.TestCase):
    """A detector reading must never be able to suppress the Art. 50(2) floor.

    This is the finding that mattered: with the floor gated on "nothing was
    detected", a single false positive from the in-sample detector meant no
    marker was written, enforce_marking passed on the phantom reading, and
    genuinely unmarked audio shipped under a printed "Marked as AI-generated".
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "out.wav")
        with wave.open(self.path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(b"\x00\x01" * 24000)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _phantom_watermark(self):
        """verify_marking reporting a watermark that is not really there."""
        return {"marker": False, "c2pa": False, "watermark": True}

    def test_tts_backend_marks_even_when_a_watermark_is_reported(self):
        from utils.ai_marking import read_ai_marker
        from workers.tts.backends.crispasr_tts_backend import CrispasrTTSBackend

        backend = CrispasrTTSBackend(model_id="auto")
        with mock.patch("utils.provenance.verify_marking", return_value=self._phantom_watermark()):
            result = backend.apply_provenance(self.path)

        self.assertTrue(result["marker"], "declarative floor was skipped")
        self.assertIsNotNone(
            read_ai_marker(self.path),
            "no marker on disk — a false-positive detector reading suppressed "
            "the only layer that was actually verifiable",
        )

    def test_transcription_backend_marks_even_when_a_watermark_is_reported(self):
        from utils.ai_marking import read_ai_marker
        from workers.transcription.backends.crispasr_backend import CrispasrBackend

        backend = CrispasrBackend(model_id="auto", device="cpu")
        with mock.patch("utils.provenance.verify_marking", return_value=self._phantom_watermark()):
            result = backend.apply_provenance(self.path)

        self.assertTrue(result["marker"])
        self.assertIsNotNone(read_ai_marker(self.path))

    def test_opting_out_still_writes_nothing(self):
        """The floor is unconditional, the attestation is still absolute."""
        from utils.ai_marking import read_ai_marker
        from workers.tts.backends.crispasr_tts_backend import CrispasrTTSBackend

        backend = CrispasrTTSBackend(model_id="auto", accept_marking_responsibility=True)
        result = backend.apply_provenance(self.path)
        self.assertTrue(result["opted_out"])
        self.assertIsNone(read_ai_marker(self.path))


# ---------------------------------------------------------------------------
# Finding 2 — --s2s produced synthetic audio with no consent gate and no
# marking check, on the one route whose only check lived in the binary.
# ---------------------------------------------------------------------------


class TestSpeechToSpeechIsGated(unittest.TestCase):
    def _backend(self, **kwargs):
        from workers.transcription.backends.crispasr_backend import CrispasrBackend

        return CrispasrBackend(model_id="auto", device="cpu", **kwargs)

    def test_s2s_without_attestation_is_refused(self):
        with self.assertRaises(PermissionError):
            self._backend(s2s=True, s2s_output="out.wav").require_s2s_consent()

    def test_s2s_requires_consent_even_with_a_stock_target_voice(self):
        """s2s re-voices a real person whatever the target voice is.

        Unlike synthesize(), this must not turn on whether --voice looks like a
        path: a preset target still yields a recording of someone saying
        something in a voice that is not theirs.
        """
        with self.assertRaises(PermissionError):
            self._backend(
                s2s=True, s2s_output="out.wav", tts_voice="af_sarah"
            ).require_s2s_consent()

    def test_s2s_requires_a_named_output_path(self):
        """Audio Susurrus cannot locate is audio it cannot mark or delete."""
        with self.assertRaises(ValueError):
            self._backend(s2s=True, i_have_rights=True).require_s2s_consent()

    def test_attested_s2s_with_an_output_path_is_allowed(self):
        self._backend(
            s2s=True, s2s_output="out.wav", i_have_rights=True
        ).require_s2s_consent()  # must not raise

    def test_no_gate_when_s2s_is_off(self):
        self._backend().require_s2s_consent()  # must not raise
        self.assertIsNone(self._backend()._s2s_output())

    def test_transcribe_marks_the_s2s_output(self):
        """The marking call must be wired into the run, not merely available."""
        import inspect

        from workers.transcription.backends.crispasr_backend import CrispasrBackend

        source = inspect.getsource(CrispasrBackend.transcribe)
        self.assertIn("require_s2s_consent", source)
        self.assertIn("apply_provenance", source)


# ---------------------------------------------------------------------------
# Finding 5 — Art. 4 AI literacy existed only in the GUI.
# ---------------------------------------------------------------------------


class TestAiLiteracyNoticeOnCli(unittest.TestCase):
    def test_about_ai_flag_exists(self):
        import cli

        parser = cli.build_parser() if hasattr(cli, "build_parser") else None
        if parser is None:
            import inspect

            self.assertIn("--about-ai", inspect.getsource(cli))
            return
        args = parser.parse_args(["--about-ai"])
        self.assertTrue(args.about_ai)

    def test_notice_renders_as_plain_text(self):
        from cli import _html_to_text

        text = _html_to_text(
            "<h3>Title</h3><p>Body &amp; more</p><ul><li>One</li><li>Two</li></ul>"
        )
        self.assertNotIn("<", text)
        self.assertIn("Title", text)
        self.assertIn("Body & more", text)
        self.assertIn("- One", text)
        self.assertIn("- Two", text)

    def test_notice_covers_the_art_4_topics(self):
        """Art. 4 is about what operators understand, so pin the content."""
        from cli import _html_to_text
        from utils.i18n import t

        text = _html_to_text(t("msg.ai_notice.body")).lower()
        for topic in ("intended purpose", "limitations", "not validated"):
            self.assertIn(topic, text, f"AI-literacy notice does not cover {topic!r}")


# ---------------------------------------------------------------------------
# Finding 3 — the docs claimed the fallback watermark needed "numpy only".
# ---------------------------------------------------------------------------


class TestDocumentedDependenciesAreReal(unittest.TestCase):
    def test_spread_spectrum_embedding_needs_soundfile(self):
        """Pin the true dependency, since COMPLIANCE.md now states it.

        Not an aspiration to remove soundfile — a check that the document and
        the code agree about what a bare install can actually mark.
        """
        import inspect

        from utils import audio_watermark

        source = inspect.getsource(audio_watermark._embed_spread_spectrum)
        self.assertIn("soundfile", source)

    def test_compliance_doc_does_not_claim_numpy_only(self):
        doc = os.path.join(os.path.dirname(__file__), "..", "..", "COMPLIANCE.md")
        with open(os.path.abspath(doc), encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("Always — numpy only", text)
        self.assertIn("soundfile", text)


if __name__ == "__main__":
    unittest.main()
