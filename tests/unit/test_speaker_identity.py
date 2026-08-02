"""The Art. 50(4) question a preset voice asks.

Susurrus decided the audible disclosure on one thing — was reference audio
supplied — and COMPLIANCE.md stated outright that a stock voice is not a
deepfake. Art. 3(60) turns on the output *resembling* an existing person, not
on how the resemblance was obtained, so a fixed-speaker model trained on one
person's corpus is a deep fake whether or not a WAV was passed.
"""

import unittest
from unittest import mock

from utils import speaker_identity as si


class TestResolution(unittest.TestCase):
    def setUp(self):
        si._reset_warnings_for_tests()

    def test_known_backends_resolve(self):
        self.assertEqual(si.resolve_speaker_identity(backend="piper"), "real_person")
        self.assertEqual(si.resolve_speaker_identity(backend="speecht5"), "real_person")
        self.assertEqual(si.resolve_speaker_identity(backend="kokoro-onnx"), "synthetic")

    def test_unlisted_backend_is_unknown_not_synthetic(self):
        """The costly error is assuming a voice is nobody."""
        self.assertEqual(si.resolve_speaker_identity(backend="something-new"), "unknown")
        self.assertEqual(si.resolve_speaker_identity(backend=None), "unknown")

    def test_override_wins(self):
        self.assertEqual(
            si.resolve_speaker_identity(backend="piper", override="synthetic"), "synthetic"
        )
        self.assertEqual(
            si.resolve_speaker_identity(backend="kokoro-onnx", override="real_person"),
            "real_person",
        )

    def test_case_and_whitespace_tolerated(self):
        self.assertEqual(si.resolve_speaker_identity(override="  Real_Person "), "real_person")

    def test_garbage_override_does_not_silently_disable_disclosure(self):
        """A typo must land on 'unknown', never on 'synthetic'."""
        self.assertEqual(
            si.resolve_speaker_identity(backend="piper", override="rael_person"), "unknown"
        )


class TestDisclosureRule(unittest.TestCase):
    def setUp(self):
        si._reset_warnings_for_tests()

    def test_cloning_always_discloses(self):
        for identity in ("real_person", "synthetic", "unknown"):
            self.assertTrue(si.requires_spoken_disclosure(True, identity, "any"))

    def test_real_person_preset_discloses_without_cloning(self):
        """The gap this whole module exists to close."""
        self.assertTrue(si.requires_spoken_disclosure(False, "real_person", "piper"))

    def test_synthetic_preset_does_not(self):
        self.assertFalse(si.requires_spoken_disclosure(False, "synthetic", "kokoro-onnx"))

    def test_unknown_warns_once_per_backend_and_does_not_force(self):
        with mock.patch.object(si.logger, "warning") as warn:
            self.assertFalse(si.requires_spoken_disclosure(False, "unknown", "edge-tts"))
            self.assertFalse(si.requires_spoken_disclosure(False, "unknown", "edge-tts"))
            self.assertFalse(si.requires_spoken_disclosure(False, "unknown", "melotts"))
        self.assertEqual(warn.call_count, 2, "expected one warning per backend, not per call")

    def test_the_warning_says_what_to_do(self):
        with mock.patch.object(si.logger, "warning") as warn:
            si.requires_spoken_disclosure(False, "unknown", "edge-tts")
        message = warn.call_args[0][0] % warn.call_args[0][1:]
        self.assertIn("--speaker-identity", message)
        self.assertIn("Art. 50(4)", message)


class TestWiredIntoProvenance(unittest.TestCase):
    """The rule has to reach the pipeline, not just exist beside it."""

    def setUp(self):
        import os
        import tempfile
        import wave

        si._reset_warnings_for_tests()
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "out.wav")
        with wave.open(self.path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(b"\x00\x01" * 24000)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, **kwargs):
        from utils.provenance import apply_provenance

        return apply_provenance(self.path, **kwargs)

    class _SpeakingBackend:
        """A backend that can actually produce the disclosure audio."""

        def __init__(self, path):
            self._path = path

        def synthesize(self, text, output_path, voice=None):
            import shutil

            shutil.copy(self._path, output_path)
            return output_path

    def test_real_person_preset_owes_a_disclosure_and_gets_one(self):
        result = self._run(
            speaker_backend="piper",
            backend=self._SpeakingBackend(self.path),
            is_cloning=False,
        )
        self.assertEqual(result["speaker_identity"], "real_person")
        self.assertTrue(
            result["spoken_required"],
            "a real-person preset owes an audible disclosure even without cloning",
        )
        self.assertTrue(result["spoken"], "the disclosure was not prepended")

    def test_synthetic_preset_owes_nothing(self):
        result = self._run(speaker_backend="kokoro-onnx", is_cloning=False)
        self.assertEqual(result["speaker_identity"], "synthetic")
        self.assertFalse(result["spoken_required"])
        self.assertFalse(result["spoken"])

    def test_override_reaches_the_pipeline(self):
        result = self._run(
            speaker_backend="kokoro-onnx",
            backend=self._SpeakingBackend(self.path),
            options={"speaker_identity": "real_person"},
            is_cloning=False,
        )
        self.assertTrue(result["spoken_required"])
        self.assertTrue(result["spoken"])

    def test_an_undeliverable_disclosure_refuses_and_deletes(self):
        """The new case joins the existing fail-closed gate, not a report.

        A real-person preset whose disclosure cannot be produced is refused for
        the same reason a cloned voice is: the listener hears no metadata.
        """
        import os

        from utils.provenance import ProvenanceError

        with self.assertRaises(ProvenanceError) as caught:
            self._run(speaker_backend="piper", is_cloning=False)  # no backend to speak it

        self.assertIn("Art. 50(4)", str(caught.exception))
        self.assertFalse(os.path.exists(self.path), "undisclosed audio was left on disk")

    def test_cli_exposes_the_override(self):
        import inspect

        import cli

        self.assertIn("--speaker-identity", inspect.getsource(cli))

    def test_gui_thread_forwards_it(self):
        import os

        path = os.path.join(os.path.dirname(__file__), "..", "..", "workers", "tts_thread.py")
        with open(os.path.abspath(path), encoding="utf-8") as f:
            source = f.read()
        self.assertIn("speaker_identity", source)
        self.assertIn("tts_backend_name", source)


if __name__ == "__main__":
    unittest.main()
