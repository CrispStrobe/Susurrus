"""Regression tests for the Art. 50 gaps found in the v2.11.0 audit.

Each test here corresponds to a route that produced synthetic audio without
marking it, or reported marking that nothing had verified. They are written
against the *behaviour* (does the file end up marked, is the clone refused)
rather than against the presence of a method, because "the method exists" is
the assertion that let the original gap ship.
"""

import math
import os
import struct
import sys
import tempfile
import unittest
import wave
from unittest import mock

# The dependency-free watermark tier is built from numpy + soundfile, and CI
# installs the package with --no-deps. Where they are absent there is no
# fallback to exercise, so these skip rather than fail: a graceful-degradation
# suite that itself fails to degrade gracefully is not much of a signal.
try:
    import numpy  # noqa: F401
    import soundfile  # noqa: F401

    _HAVE_AUDIO_STACK = True
except ImportError:  # pragma: no cover - minimal installs only
    _HAVE_AUDIO_STACK = False

requires_audio_stack = unittest.skipUnless(_HAVE_AUDIO_STACK, "numpy/soundfile not installed")


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _write_wav(path, frames=4000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(
            b"".join(struct.pack("<h", int(3000 * math.sin(i / 20))) for i in range(frames))
        )
    return path


@requires_audio_stack
class TestFFIBackendProvenance(unittest.TestCase):
    """crispasr-ffi synthesizes in-process, so the binary's marking never runs.

    ``cli._run_tts`` routes ``--tts-backend crispasr-ffi`` into the CrispASR
    branch on a ``startswith`` match. The class is a *transcription* backend:
    it wrote raw PCM to a WAV with no marking and no consent gate, then raised
    AttributeError on the marking call — leaving unmarked audio on disk.
    """

    def _backend(self, **kwargs):
        from workers.transcription.backends.crispasr_ffi_backend import CrispasrFFIBackend

        return CrispasrFFIBackend(model_id="auto", device="cpu", **kwargs)

    def test_exposes_apply_provenance(self):
        self.assertTrue(hasattr(self._backend(), "apply_provenance"))

    def test_marks_synthesized_output(self):
        from utils.ai_marking import read_ai_marker

        with tempfile.TemporaryDirectory() as d:
            path = _write_wav(os.path.join(d, "out.wav"))
            result = self._backend().apply_provenance(path)
            self.assertTrue(result["marker"])
            self.assertIsNotNone(read_ai_marker(path))

    def test_cloning_without_consent_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            ref = _write_wav(os.path.join(d, "ref.wav"))
            backend = self._backend(tts_voice=ref)
            with self.assertRaises(PermissionError):
                backend.synthesize("hello", os.path.join(d, "out.wav"))

    def test_cloning_with_consent_is_allowed_through_the_gate(self):
        """Consent must let the call proceed past the gate (session opens)."""
        with tempfile.TemporaryDirectory() as d:
            ref = _write_wav(os.path.join(d, "ref.wav"))
            backend = self._backend(tts_voice=ref, i_have_rights=True)
            with mock.patch.object(backend, "_ensure_session", side_effect=RuntimeError("session")):
                with self.assertRaises(RuntimeError):
                    backend.synthesize("hello", os.path.join(d, "out.wav"))

    def test_preset_voice_is_not_treated_as_cloning(self):
        backend = self._backend(tts_voice="af_sarah")
        self.assertIsNone(backend.resolve_reference_audio())


class TestCrispasrCloneGate(unittest.TestCase):
    """The CrispASR routes relied entirely on the binary to refuse cloning."""

    def _cli_backend(self, **kwargs):
        from workers.transcription.backends.crispasr_backend import CrispasrBackend

        return CrispasrBackend(model_id="auto", device="cpu", **kwargs)

    def test_cli_route_refuses_unconsented_clone(self):
        with tempfile.TemporaryDirectory() as d:
            ref = _write_wav(os.path.join(d, "ref.wav"))
            backend = self._cli_backend(tts_voice=ref)
            with self.assertRaises(PermissionError):
                backend.synthesize("hello", os.path.join(d, "out.wav"))

    def test_gui_route_refuses_unconsented_clone(self):
        from workers.tts.backends.crispasr_tts_backend import CrispasrTTSBackend

        with tempfile.TemporaryDirectory() as d:
            ref = _write_wav(os.path.join(d, "ref.wav"))
            backend = CrispasrTTSBackend(model_id="auto", voice=ref)
            with self.assertRaises(PermissionError):
                backend.synthesize("hello", os.path.join(d, "out.wav"))

    def test_preset_voice_needs_no_attestation(self):
        """A preset name is not reference audio — don't demand consent."""
        backend = self._cli_backend(tts_voice="af_sarah")
        self.assertIsNone(backend.resolve_reference_audio())


class TestCloningDetection(unittest.TestCase):
    """is_cloning() short-circuited on the first truthy candidate."""

    def _dummy(self, **kwargs):
        from workers.tts.backends.base import TTSBackend

        class Dummy(TTSBackend):
            def synthesize(self, text, output_path, voice=None):
                return output_path

        return Dummy(**kwargs)

    def test_voice_name_does_not_mask_reference_audio(self):
        """A selected voice name must not hide a real cloning reference.

        ``voice or kwargs["voice"] or kwargs["reference_audio"]`` returned the
        non-path voice name, whose isfile() is False — so a GUI clone with a
        voice selected reported "not cloning" and skipped the Art. 50(4)
        audible disclosure entirely.
        """
        with tempfile.TemporaryDirectory() as d:
            ref = _write_wav(os.path.join(d, "ref.wav"))
            backend = self._dummy(voice="af_sarah", reference_audio=ref)
            self.assertTrue(backend.is_cloning())
            self.assertEqual(backend.resolve_reference_audio(), ref)

    def test_disclosure_is_not_spoken_in_the_cloned_voice(self):
        """While synthesizing the disclosure, the clone reference is withheld."""
        with tempfile.TemporaryDirectory() as d:
            ref = _write_wav(os.path.join(d, "ref.wav"))
            backend = self._dummy(reference_audio=ref, i_have_rights=True)
            self.assertEqual(backend.resolve_reference_audio(), ref)

            backend._synthesizing_disclosure = True
            self.assertIsNone(
                backend.resolve_reference_audio(),
                "the disclosure must not be spoken by the impersonated voice",
            )


class TestChatterboxConsentRoutes(unittest.TestCase):
    """Every cloning entry point converges on one gate."""

    def _backend(self, **kwargs):
        from workers.tts.backends.chatterbox_tts_backend import ChatterboxTTSBackend

        return ChatterboxTTSBackend(model_id="chatterbox", **kwargs)

    def test_reference_audio_kwarg_route_is_gated(self):
        with tempfile.TemporaryDirectory() as d:
            ref = _write_wav(os.path.join(d, "ref.wav"))
            with self.assertRaises(PermissionError):
                self._backend(reference_audio=ref).synthesize("hi", "out.wav")

    def test_voice_kwarg_route_is_gated(self):
        """This route was unguarded: only the positional voice was checked."""
        with tempfile.TemporaryDirectory() as d:
            ref = _write_wav(os.path.join(d, "ref.wav"))
            with self.assertRaises(PermissionError):
                self._backend(voice=ref).synthesize("hi", "out.wav")

    def test_positional_voice_route_is_gated(self):
        with tempfile.TemporaryDirectory() as d:
            ref = _write_wav(os.path.join(d, "ref.wav"))
            with self.assertRaises(PermissionError):
                self._backend().synthesize("hi", "out.wav", voice=ref)


class TestOptOutRequiresAttestation(unittest.TestCase):
    """Reducing provenance requires taking responsibility, as the binary does."""

    def _args(self, **over):
        import argparse

        base = dict(
            no_watermark=False,
            no_c2pa=False,
            no_spoken_disclaimer=False,
            accept_marking_responsibility=False,
        )
        base.update(over)
        return argparse.Namespace(**base)

    def test_bare_opt_out_is_refused(self):
        import cli

        for flag in ("no_watermark", "no_c2pa", "no_spoken_disclaimer"):
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit) as ctx:
                    cli._require_marking_attestation(self._args(**{flag: True}))
                self.assertEqual(ctx.exception.code, 2)

    def test_attested_opt_out_proceeds(self):
        import cli

        cli._require_marking_attestation(
            self._args(no_watermark=True, accept_marking_responsibility=True)
        )

    def test_no_opt_out_proceeds(self):
        import cli

        cli._require_marking_attestation(self._args())


class TestAuditIsRecordedAfterTheRun(unittest.TestCase):
    """An Art. 12 entry must document an event, not an intention."""

    def test_parsing_arguments_does_not_write_an_entry(self):
        """Reading the flags must not by itself assert that people were identified."""
        import inspect

        import cli

        source = inspect.getsource(cli.main)
        self.assertNotIn(
            "_audit_speaker_biometrics(args)",
            source,
            "the audit entry is written at parse time, before the run happened",
        )

    def test_transcription_records_the_event(self):
        import inspect

        import cli

        source = inspect.getsource(cli._run_transcribe)
        self.assertIn("_audit_speaker_biometrics(args)", source)


if __name__ == "__main__":
    unittest.main()
