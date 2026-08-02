"""Test that provenance features are actually *invoked*, not merely defined.

The v2.10.0 audit found `TTSBackend.sign_output()` defined with zero callers
while the tests asserted only that the method existed — green suite, unmarked
output. These tests assert the call sites, so a disconnected hook fails here.
"""

import argparse
import os
import sys
import tempfile
import unittest
import wave
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from workers.tts.backends.base import TTSBackend  # noqa: E402


def _pyqt6_available():
    """Local check — importing tests.conftest drags in pyannote/NeMo."""
    try:
        import PyQt6  # noqa: F401

        return True
    except ImportError:
        return False


skip_no_pyqt6 = unittest.skipUnless(_pyqt6_available(), "PyQt6 not installed")


def _write_wav(path):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 100)


class RecordingBackend(TTSBackend):
    """Stand-in backend that records how it was driven."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.provenance_calls = []
        self.output_path = None

    def synthesize(self, text, output_path="out.wav", voice=None):
        self.output_path = output_path
        return output_path

    def apply_provenance(self, output_path, model=None, voice=None, locale=None):
        self.provenance_calls.append(output_path)
        from utils.provenance import new_result

        return new_result(watermark=True, marker=True, c2pa=True)


class TestCliCallsProvenance(unittest.TestCase):
    """The CLI must mark synthesized audio on both routing branches."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)

    def _args(self, **over):
        base = dict(
            text="hello",
            input_file=None,
            tts_backend="piper",
            backend="piper",
            tts_output="out.wav",
            model=None,
            device="cpu",
            language=None,
            voice=None,
            list_voices=False,
            i_have_rights=False,
            no_c2pa=False,
            no_watermark=False,
            no_spoken_disclaimer=False,
            accept_marking_responsibility=False,
            c2pa_cert=None,
            c2pa_key=None,
        )
        base.update(over)
        return argparse.Namespace(**base)

    def test_python_backend_path_marks_output(self):
        import cli

        backend = RecordingBackend()
        with mock.patch.object(cli, "get_tts_backend_class", return_value=lambda **kw: backend):
            cli._run_tts(self._args())

        self.assertEqual(
            backend.provenance_calls,
            ["out.wav"],
            "CLI synthesized without marking the output (Art. 50(2))",
        )

    def test_python_backend_receives_provenance_kwargs(self):
        """cert/key/consent must reach the backend, not be dropped en route."""
        import cli

        captured = {}

        def factory(**kwargs):
            captured.update(kwargs)
            return RecordingBackend()

        with mock.patch.object(cli, "get_tts_backend_class", return_value=factory):
            cli._run_tts(
                self._args(
                    i_have_rights=True,
                    no_c2pa=True,
                    no_watermark=True,
                    accept_marking_responsibility=True,
                    c2pa_cert="/certs/my.pem",
                    c2pa_key="/certs/my.key",
                )
            )

        self.assertTrue(captured["i_have_rights"])
        self.assertTrue(captured["no_c2pa"])
        self.assertTrue(
            captured["no_watermark"],
            "--no-watermark was dropped before reaching Python backends",
        )
        self.assertIn("no_spoken_disclaimer", captured)
        self.assertEqual(captured["c2pa_cert"], "/certs/my.pem")
        self.assertEqual(captured["c2pa_key"], "/certs/my.key")

    def _crispasr_backend_class(self):
        """Resolve the class the CLI actually uses for `--tts` on CrispASR.

        ``cli.get_backend_class`` installs placeholder modules under
        ``workers.transcription`` so it can import without PyQt6. That is
        harmless in a one-shot CLI process but poisons later imports in a
        shared test process, so snapshot and restore the affected keys.
        """
        import cli

        touched = ["workers", "workers.transcription", "workers.transcription.backends"]
        saved = {k: sys.modules.get(k) for k in touched}

        def restore():
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

        self.addCleanup(restore)
        return cli.get_backend_class("crispasr")

    def test_crispasr_branch_backend_supports_provenance(self):
        """The CLI's CrispASR TTS branch resolves the *transcription* class.

        It exposes synthesize(), so it must also expose apply_provenance() or
        the marking call raises AttributeError on every CrispASR synthesis.
        """
        backend_class = self._crispasr_backend_class()
        self.assertTrue(hasattr(backend_class, "synthesize"))
        self.assertTrue(hasattr(backend_class, "apply_provenance"))

        backend = backend_class(model_id="auto", device="cpu")

        # Marking must be read off the file, not inferred from the flags: a
        # binary built without C2PA support, or an engine that cannot
        # watermark, previously still produced a confident "marked" report.
        path = os.path.join(self._dir.name, "out.wav")
        _write_wav(path)
        result = backend.apply_provenance(path)
        self.assertFalse(result["opted_out"])
        self.assertTrue(result["marker"], "unmarked binary output needs a marker floor")

        from utils.ai_marking import read_ai_marker

        self.assertIsNotNone(read_ai_marker(path))

    def test_crispasr_opt_out_is_self_consistent(self):
        """Opting out must clear every layer, not just set the flag."""
        backend = self._crispasr_backend_class()(
            model_id="auto", device="cpu", accept_marking_responsibility=True
        )
        path = os.path.join(self._dir.name, "optout.wav")
        _write_wav(path)
        result = backend.apply_provenance(path)
        self.assertTrue(result["opted_out"])
        self.assertFalse(result["c2pa"])
        self.assertFalse(result["marker"])

        from utils.ai_marking import read_ai_marker

        self.assertIsNone(read_ai_marker(path), "opt-out must not mark the file")

    def test_clone_without_consent_exits_nonzero(self):
        import cli

        class RefusingBackend(RecordingBackend):
            def synthesize(self, text, output_path="out.wav", voice=None):
                self.require_clone_consent("/ref.wav")
                return output_path

        backend = RefusingBackend()
        with mock.patch.object(cli, "get_tts_backend_class", return_value=lambda **kw: backend):
            with self.assertRaises(SystemExit) as ctx:
                cli._run_tts(self._args(voice="/ref.wav"))
        self.assertEqual(ctx.exception.code, 2)
        self.assertEqual(backend.provenance_calls, [], "refused output must not be marked")


@skip_no_pyqt6
class TestTTSThreadCallsProvenance(unittest.TestCase):
    """The GUI thread must mark output and forward every provenance control."""

    def _run_thread(self, args):
        from workers.tts_thread import TTSThread

        backend = RecordingBackend()
        captured = {}

        def factory(name, **kwargs):
            captured.update(kwargs)
            return backend

        base = {"tts_backend": "piper", "text": "hello", "output_path": "out.wav"}
        base.update(args)
        with mock.patch("workers.tts.backends.get_tts_backend", factory):
            TTSThread(base).run()
        return backend, captured

    def test_marks_output(self):
        backend, _ = self._run_thread({})
        self.assertEqual(
            backend.provenance_calls,
            ["out.wav"],
            "TTSThread synthesized without marking the output (Art. 50(2))",
        )

    def test_forwards_every_provenance_control(self):
        """Each GUI widget must reach the backend — no silently inert checkbox."""
        _, captured = self._run_thread(
            {
                "i_have_rights": True,
                "no_c2pa": True,
                "no_watermark": True,
                "no_spoken_disclaimer": True,
                "accept_marking_responsibility": True,
                "c2pa_cert": "/certs/my.pem",
                "c2pa_key": "/certs/my.key",
                "reference_audio": "/ref.wav",
            }
        )
        self.assertTrue(captured["i_have_rights"])
        self.assertTrue(captured["no_c2pa"])
        self.assertTrue(
            captured["no_watermark"],
            "--no-watermark was dropped before reaching Python backends",
        )
        self.assertTrue(captured["no_spoken_disclaimer"])
        self.assertTrue(captured["accept_marking_responsibility"])
        self.assertEqual(captured["c2pa_cert"], "/certs/my.pem")
        self.assertEqual(captured["c2pa_key"], "/certs/my.key")
        self.assertEqual(captured["reference_audio"], "/ref.wav")

    def test_crispasr_branch_forwards_provenance(self):
        from workers.tts_thread import TTSThread

        captured = {}

        class FakeCrispasr(RecordingBackend):
            def __init__(self, **kwargs):
                super().__init__()
                captured.update(kwargs)

        with mock.patch(
            "workers.tts.backends.crispasr_tts_backend.CrispasrTTSBackend", FakeCrispasr
        ):
            TTSThread(
                {
                    "tts_backend": "crispasr:kokoro",
                    "text": "hello",
                    "output_path": "out.wav",
                    "no_watermark": True,
                    "no_c2pa": True,
                    "accept_marking_responsibility": True,
                    "c2pa_cert": "/certs/my.pem",
                    "ref_text": "reference sentence",
                }
            ).run()

        self.assertTrue(captured["no_watermark"])
        self.assertTrue(captured["no_c2pa"])
        self.assertEqual(captured["c2pa_cert"], "/certs/my.pem")
        self.assertEqual(captured["ref_text"], "reference sentence")


class TestSpeakerBiometricsWarning(unittest.TestCase):
    """Enrolling a named speaker stores biometrics — warn without consent."""

    def _ns(self, **over):
        base = dict(
            speaker_db=None,
            enroll_speaker=None,
            expect_speakers=None,
            speaker_db_consent=False,
        )
        base.update(over)
        return argparse.Namespace(**base)

    def test_warns_on_unconsented_enrollment(self):
        import cli

        self.assertTrue(cli._warn_speaker_biometrics(self._ns(enroll_speaker="alice")))
        self.assertTrue(cli._warn_speaker_biometrics(self._ns(speaker_db="/db")))
        self.assertTrue(cli._warn_speaker_biometrics(self._ns(expect_speakers="alice,bob")))

    def test_silent_when_consent_attested(self):
        import cli

        self.assertFalse(
            cli._warn_speaker_biometrics(self._ns(enroll_speaker="alice", speaker_db_consent=True))
        )

    def test_silent_when_biometrics_unused(self):
        import cli

        self.assertFalse(cli._warn_speaker_biometrics(self._ns()))


class TestVerifyCommand(unittest.TestCase):
    """`--verify-c2pa` answers "is this marked?", across both layers."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "out.wav")
        _write_wav(self.path)

    def tearDown(self):
        self._dir.cleanup()

    def _verify(self):
        import cli

        argv = ["susurrus", "--verify-c2pa", self.path]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as ctx:
                cli.main()
        return ctx.exception.code

    def test_unmarked_file_exits_nonzero(self):
        self.assertEqual(self._verify(), 1)

    def test_declarative_marker_alone_counts_as_marked(self):
        """Absent c2pa-audio, the marker is the only evidence — it must count.

        Previously this exited 1 whenever the optional library was missing,
        conflating "cannot check" with "not marked".
        """
        from utils.ai_marking import embed_wav_ai_marker

        embed_wav_ai_marker(self.path, model="piper")
        self.assertEqual(self._verify(), 0)


class TestProvenanceLayering(unittest.TestCase):
    """apply_provenance must fall back to the marker when C2PA is unavailable."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "out.wav")
        _write_wav(self.path)

    def tearDown(self):
        self._dir.cleanup()

    class Dummy(TTSBackend):
        def synthesize(self, text, output_path, voice=None):
            return output_path

    def test_marker_applied_when_c2pa_missing(self):
        from utils.ai_marking import is_ai_marked

        with mock.patch("utils.c2pa_signing.sign_audio_file", return_value=False):
            result = self.Dummy().apply_provenance(self.path)

        self.assertFalse(result["c2pa"])
        self.assertTrue(result["marker"], "no marking layer survived a missing c2pa lib")
        self.assertTrue(is_ai_marked(self.path))

    def test_c2pa_receives_cert_and_key(self):
        with mock.patch("utils.c2pa_signing.sign_audio_file", return_value=True) as signer:
            self.Dummy(c2pa_cert="/c.pem", c2pa_key="/k.pem").apply_provenance(self.path)
        signer.assert_called_once_with(self.path, cert_pem="/c.pem", key_pem="/k.pem", model=None)

    def test_no_c2pa_skips_signing_but_still_marks(self):
        from utils.ai_marking import is_ai_marked

        with mock.patch("utils.c2pa_signing.sign_audio_file") as signer:
            result = self.Dummy(no_c2pa=True).apply_provenance(self.path)
        signer.assert_not_called()
        self.assertTrue(result["marker"])
        self.assertTrue(is_ai_marked(self.path))

    def test_accept_responsibility_skips_all_marking(self):
        from utils.ai_marking import is_ai_marked

        result = self.Dummy(accept_marking_responsibility=True).apply_provenance(self.path)
        self.assertTrue(result["opted_out"])
        self.assertFalse(result["c2pa"])
        self.assertFalse(result["marker"])
        self.assertFalse(is_ai_marked(self.path))

    def test_mp3_output_is_marked(self):
        """MP3 is not a hypothetical: edge-tts synthesizes it natively."""
        import os
        import shutil
        import subprocess

        from utils.ai_marking import read_ai_marker

        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not available to build an MP3 fixture")

        mp3 = os.path.join(self._dir.name, "out.mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", self.path, "-codec:a", "libmp3lame", mp3],
            check=True,
        )
        result = self.Dummy().apply_provenance(mp3)
        self.assertTrue(result["marker"])
        self.assertFalse(result["unsupported_format"])
        self.assertIsNotNone(read_ai_marker(mp3))

    def test_unmarkable_container_is_refused_and_deleted(self):
        """An unmarkable container fails closed rather than surfacing a warning.

        Previously this asserted that the result merely *reported* the gap.
        Reporting is not enough: the unmarked file stayed on disk under the
        name the user asked for. Watermarking and C2PA are mocked off so the
        outcome does not depend on which optional packages are present.
        """
        import os
        import shutil
        from unittest import mock

        from utils.provenance import ProvenanceError

        target = os.path.join(self._dir.name, "out.opus")
        shutil.copyfile(self.path, target)

        with mock.patch("utils.audio_watermark.embed_watermark", return_value=False):
            with self.assertRaises(ProvenanceError):
                self.Dummy(no_c2pa=True).apply_provenance(target)

        self.assertFalse(os.path.exists(target), "unmarked audio must not survive")


if __name__ == "__main__":
    unittest.main()
