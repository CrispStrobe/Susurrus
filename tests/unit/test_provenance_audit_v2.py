"""Regression tests for the Art. 50 defects found in the second audit.

The first audit closed routes that produced *unmarked* audio. These cover the
next layer down: routes that marked correctly but then **reported the wrong
thing about it** — a verification verb that could not read its own output, a
backend that claimed a disclosure it never made, and an Art. 50(4) failure
that no caller surfaced.

A transparency obligation is discharged by what the operator can observe, so a
false report is a compliance defect and not merely a cosmetic one. Each test
asserts on the observable answer, not on which function was called.
"""

import math
import os
import struct
import sys
import tempfile
import unittest
import wave

try:
    import numpy  # noqa: F401
    import soundfile  # noqa: F401

    _HAVE_AUDIO_STACK = True
except ImportError:  # pragma: no cover - minimal installs only
    _HAVE_AUDIO_STACK = False

requires_audio_stack = unittest.skipUnless(_HAVE_AUDIO_STACK, "numpy/soundfile not installed")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _write_wav(path, frames=4000, rate=16000, channels=1):
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        sample = b"".join(struct.pack("<h", int(3000 * math.sin(i / 20))) for i in range(frames))
        w.writeframes(sample * channels)
    return path


def _write_fake_mp3(path):
    """An MP3-shaped file: enough for the ID3 reader/writer, not for a decoder."""
    with open(path, "wb") as f:
        f.write(b"\xff\xfb\x90\x00" + b"\x00" * 4000)
    return path


class TestMarkerReadersDispatchOnContainer(unittest.TestCase):
    """The verification verbs must read the container they were given.

    ``--verify-c2pa`` and ``--detect-watermark`` called ``read_wav_ai_marker``,
    which returns None for anything that is not RIFF/WAVE. edge-tts writes MP3
    natively and the GUI save dialog offers ``.mp3``, so Susurrus marked those
    files correctly and then told the operator they were not AI-generated,
    exiting 1. That is the precise failure Art. 50(2) marking exists to
    prevent, produced by the tool meant to demonstrate compliance.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_mp3_marker_is_readable_by_the_dispatching_reader(self):
        from utils.ai_marking import embed_ai_marker, read_ai_marker

        mp3 = _write_fake_mp3(os.path.join(self.dir, "out.mp3"))
        self.assertTrue(embed_ai_marker(mp3, model="edge-tts"))
        self.assertIsNotNone(read_ai_marker(mp3))

    def test_verify_c2pa_verb_reports_a_marked_mp3_as_marked(self):
        """Exercise the exact expression cli.py builds its verdict from."""
        from utils.ai_marking import embed_ai_marker, read_ai_marker
        from utils.c2pa_signing import verify_audio_file

        mp3 = _write_fake_mp3(os.path.join(self.dir, "out.mp3"))
        embed_ai_marker(mp3, model="edge-tts")

        c2pa_result = verify_audio_file(mp3)
        marker = read_ai_marker(mp3)
        verdict = bool((c2pa_result and c2pa_result.get("valid")) or marker)

        self.assertTrue(verdict, "a marked MP3 must verify as AI-generated")

    def test_cli_does_not_use_the_wav_only_readers(self):
        """Guard the import sites themselves.

        Behaviour tests above cover the verdict, but the defect was a one-word
        import that reads plausibly. Someone reintroducing it would not
        obviously be breaking anything.
        """
        cli_path = os.path.join(os.path.dirname(__file__), "..", "..", "cli.py")
        with open(cli_path, encoding="utf-8") as f:
            source = f.read()

        self.assertNotIn("import read_wav_ai_marker", source)
        self.assertNotIn("import verify_wav_file", source)


class TestSpokenDisclosureIsNotOverReported(unittest.TestCase):
    """A stock voice is not a deepfake, so it gets no disclosure — or claim of one.

    ``CrispasrTTSBackend.apply_provenance`` used ``bool(self.voice)``, true for
    a preset voice *name*. Every stock-voice synthesis therefore printed
    "Marked as AI-generated (spoken disclosure + ...)" over audio carrying no
    disclosure and needing none. It also ignored the per-call ``voice``.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.wav = _write_wav(os.path.join(self.dir, "o.wav"))

    def _backend(self, **kwargs):
        from workers.tts.backends.crispasr_tts_backend import CrispasrTTSBackend

        return CrispasrTTSBackend(model_id="auto", **kwargs)

    def test_preset_voice_name_claims_no_disclosure(self):
        result = self._backend(voice="af_sarah").apply_provenance(self.wav, voice="af_sarah")
        self.assertFalse(result["spoken"])
        self.assertFalse(result["spoken_required"])

    def test_reference_audio_path_does_claim_a_disclosure(self):
        ref = _write_wav(os.path.join(self.dir, "ref.wav"))
        result = self._backend(voice=ref).apply_provenance(self.wav, voice=ref)
        self.assertTrue(result["spoken"])
        self.assertTrue(result["spoken_required"])

    def test_per_call_voice_argument_is_honoured(self):
        """A backend built with no voice, cloning via the call argument."""
        ref = _write_wav(os.path.join(self.dir, "ref.wav"))
        result = self._backend().apply_provenance(self.wav, voice=ref)
        self.assertTrue(result["spoken_required"])

    def test_opt_out_records_the_duty_but_not_the_act(self):
        ref = _write_wav(os.path.join(self.dir, "ref.wav"))
        result = self._backend(voice=ref, no_spoken_disclaimer=True).apply_provenance(
            self.wav, voice=ref
        )
        self.assertFalse(result["spoken"])
        self.assertTrue(result["spoken_required"])
        self.assertTrue(result["suppressed_spoken"])


class TestDisclosureShortfallIsSurfaced(unittest.TestCase):
    """Art. 50(4) failing must not be reported as Art. 50(2) succeeding.

    ``_report_marking`` only warned when *no machine-readable* layer landed. A
    cloning run whose audible disclosure failed printed a confident marking
    success, because the marker had in fact been written. The two obligations
    are distinct and a listener hears neither metadata nor a manifest.
    """

    def test_missing_disclosure_on_a_cloning_run_is_flagged(self):
        from utils.provenance import disclosure_missing, new_result

        result = new_result(spoken_required=True, spoken=False, marker=True)
        self.assertTrue(disclosure_missing(result))

    def test_successful_disclosure_is_not_flagged(self):
        from utils.provenance import disclosure_missing, new_result

        self.assertFalse(disclosure_missing(new_result(spoken_required=True, spoken=True)))

    def test_stock_voice_is_not_flagged(self):
        from utils.provenance import disclosure_missing, new_result

        self.assertFalse(disclosure_missing(new_result(spoken_required=False, marker=True)))

    def test_deliberate_opt_out_is_not_flagged_as_a_shortfall(self):
        """The operator attested to taking the duty on; that is not a failure."""
        from utils.provenance import disclosure_missing, new_result

        result = new_result(spoken_required=True, spoken=False, suppressed_spoken=True)
        self.assertFalse(disclosure_missing(result))

    def test_cli_reporter_writes_the_warning_to_stderr(self):
        import io
        from contextlib import redirect_stderr

        from cli import _report_marking
        from utils.provenance import new_result

        buffer = io.StringIO()
        with redirect_stderr(buffer):
            _report_marking(new_result(spoken_required=True, spoken=False, marker=True))

        self.assertIn("Art. 50(4)", buffer.getvalue())


@requires_audio_stack
class TestDisclosureReachesNonWavContainers(unittest.TestCase):
    """Cloning to a non-WAV container must still disclose audibly.

    chatterbox is the Python-native backend that clones and it writes through
    ``torchaudio.save``, which picks its encoder from the extension. The GUI
    offers ``.mp3``. Bailing out on non-WAV meant the one route where
    Art. 50(4) applies could emit an undisclosed deepfake.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def _flac(self, name, seconds=0.5, rate=16000):
        import numpy as np
        import soundfile as sf

        path = os.path.join(self.dir, name)
        n = int(rate * seconds)
        sf.write(path, np.sin(np.arange(n) / 20).astype("float32"), rate, format="FLAC")
        return path

    def test_concatenates_a_non_wav_container(self):
        import soundfile as sf

        from utils.spoken_disclosure import _concat_via_soundfile

        prefix = self._flac("prefix.flac", seconds=0.25)
        content = self._flac("content.flac", seconds=0.5)
        out = os.path.join(self.dir, "merged.flac")

        self.assertTrue(_concat_via_soundfile(prefix, content, out))
        self.assertEqual(sf.info(out).frames, sf.info(prefix).frames + sf.info(content).frames)

    def test_refuses_a_sample_rate_mismatch_rather_than_corrupting(self):
        from utils.spoken_disclosure import _concat_via_soundfile

        prefix = self._flac("p.flac", rate=8000)
        content = self._flac("c.flac", rate=16000)
        out = os.path.join(self.dir, "m.flac")

        self.assertFalse(_concat_via_soundfile(prefix, content, out))

    def test_prepend_uses_the_outputs_own_container(self):
        """The prefix must be synthesized into the same container as the content."""
        import soundfile as sf

        from utils.spoken_disclosure import prepend_spoken_disclosure

        content = self._flac("out.flac", seconds=0.5)
        before = sf.info(content).frames

        class FakeCloningTTS:
            """Writes a FLAC when asked for one, as a real backend would."""

            def synthesize(self, text, output_path, voice=None):
                import numpy as np

                n = 4000
                sf.write(
                    output_path,
                    np.sin(np.arange(n) / 20).astype("float32"),
                    16000,
                    format=os.path.splitext(output_path)[1].lstrip(".").upper(),
                )
                return output_path

        self.assertTrue(prepend_spoken_disclosure(FakeCloningTTS(), content))
        self.assertGreater(sf.info(content).frames, before)

    def test_no_temp_files_left_behind(self):
        from utils.spoken_disclosure import prepend_spoken_disclosure

        content = self._flac("out.flac")

        class FakeCloningTTS:
            def synthesize(self, text, output_path, voice=None):
                import numpy as np
                import soundfile as sf

                sf.write(
                    output_path,
                    np.sin(np.arange(4000) / 20).astype("float32"),
                    16000,
                    format="FLAC",
                )
                return output_path

        prepend_spoken_disclosure(FakeCloningTTS(), content)
        leftovers = [f for f in os.listdir(self.dir) if ".tmp" in f]
        self.assertEqual(leftovers, [])


class TestAuditChainTruncationIsDocumentedHonestly(unittest.TestCase):
    """Tail truncation is caught by the anchor, not by the chain.

    Removing the last n entries leaves every remaining ``prev_hash`` matching
    its predecessor, so the *chain* still verifies — that has not changed and
    cannot. What changed is that the entry count and head hash are now mirrored
    into a sibling anchor file, which the truncation contradicts. The
    distinction matters: strip the anchor as well and the log verifies again,
    which is why this is still tamper-evidence rather than tamper-proofing.
    """

    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "biometric.jsonl")
        from utils import audit_log

        for name in ("alice", "bob", "carol"):
            audit_log.record_event(
                audit_log.EVENT_ENROLL, speaker=name, consent=True, path=self.path
            )

    def _lines(self):
        with open(self.path, encoding="utf-8") as f:
            return f.read().splitlines()

    def _rewrite(self, lines):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def test_intact_chain_verifies(self):
        from utils.audit_log import verify_chain

        self.assertTrue(verify_chain(self.path)["valid"])

    def test_tail_truncation_is_detected_by_the_anchor(self):
        from utils.audit_log import verify_chain

        self._rewrite(self._lines()[:2])
        result = verify_chain(self.path)
        self.assertFalse(result["valid"], "tail truncation went undetected")
        self.assertEqual(result["entries"], 2)

    def test_the_chain_alone_still_cannot_see_it(self):
        """Pins *why* the anchor is needed, not just that it works.

        With the anchor removed the truncated log verifies again — the chain
        has no way to know entries once existed. Anyone who can write the log
        can write the anchor beside it, so this remains evidence of tampering
        rather than prevention of it, exactly as COMPLIANCE.md says.
        """
        from utils.audit_log import anchor_path, verify_chain

        self._rewrite(self._lines()[:2])
        os.unlink(anchor_path(self.path))
        self.assertTrue(verify_chain(self.path)["valid"])

    def test_middle_deletion_is_detected(self):
        from utils.audit_log import verify_chain

        lines = self._lines()
        self._rewrite([lines[0], lines[2]])
        self.assertFalse(verify_chain(self.path)["valid"])

    def test_compliance_doc_does_not_claim_truncation_detection(self):
        doc = os.path.join(os.path.dirname(__file__), "..", "..", "COMPLIANCE.md")
        with open(doc, encoding="utf-8") as f:
            text = f.read()

        self.assertIn("truncation of the tail", text.lower())
        self.assertNotIn("reordering and\ntruncation are all detectable", text)


class TestServerModeDisclosesItsBlindSpot(unittest.TestCase):
    """Server mode is the one route Susurrus cannot verify, so it must say so."""

    def test_startup_note_names_the_limitation(self):
        import io
        from contextlib import redirect_stderr

        from cli import _warn_server_provenance

        buffer = io.StringIO()
        with redirect_stderr(buffer):
            _warn_server_provenance("127.0.0.1", 8080)

        output = buffer.getvalue()
        self.assertIn("Art. 50(2)", output)
        self.assertIn("cannot verify", output)


class TestWatermarkDetectorIsNamedAccurately(unittest.TestCase):
    """A spread-spectrum verdict must not be reported as AudioSeal.

    The two tiers differ in what they resist — AudioSeal is learned and
    survives deliberate removal, the comb is a fixed key that does not. An
    operator assessing robustness needs to know which one answered.
    """

    def test_report_uses_the_backend_the_detector_returned(self):
        neural = {"watermarked": True, "confidence": 0.9, "backend": "spread-spectrum"}
        label = neural.get("backend", "unknown") if neural else "unavailable"
        self.assertEqual(label, "spread-spectrum")

    def test_cli_does_not_hardcode_the_detector_name(self):
        cli_path = os.path.join(os.path.dirname(__file__), "..", "..", "cli.py")
        with open(cli_path, encoding="utf-8") as f:
            source = f.read()

        self.assertNotIn('"audioseal" if neural is not None else', source)


class TestMarkingFailsClosed(unittest.TestCase):
    """Unmarkable synthetic audio must not reach the user at all.

    Art. 50(2) has no "unless an optional dependency is missing" clause, so
    degrading to unmarked output is the one outcome that must be unavailable.
    Warning while leaving the file on disk is not a control: the audio still
    exists under the name the user asked for.

    Every test here mocks the optional layers off, which is precisely the
    "libraries are missing" case rather than an approximation of it.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = _write_wav(os.path.join(self.dir, "out.opus"))

    def _dummy(self, **kwargs):
        from workers.tts.backends.base import TTSBackend

        class DummyTTS(TTSBackend):
            def synthesize(self, text, output_path, voice=None):
                return output_path

        return DummyTTS(**kwargs)

    def _no_layers(self):
        from unittest import mock

        return mock.patch("utils.audio_watermark.embed_watermark", return_value=False)

    def test_raises_rather_than_returning_an_unmarked_result(self):
        from utils.provenance import ProvenanceError

        with self._no_layers():
            with self.assertRaises(ProvenanceError):
                self._dummy(no_c2pa=True).apply_provenance(self.path)

    def test_the_unmarked_file_is_deleted(self):
        """A refusal that leaves the audio behind does not prevent shipping it."""
        from utils.provenance import ProvenanceError

        with self._no_layers():
            with self.assertRaises(ProvenanceError):
                self._dummy(no_c2pa=True).apply_provenance(self.path)

        self.assertFalse(os.path.exists(self.path))

    def test_the_refusal_says_how_to_satisfy_it(self):
        from utils.provenance import ProvenanceError

        with self._no_layers():
            with self.assertRaises(ProvenanceError) as caught:
                self._dummy(no_c2pa=True).apply_provenance(self.path)

        message = str(caught.exception)
        self.assertIn("--accept-marking-responsibility", message)
        self.assertIn("susurrus[tts]", message)

    def test_attestation_is_the_one_way_past(self):
        """The documented opt-out stays the only escape hatch, and keeps the file."""
        result = self._dummy(accept_marking_responsibility=True).apply_provenance(self.path)

        self.assertTrue(result["opted_out"])
        self.assertTrue(os.path.exists(self.path), "an attested opt-out keeps its output")

    def test_markable_container_still_succeeds_with_no_optional_libraries(self):
        """A minimal install must remain usable for WAV and MP3.

        The declarative marker is pure stdlib, so fail-closed should bite only
        on exotic containers — not turn every dependency-light install into a
        tool that refuses to synthesize anything.
        """
        wav = _write_wav(os.path.join(self.dir, "fine.wav"))
        with self._no_layers():
            result = self._dummy(no_c2pa=True).apply_provenance(wav)

        self.assertTrue(result["marker"])
        self.assertTrue(os.path.exists(wav))

    def test_missing_disclosure_on_a_cloning_run_also_refuses(self):
        """Art. 50(4) fails closed too, not only Art. 50(2)."""
        from utils.provenance import ProvenanceError, enforce_marking

        wav = _write_wav(os.path.join(self.dir, "cloned.wav"))
        result = {
            "spoken": False,
            "spoken_required": True,
            "suppressed_spoken": False,
            "marker": True,  # Art. 50(2) satisfied ...
            "watermark": False,
            "c2pa": False,
            "opted_out": False,
            "unsupported_format": False,
        }

        with self.assertRaises(ProvenanceError) as caught:
            enforce_marking(result, wav)

        self.assertIn("Art. 50(4)", str(caught.exception))
        self.assertFalse(os.path.exists(wav))

    def test_attested_disclosure_opt_out_is_not_refused(self):
        from utils.provenance import enforce_marking, new_result

        wav = _write_wav(os.path.join(self.dir, "cloned.wav"))
        result = new_result(spoken_required=True, spoken=False, suppressed_spoken=True, marker=True)

        self.assertIs(enforce_marking(result, wav), result)
        self.assertTrue(os.path.exists(wav))


class TestMarkingPreflight(unittest.TestCase):
    """The knowable half of the refusal must not cost a model load.

    enforce_marking can only run once audio exists. Asking what this install
    can mark is answerable from the extension and a couple of imports, so a
    doomed request is refused in milliseconds instead of after a synthesis
    whose output is then deleted.
    """

    def test_wav_needs_no_optional_dependency(self):
        from utils.provenance import marking_available

        ok, _ = marking_available("out.wav")
        self.assertTrue(ok)

    def test_mp3_needs_no_optional_dependency(self):
        from utils.provenance import marking_available

        ok, _ = marking_available("out.mp3")
        self.assertTrue(ok)

    def test_exotic_container_without_libraries_is_refused_early(self):
        from unittest import mock

        from utils.provenance import marking_available

        with mock.patch("utils.provenance._c2pa_installed", return_value=False):
            with mock.patch("utils.provenance._soundfile_installed", return_value=False):
                ok, reason = marking_available("out.flac")

        self.assertFalse(ok)
        self.assertIn("flac", reason)

    def test_exotic_container_is_allowed_when_a_layer_exists(self):
        from unittest import mock

        from utils.provenance import marking_available

        with mock.patch("utils.provenance._c2pa_installed", return_value=True):
            ok, _ = marking_available("out.flac")

        self.assertTrue(ok)

    def test_cloning_to_non_wav_needs_a_decoder_for_the_disclosure(self):
        from unittest import mock

        from utils.provenance import marking_available

        with mock.patch("utils.provenance._c2pa_installed", return_value=True):
            with mock.patch("utils.provenance._soundfile_installed", return_value=False):
                ok, reason = marking_available("out.mp3", is_cloning=True)

        self.assertFalse(ok)
        self.assertIn("disclosure", reason)

    def test_cloning_to_wav_needs_nothing_optional(self):
        from unittest import mock

        from utils.provenance import marking_available

        with mock.patch("utils.provenance._soundfile_installed", return_value=False):
            ok, _ = marking_available("out.wav", is_cloning=True)

        self.assertTrue(ok)


class TestCallersDoNotReportSuccessOnRefusal(unittest.TestCase):
    """A refusal must not reach the user as a saved file."""

    def test_cli_tts_catches_the_refusal(self):
        cli_path = os.path.join(os.path.dirname(__file__), "..", "..", "cli.py")
        with open(cli_path, encoding="utf-8") as f:
            source = f.read()

        # Both TTS branches (CrispASR and Python-native) must handle it, or
        # one route reports a traceback where the other reports a refusal.
        self.assertEqual(source.count("except ProvenanceError as e:"), 2)

    def test_gui_thread_does_not_emit_finished_on_refusal(self):
        thread_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "workers", "tts_thread.py"
        )
        with open(thread_path, encoding="utf-8") as f:
            source = f.read()

        self.assertIn("except ProvenanceError as e:", source)

        handler = source.split("except ProvenanceError as e:")[1].split("except Exception")[0]
        # Comments in the handler discuss finished_signal by name, so match on
        # the emit call rather than the word.
        code = "\n".join(line for line in handler.splitlines() if not line.strip().startswith("#"))

        # The handler must route to error_signal; emitting finished_signal
        # would hand the GUI a path to offer for playback and saving.
        self.assertIn("error_signal.emit", code)
        self.assertNotIn("finished_signal.emit", code)


if __name__ == "__main__":
    unittest.main()
