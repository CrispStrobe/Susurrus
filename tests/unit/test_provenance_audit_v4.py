"""Regressions for the fourth EU AI Act audit.

Each test names the failure it prevents rather than the code it touches. The
findings this file covers:

1. Every narrow provenance opt-out required the responsibility attestation, and
   the attestation short-circuited the whole pipeline — so ``--no-c2pa`` did not
   skip "only the cryptographic layer" as documented, it skipped all of them.
2. The Art. 50(4) ``unknown`` speaker warning existed only as a log record, and
   in the GUI nothing carried log records to the operator.
3. The CLI preflight asked ``os.path.isfile(voice)`` while the authoritative
   gate asked :func:`would_clone`, so the two disagreed about the two
   documented no-path cloning routes.
"""

import os
import shutil
import tempfile
import unittest
import wave

from utils import provenance
from workers.tts.backends.base import would_clone


def _pyqt6_available():
    try:
        import PyQt6  # noqa: F401

        return True
    except ImportError:
        return False


skip_no_pyqt6 = unittest.skipUnless(_pyqt6_available(), "PyQt6 not installed")


def _write_wav(path, frames=8000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * frames)
    return path


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def wav(self, name="out.wav"):
        return _write_wav(os.path.join(self.dir, name))


# ---------------------------------------------------------------------------
# Finding 1 — the narrow opt-outs were unreachable on the Python-native path.
# ---------------------------------------------------------------------------


class TestNarrowOptOutsActuallyNarrow(_Tmp):
    """A flag that names one layer must not silently drop the other two.

    Each of --no-watermark / --no-c2pa / --no-spoken-disclaimer requires
    --accept-marking-responsibility alongside it, and the attestation used to
    return before any layer ran. So the documented "skips only the cryptographic
    layer" was unachievable: the combination the CLI *forces* you to pass
    produced completely unmarked audio. The CrispASR routes forward the flags to
    the binary individually and did honour the documented behaviour, which is
    the backend-dependent split the attestation rule exists to prevent.
    """

    def test_no_c2pa_still_marks(self):
        path = self.wav()
        result = provenance.apply_provenance(
            path,
            options={"accept_marking_responsibility": True, "no_c2pa": True},
            model="test",
        )
        self.assertFalse(result["opted_out"], "a narrow opt-out is not a full opt-out")
        self.assertTrue(result["marker"], "the declarative floor must still land")
        self.assertFalse(result["c2pa"], "the named layer is the one that is skipped")

    def test_no_watermark_still_marks(self):
        path = self.wav()
        result = provenance.apply_provenance(
            path,
            options={"accept_marking_responsibility": True, "no_watermark": True},
            model="test",
        )
        self.assertFalse(result["opted_out"])
        self.assertTrue(result["marker"])
        self.assertFalse(result["watermark"])

    def test_bare_attestation_is_still_a_full_opt_out(self):
        """The documented behaviour for the attestation on its own is unchanged."""
        path = self.wav()
        result = provenance.apply_provenance(
            path,
            options={"accept_marking_responsibility": True},
            model="test",
        )
        self.assertTrue(result["opted_out"])
        self.assertFalse(provenance.marking_applied(result))

    def test_marking_still_required_without_attestation(self):
        """Narrowing must not become a way past the gate without attesting."""
        path = self.wav()
        result = provenance.apply_provenance(path, options={"no_c2pa": True}, model="test")
        self.assertFalse(result["opted_out"])
        self.assertTrue(result["marker"])

    def test_narrowed_opt_outs_reports_only_named_layers(self):
        self.assertEqual(provenance.narrowed_opt_outs({}), [])
        self.assertEqual(
            provenance.narrowed_opt_outs({"accept_marking_responsibility": True}),
            [],
            "the attestation alone names no layer",
        )
        self.assertEqual(provenance.narrowed_opt_outs({"no_c2pa": True}), ["no_c2pa"])


# ---------------------------------------------------------------------------
# Finding 2 — the unknown-speaker warning never reached a GUI operator.
# ---------------------------------------------------------------------------


@skip_no_pyqt6
class TestUnknownSpeakerReachesTheOperator(unittest.TestCase):
    """The `unknown` policy is defensible only if the operator is told.

    Susurrus deliberately does not force a disclosure for a preset voice whose
    provenance nobody established, on the grounds that the gap is "loud rather
    than silent". The loudness was one logger.warning; the GUI routes logging to
    a stderr a packaged app has no window for, and attaches its log-viewer
    handler only once the user opens Tools > Logs. 43 of 59 exposed backends
    resolve to `unknown`, so this is the common path.
    """

    def _describe(self, marking, backend="crispasr:bark"):
        from workers.tts_thread import _describe_unknown_speaker

        return _describe_unknown_speaker(marking, backend)

    def test_unknown_preset_warns(self):
        message = self._describe(
            provenance.new_result(speaker_identity="unknown", marker=True),
        )
        self.assertIsNotNone(message, "an unanswered Art. 50(4) question must be surfaced")
        self.assertIn("crispasr:bark", message)
        self.assertIn("50(4)", message)

    def test_known_speaker_is_quiet(self):
        for identity in ("real_person", "synthetic"):
            self.assertIsNone(
                self._describe(provenance.new_result(speaker_identity=identity, marker=True)),
                f"{identity} is an answer, not a question",
            )

    def test_cloning_run_is_quiet(self):
        """A cloned voice already gets the disclosure; the preset question is moot."""
        self.assertIsNone(
            self._describe(
                provenance.new_result(
                    speaker_identity="unknown", spoken_required=True, spoken=True, marker=True
                )
            )
        )


class TestUnknownSpeakerWarningIsLocalized(unittest.TestCase):
    """A German operator must get the warning in German, as every other does."""

    def test_translation_keys_exist_in_every_language(self):
        from utils.translations import de, en

        for table in (en.STRINGS, de.STRINGS):
            self.assertIn("warn.speaker_identity_unknown", table)
            self.assertIn("{backend}", table["warn.speaker_identity_unknown"])
            self.assertIn("50(4)", table["warn.speaker_identity_unknown"])


# ---------------------------------------------------------------------------
# Finding 3 — preflight and the authoritative gate answered "is this cloning?"
# differently.
# ---------------------------------------------------------------------------


class TestCloningIsOneQuestion(_Tmp):
    """The cheap check and the real check must agree about what cloning is.

    Nothing unmarked escaped through the disagreement — the post-synthesis gate
    still fails closed — but the operator paid a model load and a synthesis for
    a refusal that was knowable up front, and two implementations of a safety
    predicate is how the earlier misses in this area started.
    """

    def test_a_recording_is_cloning(self):
        ref = _write_wav(os.path.join(self.dir, "victim.wav"))
        self.assertEqual(would_clone(voices=(ref,)), ref)

    def test_a_bare_name_under_voice_dir_is_cloning(self):
        _write_wav(os.path.join(self.dir, "alice.wav"))
        self.assertIsNotNone(
            would_clone(voices=("alice",), voice_dir=self.dir),
            "the documented ergonomic way to clone must not read as a preset",
        )

    def test_a_bare_name_resolving_to_nothing_is_a_preset(self):
        self.assertIsNone(would_clone(voices=("af_sarah",), voice_dir=self.dir))

    def test_a_voice_bank_selection_is_cloning(self):
        self.assertEqual(
            would_clone(backend_name="crispasr:cosyvoice3-tts", voices=("someone",)),
            "someone",
        )

    def test_a_preset_name_on_an_ordinary_backend_is_not(self):
        self.assertIsNone(would_clone(backend_name="piper", voices=("de_DE-thorsten",)))

    def test_no_voice_is_not_cloning(self):
        self.assertIsNone(would_clone(backend_name="piper", voices=(None, "")))

    def test_the_backend_gate_delegates_to_the_same_answer(self):
        """Whatever would_clone says, resolve_reference_audio must say too."""
        from workers.tts.backends.base import TTSBackend

        class _Stub(TTSBackend):
            def synthesize(self, text, output_path, voice=None):  # pragma: no cover
                return output_path

        _write_wav(os.path.join(self.dir, "alice.wav"))
        backend = _Stub(tts_backend_name="piper", voice_dir=self.dir)
        self.assertEqual(
            backend.resolve_reference_audio("alice"),
            would_clone(backend_name="piper", voices=("alice",), voice_dir=self.dir),
        )


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Finding 4 — COMPLIANCE.md drifted from the tables it describes.
# ---------------------------------------------------------------------------


class TestComplianceDocMatchesTheCode(unittest.TestCase):
    """The document is the artifact a regulator or a forker actually reads.

    Both of the documentation findings in this audit were drift: the coverage
    numbers described 40 backends as classified when 24 of those 40 record
    `unknown`, and the per-voice section still described a rule its test had
    stopped enforcing. Prose cannot be pinned in general, but the *numbers* can,
    and those are what someone would quote.
    """

    @staticmethod
    def _doc():
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "..", "COMPLIANCE.md")
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    @staticmethod
    def _counts():
        from config import CRISPASR_TTS_BACKENDS
        from utils.speaker_identity import BACKEND_SPEAKER_IDENTITY as table

        exposed = ["edge-tts", "piper", "kokoro-onnx", "chatterbox", "speecht5"]
        exposed += [f"crispasr:{b}" for b in CRISPASR_TTS_BACKENDS]
        determinate = [b for b in exposed if table.get(b) in ("real_person", "synthetic")]
        listed_unknown = [b for b in exposed if table.get(b) == "unknown"]
        absent = [b for b in exposed if b not in table]
        return {
            "exposed": len(exposed),
            "determinate": len(determinate),
            "listed_unknown": len(listed_unknown),
            "absent": len(absent),
            "unknown_total": len(listed_unknown) + len(absent),
        }

    def test_the_documented_numbers_are_the_real_ones(self):
        doc = self._doc()
        counts = self._counts()
        self.assertIn(
            f"exposes {counts['exposed']} TTS backends",
            doc,
            "the backend count in COMPLIANCE.md no longer matches the registry",
        )
        for label, value in (
            ("determinate", counts["determinate"]),
            ("listed_unknown", counts["listed_unknown"]),
            ("absent", counts["absent"]),
        ):
            self.assertIn(
                f"| {value} |" if label != "determinate" else f"| **{value}** |",
                doc,
                f"the {label} count ({value}) is not the one COMPLIANCE.md states",
            )
        self.assertIn(
            f"**{counts['unknown_total']} of {counts['exposed']}**",
            doc,
            "the headline unknown ratio is stale",
        )

    def test_no_table_entry_names_a_backend_that_does_not_exist(self):
        """A verdict for a backend nobody can select is a verdict nobody applies."""
        from config import CRISPASR_TTS_BACKENDS
        from utils.speaker_identity import BACKEND_SPEAKER_IDENTITY as table

        exposed = {"edge-tts", "piper", "kokoro-onnx", "chatterbox", "speecht5"}
        exposed |= {f"crispasr:{b}" for b in CRISPASR_TTS_BACKENDS}
        self.assertEqual(sorted(set(table) - exposed), [])

    def test_the_document_does_not_claim_the_per_voice_table_is_empty(self):
        """It said so for several releases after the table was filled."""
        from utils.speaker_identity import VOICE_SPEAKER_IDENTITY

        if VOICE_SPEAKER_IDENTITY:
            self.assertNotIn("so that table is empty", self._doc())

    def test_art_50_5_is_addressed(self):
        """It was absent entirely — not argued and dismissed, just missing."""
        doc = self._doc()
        self.assertIn("Art. 50(5)", doc)
        self.assertIn("accessib", doc.lower())


# ---------------------------------------------------------------------------
# Finding 5 — Art. 50(5): the disclosure existed only in a form you must hear.
# ---------------------------------------------------------------------------


class TestAccessibleDisclosureForm(unittest.TestCase):
    """An audible-only disclosure reaches nobody who cannot hear it.

    Art. 50(5) requires the Art. 50(1)-(4) information to conform to applicable
    accessibility requirements. An audio file has nowhere to put a caption, so
    the accessible form has to be composed by the deployer — but the *wording*
    should not be theirs to reinvent, or the written and spoken disclosures
    drift apart and neither is authoritative.
    """

    def test_the_spoken_phrase_is_reachable_as_text(self):
        from utils.spoken_disclosure import disclosure_text

        for locale in ("en", "de"):
            phrase = disclosure_text(locale=locale)
            self.assertTrue(phrase and phrase.strip(), f"no disclosure phrase for {locale}")

    def test_english_and_german_differ(self):
        """A localized disclosure that is not localized is a bug, not a default."""
        from utils.spoken_disclosure import disclosure_text

        self.assertNotEqual(disclosure_text(locale="en"), disclosure_text(locale="de"))

    def test_the_written_form_is_the_spoken_one(self):
        """Same source string, so the two cannot drift."""
        from utils.i18n import t
        from utils.spoken_disclosure import disclosure_text

        self.assertEqual(disclosure_text(locale="en"), t("disclosure.spoken", locale="en"))

    def test_the_cli_exposes_it(self):
        import subprocess
        import sys

        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", ".."))
        proc = subprocess.run(
            [sys.executable, "-m", "cli", "--disclosure-text", "--language", "de"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        from utils.spoken_disclosure import disclosure_text

        self.assertEqual(proc.stdout.strip(), disclosure_text(locale="de"))


# ---------------------------------------------------------------------------
# Finding 6 — the in-sample watermark was contingent on an extra.
# ---------------------------------------------------------------------------


class TestMarkingDependenciesAreNotOptional(unittest.TestCase):
    """Art. 50(2) marking must not depend on an extra nobody selected.

    soundfile and numpy are what the spread-spectrum watermark is built from.
    While they sat in [tts], a bare install could apply nothing but strippable
    metadata — and that install can still drive the CrispASR binary, so it was
    the one configuration able to emit audio with metadata-only marking.
    """

    @staticmethod
    def _pyproject():
        """Parsed, not grepped — a ']' inside a comment fooled the string scan."""
        try:
            import tomllib
        except ImportError:  # pragma: no cover - Python 3.9/3.10
            raise unittest.SkipTest("tomllib requires Python 3.11+")

        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "..", "pyproject.toml")
        with open(path, "rb") as fh:
            return tomllib.load(fh)

    def test_soundfile_and_numpy_are_base_dependencies(self):
        base = " ".join(self._pyproject()["project"]["dependencies"])
        for package in ("soundfile", "numpy"):
            self.assertIn(
                package,
                base,
                f"{package} must be a base dependency: the in-sample watermark "
                "is built from it, and Art. 50(2) has no 'unless an extra was "
                "selected' clause",
            )

    def test_the_declared_version_is_the_documented_one(self):
        """COMPLIANCE.md pins itself to a release; the pin has to be real."""
        version = self._pyproject()["project"]["version"]
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "..", "..", "COMPLIANCE.md"), encoding="utf-8") as fh:
            doc = fh.read()
        self.assertIn(f"Applies to Susurrus {version}", doc)

        import importlib

        pkg = importlib.import_module("__init__")
        self.assertEqual(pkg.__version__, version, "__init__.py is out of sync")
