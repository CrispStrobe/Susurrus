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


class TestClassificationTable(unittest.TestCase):
    """The table is evidence, so it has to stay answerable."""

    def test_every_value_is_legal(self):
        for key, value in si.BACKEND_SPEAKER_IDENTITY.items():
            self.assertIn(value, si.SPEAKER_IDENTITY_VALUES, f"{key} -> {value!r}")

    def test_keys_are_lowercase(self):
        """resolve() lowercases its input, so an uppercase key never matches."""
        for key in si.BACKEND_SPEAKER_IDENTITY:
            self.assertEqual(key, key.lower(), f"unreachable table entry: {key!r}")

    def test_entries_name_real_backends(self):
        """A typo'd key is a silent no-op that reads like a classification."""
        from config import TTS_BACKEND_MAP

        known = {name.lower() for name in TTS_BACKEND_MAP}
        # 'speecht5'/'piper' etc. are Python-native backend ids that are not
        # keys of TTS_BACKEND_MAP; only check the crispasr: ones, which are.
        for key in si.BACKEND_SPEAKER_IDENTITY:
            if key.startswith("crispasr:"):
                self.assertIn(key, known, f"table classifies a backend that does not exist: {key}")

    def test_unlisted_backends_warn_rather_than_pass_silently(self):
        """Susurrus exposes far more backends than are classified.

        That is acceptable — 'unknown' is an honest answer — but it must be a
        *loud* one, or an unclassified real-person voice ships undisclosed and
        nothing says so.
        """
        from config import TTS_BACKEND_MAP

        unlisted = [n for n in TTS_BACKEND_MAP if n.lower() not in si.BACKEND_SPEAKER_IDENTITY]
        self.assertTrue(unlisted, "fixture assumption: some backends are unclassified")

        si._reset_warnings_for_tests()
        with mock.patch.object(si.logger, "warning") as warn:
            identity = si.resolve_speaker_identity(backend=unlisted[0])
            si.requires_spoken_disclosure(False, identity, unlisted[0])
        self.assertEqual(identity, "unknown")
        self.assertEqual(warn.call_count, 1)


class TestModelAwareVerdicts(unittest.TestCase):
    """One backend can serve checkpoints with different answers.

    Every case here is a classification this project got wrong first time by
    reasoning from a name instead of from a model card, and that CrispASR's
    researched table corrected.
    """

    def setUp(self):
        si._reset_warnings_for_tests()

    def test_kokoro_english_is_synthetic_but_the_hui_finetune_is_not(self):
        """CONFLICT held at unknown rather than inheriting either neighbour.

        The German backbone is trained on HUI-Audio-Corpus-German, whose
        narrators are the same named people cited when marking FastPitch German
        real_person. Whether a style vector derived from it is recognisably one
        of them is unanswered, so it is not answered.
        """
        self.assertEqual(
            si.resolve_speaker_identity(backend="crispasr:kokoro", model="kokoro-82m-q8_0.gguf"),
            "synthetic",
        )
        self.assertEqual(
            si.resolve_speaker_identity(
                backend="crispasr:kokoro", model="kokoro-de-hui-base-q8_0.gguf"
            ),
            "unknown",
        )

    def test_orpheus_checkpoints_differ(self):
        self.assertEqual(
            si.resolve_speaker_identity(
                backend="crispasr:orpheus", model="kartoffel-orpheus-de-natural-q8_0.gguf"
            ),
            "real_person",
        )
        self.assertEqual(
            si.resolve_speaker_identity(backend="crispasr:orpheus", model="orpheus-3b-0.1-ft.gguf"),
            "unknown",
        )

    def test_a_renamed_checkpoint_fails_safe(self):
        """Filename matching is allowed only because its failure is safe.

        A rename must turn a known answer back into a question, never turn
        real_person into synthetic.
        """
        self.assertEqual(
            si.resolve_speaker_identity(backend="crispasr:orpheus", model="my-copy.gguf"),
            "unknown",
        )

    def test_a_name_only_becomes_a_verdict_once_the_card_agrees(self):
        """Held at unknown while only the repo name said "synthetic".

        The provider's card has since been read — "trained on synthetic German
        speech" — so the verdict is now synthetic on evidence that happens to
        agree with the name. The name was never the evidence and still isn't;
        this asserts the endpoint, and the comment beside the entry carries
        the reason it moved.
        """
        self.assertEqual(
            si.resolve_speaker_identity(backend="crispasr:kartoffel-orpheus-de-synthetic"),
            "synthetic",
        )

    def test_a_sibling_projects_verdict_does_not_port_across_weights(self):
        """It did not port — but reading the right card settled it anyway.

        crispasr:fastpitch is NVIDIA English, not the German NeMo model, so a
        sibling project's verdict for the latter was correctly refused. The
        English card then turned out to say "trained on LJSpeech": 13,100 clips
        of one LibriVox narrator, Linda Johnson. Same answer, arrived at from
        the evidence for *this* checkpoint.
        """
        self.assertEqual(si.resolve_speaker_identity(backend="crispasr:fastpitch"), "real_person")

    def test_operator_supplied_speaker_cannot_have_a_backend_verdict(self):
        """crispasr:speecht5 takes its x-vector from --voice, per invocation.

        The Python-native backend is different: it bakes in CMU ARCTIC
        speaker_idx 7306, so there the answer is knowable.
        """
        self.assertEqual(si.resolve_speaker_identity(backend="crispasr:speecht5"), "unknown")
        self.assertEqual(si.resolve_speaker_identity(backend="speecht5"), "real_person")

    def test_model_rules_are_wellformed(self):
        for backend, needle, identity in si.MODEL_RULES:
            self.assertEqual(backend, backend.lower())
            self.assertEqual(needle, needle.lower())
            self.assertIn(identity, si.SPEAKER_IDENTITY_VALUES)

    def test_model_reaches_the_pipeline(self):
        import os
        import tempfile
        import wave

        from utils.provenance import apply_provenance

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "o.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(b"\x00\x01" * 12000)

        result = apply_provenance(
            path, model="kokoro-de-hui-base-q8_0.gguf", speaker_backend="crispasr:kokoro"
        )
        self.assertEqual(
            result["speaker_identity"],
            "unknown",
            "the loaded checkpoint did not reach the verdict",
        )


class TestPerVoiceOverrides(unittest.TestCase):
    """A backend-level answer is wrong for a model with mixed voices.

    SauerkrautTTS is the known case — Tom and Anna are studio recordings of
    people, Max and Lena are not. Susurrus has no mixed backend today, so the
    table is empty; the lookup exists so the alternative isn't classifying a
    mixed model by its riskiest voice and disclosing over the rest.
    """

    def setUp(self):
        si._reset_warnings_for_tests()
        self._saved = dict(si.VOICE_SPEAKER_IDENTITY)

    def tearDown(self):
        si.VOICE_SPEAKER_IDENTITY.clear()
        si.VOICE_SPEAKER_IDENTITY.update(self._saved)

    def test_the_shipped_german_kokoro_packs_resolve(self):
        """The mixed case is real now, not hypothetical.

        The kokoro-de-hui base is documented speaker-neutral, so the voice a
        listener hears is the voicepack's. ``df_eva`` and ``dm_bernd`` carry the
        names of two documented HUI-Audio-Corpus-German narrators — the corpus
        this backbone is trained on — while ``df_victoria`` comes from a
        different source with no published provenance.
        """
        for voice in ("df_eva", "dm_bernd"):
            self.assertEqual(
                si.resolve_speaker_identity(
                    backend="crispasr:kokoro", voice=voice, model="kokoro-de-hui-base-q8_0.gguf"
                ),
                "real_person",
                f"{voice} matches a named narrator of the documented training corpus",
            )
        # Person-shaped name, synthetic answer — on the base's documented
        # training data, not on the name.
        self.assertEqual(
            si.resolve_speaker_identity(
                backend="crispasr:kokoro", voice="df_victoria", model="kokoro-de-hui-base-q8_0.gguf"
            ),
            "synthetic",
        )

    def test_every_per_voice_verdict_cites_its_evidence(self):
        """The rule is about what a verdict rests on, not which value it takes.

        An earlier version of this test simply forbade ``synthetic`` per voice,
        on the grounds that a name-derived synthetic silently removes a
        disclosure. That was the right instinct and the wrong rule: df_victoria
        and dm_martin *are* synthetic, on the provider's evidence that their
        base was "trained entirely on synthetic (TTS-generated) audio" — a
        documented fact that happens to point the same way the name does.
        Forbidding the value would have rejected the evidence along with the
        guess. Requiring the evidence is what actually separates them.
        """
        for key, entry in si.VOICE_SPEAKER_IDENTITY.items():
            self.assertIsInstance(entry, tuple, f"{key} has no evidence attached")
            identity, evidence = entry
            self.assertIn(identity, si.SPEAKER_IDENTITY_VALUES, key)
            self.assertTrue(
                evidence and len(evidence) > 20,
                f"{key} -> {identity!r} cites no evidence; a verdict with no "
                "source is a guess wearing a table entry's clothes",
            )

    def test_per_voice_entry_beats_the_backend(self):
        si.VOICE_SPEAKER_IDENTITY[("kokoro-onnx", "tom")] = (
            "real_person",
            "test fixture with stated evidence",
        )
        self.assertEqual(
            si.resolve_speaker_identity(backend="kokoro-onnx", voice="Tom"), "real_person"
        )
        self.assertEqual(si.resolve_speaker_identity(backend="kokoro-onnx"), "synthetic")

    def test_override_still_beats_per_voice(self):
        si.VOICE_SPEAKER_IDENTITY[("kokoro-onnx", "tom")] = (
            "real_person",
            "test fixture with stated evidence",
        )
        self.assertEqual(
            si.resolve_speaker_identity(backend="kokoro-onnx", voice="tom", override="synthetic"),
            "synthetic",
        )

    def test_voice_reaches_the_pipeline(self):
        import os
        import tempfile
        import wave

        from utils.provenance import apply_provenance

        si.VOICE_SPEAKER_IDENTITY[("kokoro-onnx", "tom")] = (
            "real_person",
            "test fixture with stated evidence",
        )
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "o.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(b"\x00\x01" * 12000)

        class Speaking:
            def synthesize(self, text, output_path, voice=None):
                import shutil

                shutil.copy(path, output_path)
                return output_path

        result = apply_provenance(
            path,
            options={"voice": "tom"},
            backend=Speaking(),
            speaker_backend="kokoro-onnx",
        )
        self.assertEqual(result["speaker_identity"], "real_person")
        self.assertTrue(result["spoken_required"])


class TestVoiceBankCloneGate(unittest.TestCase):
    """A clone that never touches the filesystem still has to be gated.

    cosyvoice3 and kugelaudio keep their voices in a bundle beside the model,
    and ``--voice`` selects one by name. The gate resolved that bare name to no
    file, concluded "preset", and let a zero-shot clone through with no rights
    attestation and no Art. 50(4) disclosure — while ``--voice victim.wav`` on
    the same backend *was* gated, which is why it looked covered. Found by
    CrispASR in its own gate; Susurrus had the identical pattern and ships both
    affected backends.
    """

    def _backend(self, name, **kwargs):
        from workers.tts.backends.base import TTSBackend

        class Dummy(TTSBackend):
            def synthesize(self, text, output_path, voice=None):
                return output_path

        return Dummy(model_id="auto", tts_backend_name=name, **kwargs)

    def test_bank_selection_counts_as_cloning(self):
        backend = self._backend("crispasr:cosyvoice3-tts")
        self.assertTrue(backend.is_cloning("some_speaker"))

    def test_bank_selection_without_attestation_is_refused(self):
        backend = self._backend("crispasr:cosyvoice3-tts")
        with self.assertRaises(PermissionError):
            backend.require_clone_consent(backend.resolve_reference_audio("some_speaker"))

    def test_bank_selection_with_attestation_is_allowed(self):
        backend = self._backend("crispasr:kugelaudio", i_have_rights=True)
        backend.require_clone_consent(backend.resolve_reference_audio("some_speaker"))

    def test_a_preset_backend_is_unaffected(self):
        """Over-gating every named voice everywhere would be its own bug."""
        backend = self._backend("crispasr:kokoro")
        self.assertFalse(backend.is_cloning("af_heart"))
        backend.require_clone_consent(backend.resolve_reference_audio("af_heart"))

    def test_no_voice_selected_is_not_cloning(self):
        self.assertFalse(self._backend("crispasr:cosyvoice3-tts").is_cloning(None))

    def test_the_disclosure_synthesis_does_not_recurse(self):
        """The guard that stops a disclosure being spoken in the cloned voice."""
        backend = self._backend("crispasr:cosyvoice3-tts")
        backend._synthesizing_disclosure = True
        self.assertIsNone(backend.resolve_reference_audio("some_speaker"))

    def test_the_transcription_route_gates_it_too(self):
        """CrispasrBackend has its own resolver and needed the same fix."""
        from workers.transcription.backends.crispasr_backend import CrispasrBackend

        backend = CrispasrBackend(
            model_id="auto", device="cpu", tts_backend_name="crispasr:cosyvoice3-tts"
        )
        self.assertEqual(backend.resolve_reference_audio("some_speaker"), "some_speaker")
        with self.assertRaises(PermissionError):
            backend.require_clone_consent(backend.resolve_reference_audio("some_speaker"))

    def test_the_cli_tells_the_backend_its_name(self):
        """Without the name the bank check can never fire on this route."""
        import inspect

        import cli

        source = inspect.getsource(cli._run_tts)
        crispasr_branch = source.split('if tts_backend.startswith("crispasr")')[1]
        self.assertIn("tts_backend_name", crispasr_branch)


class TestGuiControl(unittest.TestCase):
    """The CLI could answer the question; the GUI could not."""

    def test_widget_offers_the_three_values_plus_the_default(self):
        source = open(
            __file__.replace("tests/unit/test_speaker_identity.py", "gui/widgets/tts_settings.py"),
            encoding="utf-8",
        ).read()
        self.assertIn("speaker_identity", source)
        for value in ("real_person", "synthetic", "unknown"):
            self.assertIn(f'"{value}"', source)

    def test_main_window_forwards_the_selection(self):
        source = open(
            __file__.replace("tests/unit/test_speaker_identity.py", "gui/main_window.py"),
            encoding="utf-8",
        ).read()
        handler = source.split("def start_synthesis")[1].split("\n    def ")[0]
        self.assertIn("speaker_identity", handler)

    def test_labels_exist_in_both_languages(self):
        from utils.i18n import t

        for locale in ("en", "de"):
            for key in (
                "label.speaker_identity",
                "opt.speaker_identity_default",
                "opt.speaker_identity_real",
                "tip.speaker_identity",
            ):
                self.assertTrue(t(key, locale=locale), f"{key} missing for {locale}")


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
