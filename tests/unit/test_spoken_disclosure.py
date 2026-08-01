"""Test the audible AI-disclosure prefix (EU AI Act Art. 50(4))."""

import os
import tempfile
import unittest
import wave

from utils.spoken_disclosure import (
    concat_wavs,
    disclosure_text,
    prepend_spoken_disclosure,
)
from workers.tts.backends.base import TTSBackend

DISCLOSURE_FRAMES = 100
CONTENT_FRAMES = 500


def _write_wav(path, frames, channels=1, width=2, rate=16000):
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * frames * channels)


def _frames(path):
    with wave.open(path, "rb") as w:
        return w.getnframes()


class FakeTTS(TTSBackend):
    """Writes a short file for the disclosure, a long one for content."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def synthesize(self, text, output_path, voice=None):
        self.calls.append((text, voice))
        is_disclosure = text == disclosure_text()
        _write_wav(output_path, DISCLOSURE_FRAMES if is_disclosure else CONTENT_FRAMES)
        return output_path


class TestDisclosureText(unittest.TestCase):
    def test_says_the_audio_is_ai_generated(self):
        text = disclosure_text("en").lower()
        self.assertIn("artificial intelligence", text)

    def test_localized(self):
        self.assertIn("künstlicher Intelligenz", disclosure_text("de"))
        self.assertNotEqual(disclosure_text("en"), disclosure_text("de"))

    def test_unknown_locale_falls_back_to_english(self):
        self.assertEqual(disclosure_text("xx"), disclosure_text("en"))


class TestConcatWavs(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.a = os.path.join(self._dir.name, "a.wav")
        self.b = os.path.join(self._dir.name, "b.wav")
        self.out = os.path.join(self._dir.name, "out.wav")

    def tearDown(self):
        self._dir.cleanup()

    def test_concatenates_frame_counts(self):
        _write_wav(self.a, 100)
        _write_wav(self.b, 500)
        self.assertTrue(concat_wavs(self.a, self.b, self.out))
        self.assertEqual(_frames(self.out), 600)

    def test_preserves_format(self):
        _write_wav(self.a, 100, channels=2, rate=22050)
        _write_wav(self.b, 500, channels=2, rate=22050)
        self.assertTrue(concat_wavs(self.a, self.b, self.out))
        with wave.open(self.out, "rb") as w:
            self.assertEqual(w.getnchannels(), 2)
            self.assertEqual(w.getframerate(), 22050)

    def test_refuses_sample_rate_mismatch(self):
        """Concatenating mismatched audio would produce audible corruption."""
        _write_wav(self.a, 100, rate=16000)
        _write_wav(self.b, 500, rate=22050)
        self.assertFalse(concat_wavs(self.a, self.b, self.out))
        self.assertFalse(os.path.exists(self.out))

    def test_refuses_channel_mismatch(self):
        _write_wav(self.a, 100, channels=1)
        _write_wav(self.b, 500, channels=2)
        self.assertFalse(concat_wavs(self.a, self.b, self.out))

    def test_refuses_missing_file(self):
        _write_wav(self.b, 500)
        self.assertFalse(concat_wavs(self.a, self.b, self.out))


class TestPrependSpokenDisclosure(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.out = os.path.join(self._dir.name, "out.wav")

    def tearDown(self):
        self._dir.cleanup()

    def test_prepends_and_grows_the_file(self):
        backend = FakeTTS()
        backend.synthesize("content", self.out)
        self.assertTrue(prepend_spoken_disclosure(backend, self.out))
        self.assertEqual(_frames(self.out), DISCLOSURE_FRAMES + CONTENT_FRAMES)

    def test_disclosure_precedes_the_content(self):
        """Order matters: a trailing disclosure is not a disclosure."""
        backend = FakeTTS()
        _write_wav(self.out, CONTENT_FRAMES)
        with wave.open(self.out, "rb") as w:
            content_bytes = w.readframes(w.getnframes())

        prepend_spoken_disclosure(backend, self.out)
        with wave.open(self.out, "rb") as w:
            merged = w.readframes(w.getnframes())
        self.assertTrue(merged.endswith(content_bytes))
        self.assertGreater(len(merged), len(content_bytes))

    def test_does_not_recurse(self):
        """Synthesizing the disclosure must not trigger another disclosure."""
        backend = FakeTTS()
        backend.synthesize("content", self.out)
        prepend_spoken_disclosure(backend, self.out)
        disclosure_calls = [c for c in backend.calls if c[0] == disclosure_text()]
        self.assertEqual(len(disclosure_calls), 1)

    def test_no_temp_files_left_behind(self):
        backend = FakeTTS()
        backend.synthesize("content", self.out)
        prepend_spoken_disclosure(backend, self.out)
        leftovers = [n for n in os.listdir(self._dir.name) if "tmp" in n]
        self.assertEqual(leftovers, [])

    def test_backend_failure_preserves_user_audio(self):
        """If the disclosure cannot be spoken, don't destroy the output."""

        class BrokenTTS(FakeTTS):
            def synthesize(self, text, output_path, voice=None):
                if text == disclosure_text():
                    raise RuntimeError("no voice available")
                return super().synthesize(text, output_path, voice)

        backend = BrokenTTS()
        backend.synthesize("content", self.out)
        self.assertFalse(prepend_spoken_disclosure(backend, self.out))
        self.assertEqual(_frames(self.out), CONTENT_FRAMES)

    def test_format_mismatch_preserves_user_audio(self):
        class MismatchedTTS(FakeTTS):
            def synthesize(self, text, output_path, voice=None):
                if text == disclosure_text():
                    _write_wav(output_path, DISCLOSURE_FRAMES, rate=44100)
                    return output_path
                return super().synthesize(text, output_path, voice)

        backend = MismatchedTTS()
        backend.synthesize("content", self.out)
        self.assertFalse(prepend_spoken_disclosure(backend, self.out))
        self.assertEqual(_frames(self.out), CONTENT_FRAMES)

    def test_skips_non_wav(self):
        self.assertFalse(prepend_spoken_disclosure(FakeTTS(), "out.mp3"))

    def test_skips_missing_file(self):
        self.assertFalse(prepend_spoken_disclosure(FakeTTS(), self.out))


class TestProvenanceIntegration(unittest.TestCase):
    """apply_provenance() must speak the disclosure only when cloning."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.out = os.path.join(self._dir.name, "out.wav")
        self.ref = os.path.join(self._dir.name, "ref.wav")
        _write_wav(self.ref, 10)

    def tearDown(self):
        self._dir.cleanup()

    def _run(self, **kwargs):
        backend = FakeTTS(**kwargs)
        backend.synthesize("content", self.out)
        return backend.apply_provenance(self.out), _frames(self.out)

    def test_cloning_gets_spoken_disclosure(self):
        result, frames = self._run(i_have_rights=True, reference_audio=self.ref)
        self.assertTrue(result["spoken"])
        self.assertEqual(frames, DISCLOSURE_FRAMES + CONTENT_FRAMES)

    def test_non_cloning_gets_no_spoken_disclosure(self):
        """A stock voice is not a deepfake; Art. 50(4) does not apply."""
        result, frames = self._run()
        self.assertFalse(result["spoken"])
        self.assertEqual(frames, CONTENT_FRAMES)

    def test_no_spoken_disclaimer_opt_out(self):
        result, frames = self._run(
            i_have_rights=True, reference_audio=self.ref, no_spoken_disclaimer=True
        )
        self.assertFalse(result["spoken"])
        self.assertEqual(frames, CONTENT_FRAMES)

    def test_marking_still_applies_when_disclosure_opted_out(self):
        """Opting out of the *audible* prefix keeps machine-readable marking."""
        result, _ = self._run(
            i_have_rights=True, reference_audio=self.ref, no_spoken_disclaimer=True
        )
        self.assertTrue(result["marker"])

    def test_full_opt_out_skips_disclosure_too(self):
        result, frames = self._run(
            i_have_rights=True,
            reference_audio=self.ref,
            accept_marking_responsibility=True,
        )
        self.assertTrue(result["opted_out"])
        self.assertFalse(result["spoken"])
        self.assertEqual(frames, CONTENT_FRAMES)


if __name__ == "__main__":
    unittest.main()
