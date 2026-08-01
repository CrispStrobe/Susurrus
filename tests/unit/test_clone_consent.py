"""Test the voice-cloning consent gate (EU AI Act Art. 50(4))."""

import os
import tempfile
import unittest
from unittest import mock

from workers.tts.backends.base import TTSBackend


class Dummy(TTSBackend):
    def synthesize(self, text, output_path, voice=None):
        return output_path


class TestRequireCloneConsent(unittest.TestCase):
    def test_blocks_cloning_without_attestation(self):
        with self.assertRaises(PermissionError) as ctx:
            Dummy().require_clone_consent("/ref.wav")
        self.assertIn("i-have-rights", str(ctx.exception))

    def test_allows_cloning_with_attestation(self):
        Dummy(i_have_rights=True).require_clone_consent("/ref.wav")  # must not raise

    def test_no_gate_when_not_cloning(self):
        Dummy().require_clone_consent(None)  # must not raise
        Dummy().require_clone_consent("")  # must not raise

    def test_attestation_defaults_off(self):
        """Consent must never be the default — it has to be an explicit act."""
        self.assertFalse(Dummy().kwargs.get("i_have_rights", False))


class TestChatterboxCloneGate(unittest.TestCase):
    """Chatterbox reaches audio_prompt_path from two directions; gate both."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.ref = os.path.join(self._dir.name, "ref.wav")
        with open(self.ref, "wb") as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVE")

    def tearDown(self):
        self._dir.cleanup()

    def _backend(self, **kwargs):
        from workers.tts.backends.chatterbox_tts_backend import ChatterboxTTSBackend

        return ChatterboxTTSBackend(**kwargs)

    def test_blocks_clone_via_voice_argument(self):
        """`--voice /path/to/someone.wav` is a cloning route."""
        with mock.patch.dict("sys.modules", {"chatterbox": mock.MagicMock()}):
            with self.assertRaises(PermissionError):
                self._backend().synthesize("hi", "out.wav", voice=self.ref)

    def test_blocks_clone_via_reference_audio_kwarg(self):
        with mock.patch.dict("sys.modules", {"chatterbox": mock.MagicMock()}):
            with self.assertRaises(PermissionError):
                self._backend(reference_audio=self.ref).synthesize("hi", "out.wav")

    def test_gate_precedes_model_download(self):
        """Refusal must happen before any model is fetched or loaded."""
        fake_tts = mock.MagicMock()
        modules = {
            "chatterbox": mock.MagicMock(),
            "chatterbox.tts": mock.MagicMock(ChatterboxTTS=fake_tts),
        }
        with mock.patch.dict("sys.modules", modules):
            with self.assertRaises(PermissionError):
                self._backend().synthesize("hi", "out.wav", voice=self.ref)
        fake_tts.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
