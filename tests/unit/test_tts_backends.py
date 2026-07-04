"""Test TTS backend registry and interface compliance."""

import unittest


class TestTTSBase(unittest.TestCase):
    """Test TTSBackend ABC."""

    def test_base_class_has_abstractmethod(self):
        from workers.tts.backends.base import TTSBackend

        self.assertIn("synthesize", TTSBackend.__abstractmethods__)

    def test_cannot_instantiate_base(self):
        from workers.tts.backends.base import TTSBackend

        with self.assertRaises(TypeError):
            TTSBackend()


class TestTTSRegistry(unittest.TestCase):
    """Test TTS backend registry."""

    def test_crispasr_tts_backend(self):
        from workers.tts.backends import get_tts_backend

        b = get_tts_backend("crispasr:kokoro", model_id="auto", device="cpu")
        self.assertEqual(b.__class__.__name__, "CrispasrTTSBackend")
        self.assertEqual(b.crispasr_backend, "kokoro")

    def test_unknown_backend_raises(self):
        from workers.tts.backends import get_tts_backend

        with self.assertRaises(ValueError):
            get_tts_backend("nonexistent-tts")

    def test_edge_tts_importable(self):
        """Edge TTS backend class can be imported (not necessarily run)."""
        from workers.tts.backends.edge_tts_backend import EdgeTTSBackend

        b = EdgeTTSBackend()
        voices = b.list_voices()
        self.assertGreater(len(voices), 0)
        self.assertIn("de-DE-KatjaNeural", voices)

    def test_piper_importable(self):
        from workers.tts.backends.piper_tts_backend import PiperTTSBackend

        b = PiperTTSBackend()
        voices = b.list_voices()
        self.assertGreater(len(voices), 0)

    def test_kokoro_onnx_importable(self):
        from workers.tts.backends.kokoro_onnx_tts_backend import KokoroOnnxTTSBackend

        b = KokoroOnnxTTSBackend()
        voices = b.list_voices()
        self.assertGreater(len(voices), 0)


class TestTranslationBase(unittest.TestCase):
    """Test TranslationBackend ABC."""

    def test_base_class_has_abstractmethod(self):
        from workers.translation.backends.base import TranslationBackend

        self.assertIn("translate", TranslationBackend.__abstractmethods__)

    def test_crispasr_translation_importable(self):
        from workers.translation.backends.crispasr_translation_backend import (
            CrispasrTranslationBackend,
        )

        b = CrispasrTranslationBackend(model_id="auto")
        langs = b.list_languages()
        self.assertGreater(len(langs), 10)


class TestTextExtraction(unittest.TestCase):
    """Test text extraction utilities."""

    def test_extract_txt(self):
        import os
        import tempfile

        from utils.text_extraction import extract_text_from_txt

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world")
            f.flush()
            result = extract_text_from_txt(f.name)
        os.unlink(f.name)
        self.assertEqual(result, "Hello world")

    def test_extract_dispatcher(self):
        from utils.text_extraction import extract_text

        with self.assertRaises(ValueError):
            extract_text("file.xyz")

    def test_extract_txt_via_dispatcher(self):
        import os
        import tempfile

        from utils.text_extraction import extract_text

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            f.flush()
            result = extract_text(f.name)
        os.unlink(f.name)
        self.assertEqual(result, "Test content")


class TestCrispasrTTSBackendKwargs(unittest.TestCase):
    """Test CrispASR TTS backend accepts new v0.8.0 kwargs."""

    def test_tts_play_kwarg(self):
        from workers.tts.backends.crispasr_tts_backend import CrispasrTTSBackend

        b = CrispasrTTSBackend(model_id="auto", tts_play=True, tts_play_device=1)
        self.assertTrue(b.tts_play)
        self.assertEqual(b.tts_play_device, 1)

    def test_tts_play_default_false(self):
        from workers.tts.backends.crispasr_tts_backend import CrispasrTTSBackend

        b = CrispasrTTSBackend(model_id="auto")
        self.assertFalse(b.tts_play)
        self.assertIsNone(b.tts_play_device)

    def test_mini_omni2_tts_backend(self):
        from workers.tts.backends import get_tts_backend

        b = get_tts_backend("crispasr:mini-omni2", model_id="auto", device="cpu")
        self.assertEqual(b.__class__.__name__, "CrispasrTTSBackend")
        self.assertEqual(b.crispasr_backend, "mini-omni2")


class TestConfigMaps(unittest.TestCase):
    """Test configuration maps."""

    def test_backend_model_map_has_crispasr_subs(self):
        from config import BACKEND_MODEL_MAP

        self.assertIn("crispasr:parakeet", BACKEND_MODEL_MAP)
        self.assertIn("crispasr:canary", BACKEND_MODEL_MAP)
        self.assertIn("crispasr:m2m100", BACKEND_MODEL_MAP)

    def test_tts_backend_map_has_entries(self):
        from config import TTS_BACKEND_MAP

        self.assertIn("edge-tts", TTS_BACKEND_MAP)
        self.assertIn("crispasr:kokoro", TTS_BACKEND_MAP)
        self.assertIn("piper", TTS_BACKEND_MAP)

    def test_default_model_for_crispasr_sub(self):
        from config import get_default_model_for_backend

        self.assertEqual(get_default_model_for_backend("crispasr:parakeet"), "auto")
        self.assertEqual(get_default_model_for_backend("crispasr"), "auto")

    def test_080_backends_in_maps(self):
        from config import BACKEND_MODEL_MAP, TTS_BACKEND_MAP

        self.assertIn("crispasr:nemotron", BACKEND_MODEL_MAP)
        self.assertIn("crispasr:mini-omni2", BACKEND_MODEL_MAP)
        self.assertIn("crispasr:mini-omni2", TTS_BACKEND_MAP)
        self.assertIn("crispasr:vibevoice-1.5b", TTS_BACKEND_MAP)

    def test_087_tts_backends_in_maps(self):
        from config import TTS_BACKEND_MAP

        for name in ("crispasr:tada", "crispasr:dots-tts", "crispasr:bananamind-tts"):
            self.assertIn(name, TTS_BACKEND_MAP, f"TTS_BACKEND_MAP missing {name}")

    def test_087_asr_backends_in_maps(self):
        from config import BACKEND_MODEL_MAP

        for name in (
            "crispasr:ark-asr",
            "crispasr:higgs-stt",
            "crispasr:moss-transcribe",
            "crispasr:gemma4-e4b",
            "crispasr:parakeet-ctc-ja",
            "crispasr:reazonspeech",
        ):
            self.assertIn(name, BACKEND_MODEL_MAP, f"BACKEND_MODEL_MAP missing {name}")


if __name__ == "__main__":
    unittest.main()
