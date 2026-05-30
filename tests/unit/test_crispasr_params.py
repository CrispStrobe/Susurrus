"""Test CrispASR PARAM_MAP command building and multi-mode support."""

import unittest


class TestCrispASRParamMap(unittest.TestCase):
    """Test that PARAM_MAP correctly maps kwargs to CLI flags."""

    def _make_backend(self, **kwargs):
        from workers.transcription.backends.crispasr_backend import CrispasrBackend

        return CrispasrBackend(model_id="test.gguf", device="cpu", **kwargs)

    def test_param_map_has_entries(self):
        from workers.transcription.backends.crispasr_backend import PARAM_MAP

        self.assertGreater(len(PARAM_MAP), 100)

    def test_bool_flag_appended(self):
        b = self._make_backend(vad=True, split_on_punct=True)
        cmd = ["crispasr"]
        b._append_params(cmd)
        self.assertIn("--vad", cmd)
        self.assertIn("--split-on-punct", cmd)

    def test_bool_false_not_appended(self):
        b = self._make_backend(vad=False)
        cmd = ["crispasr"]
        b._append_params(cmd)
        self.assertNotIn("--vad", cmd)

    def test_str_param_with_value(self):
        b = self._make_backend(crispasr_backend="parakeet")
        cmd = ["crispasr"]
        b._append_params(cmd)
        idx = cmd.index("--backend")
        self.assertEqual(cmd[idx + 1], "parakeet")

    def test_int_param(self):
        b = self._make_backend(seed=42, beam_size=5)
        cmd = ["crispasr"]
        b._append_params(cmd)
        idx = cmd.index("--seed")
        self.assertEqual(cmd[idx + 1], "42")
        idx = cmd.index("--beam-size")
        self.assertEqual(cmd[idx + 1], "5")

    def test_float_param(self):
        b = self._make_backend(temperature=0.8)
        cmd = ["crispasr"]
        b._append_params(cmd)
        idx = cmd.index("-tp")
        self.assertEqual(cmd[idx + 1], "0.8")

    def test_none_param_skipped(self):
        b = self._make_backend()
        cmd = ["crispasr"]
        b._append_params(cmd)
        # No params should be added for default empty kwargs
        self.assertEqual(cmd, ["crispasr"])

    def test_multiple_params_combined(self):
        b = self._make_backend(
            crispasr_backend="canary",
            vad=True,
            temperature=0.5,
            diarize=True,
            diarize_method="pyannote",
            auto_download=True,
        )
        cmd = ["crispasr"]
        b._append_params(cmd)
        self.assertIn("--backend", cmd)
        self.assertIn("--vad", cmd)
        self.assertIn("-tp", cmd)
        self.assertIn("--diarize", cmd)
        self.assertIn("--diarize-method", cmd)
        self.assertIn("--auto-download", cmd)

    def test_tts_params(self):
        b = self._make_backend(tts_text="hello", tts_output="out.wav", tts_voice="voice.gguf")
        cmd = ["crispasr"]
        b._append_params(cmd)
        self.assertIn("--tts", cmd)
        self.assertIn("--tts-output", cmd)
        self.assertIn("--voice", cmd)

    def test_translation_params(self):
        b = self._make_backend(text_input="hello", tr_source_lang="en", tr_target_lang="de")
        cmd = ["crispasr"]
        b._append_params(cmd)
        self.assertIn("--text", cmd)
        self.assertIn("--tr-sl", cmd)
        self.assertIn("--tr-tl", cmd)

    def test_streaming_params(self):
        b = self._make_backend(stream=True, mic=True, stream_json=True, stream_step=3000)
        cmd = ["crispasr"]
        b._append_params(cmd)
        self.assertIn("--stream", cmd)
        self.assertIn("--mic", cmd)
        self.assertIn("--stream-json", cmd)
        self.assertIn("--stream-step", cmd)

    def test_server_params(self):
        b = self._make_backend(server=True, host="0.0.0.0", port=9090)
        cmd = ["crispasr"]
        b._append_params(cmd)
        self.assertIn("--server", cmd)
        self.assertIn("--host", cmd)
        self.assertIn("--port", cmd)


class TestBackendSubNotation(unittest.TestCase):
    """Test crispasr:<subbackend> notation in registry."""

    def test_subbackend_parsed(self):
        from workers.transcription.backends import get_backend

        b = get_backend("crispasr:parakeet", model_id="auto", device="cpu")
        self.assertEqual(b.__class__.__name__, "CrispasrBackend")
        self.assertEqual(b.extra_kwargs.get("crispasr_backend"), "parakeet")

    def test_plain_crispasr_works(self):
        from workers.transcription.backends import get_backend

        b = get_backend("crispasr", model_id="auto", device="cpu")
        self.assertEqual(b.__class__.__name__, "CrispasrBackend")

    def test_unknown_backend_raises(self):
        from workers.transcription.backends import get_backend

        with self.assertRaises(ValueError):
            get_backend("nonexistent-backend")


class TestAutoModel(unittest.TestCase):
    """Test auto model support."""

    def test_auto_model_gets_auto_download(self):
        from workers.transcription.backends.crispasr_backend import CrispasrBackend

        b = CrispasrBackend(model_id="auto", device="cpu")
        try:
            cmd, _ = b._build_base_cmd()
        except FileNotFoundError:
            self.skipTest("crispasr binary not available")
        self.assertIn("--auto-download", cmd)

    def test_auto_quant_model(self):
        from workers.transcription.backends.crispasr_backend import CrispasrBackend

        b = CrispasrBackend(model_id="auto:q8_0", device="cpu")
        try:
            cmd, _ = b._build_base_cmd()
        except FileNotFoundError:
            self.skipTest("crispasr binary not available")
        self.assertIn("auto:q8_0", cmd)


if __name__ == "__main__":
    unittest.main()
