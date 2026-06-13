"""Test CrispASR PARAM_MAP command building and multi-mode support."""

import unittest

from utils.crispasr_utils import find_crispasr

skip_no_crispasr = unittest.skipUnless(find_crispasr(), "crispasr binary not available")


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


class TestCrispASR071Sync(unittest.TestCase):
    """Coverage for the CrispASR 0.7.1 interface sync."""

    def _make_backend(self, **kwargs):
        from workers.transcription.backends.crispasr_backend import CrispasrBackend

        return CrispasrBackend(model_id="test.gguf", device="cpu", **kwargs)

    def test_parakeet_decoder_takes_value(self):
        # Regression: parakeet_decoder is a valued flag (ctc|tdt|maes), not bool.
        from workers.transcription.backends.crispasr_backend import PARAM_MAP

        self.assertIs(PARAM_MAP["parakeet_decoder"][1], str)
        b = self._make_backend(parakeet_decoder="tdt")
        cmd = ["crispasr"]
        b._append_params(cmd)
        idx = cmd.index("--parakeet-decoder")
        self.assertEqual(cmd[idx + 1], "tdt")

    def test_new_071_flags_present(self):
        from workers.transcription.backends.crispasr_backend import PARAM_MAP

        for key, flag in [
            ("hotwords_file", "--hotwords-file"),
            ("hotwords_boost", "--hotwords-boost"),
            ("g2p_dict", "--g2p-dict"),
            ("i_have_rights", "--i-have-rights"),
            ("no_spoken_disclaimer", "--no-spoken-disclaimer"),
            ("watermark_model", "--watermark-model"),
            ("detect_watermark", "--detect-watermark"),
            ("c2pa_cert", "--c2pa-cert"),
            ("tts_ref_asr", "--ref-asr"),
            ("dry_run_resolve", "--dry-run-resolve"),
        ]:
            self.assertIn(key, PARAM_MAP, f"missing PARAM_MAP key {key}")
            self.assertEqual(PARAM_MAP[key][0], flag)

    def test_provenance_bool_flags_emit(self):
        b = self._make_backend(i_have_rights=True, no_spoken_disclaimer=True)
        cmd = ["crispasr"]
        b._append_params(cmd)
        self.assertIn("--i-have-rights", cmd)
        self.assertIn("--no-spoken-disclaimer", cmd)

    def test_hotwords_value_flags_emit(self):
        b = self._make_backend(hotwords="Tokyo,CrispASR", hotwords_boost=3.0, g2p_dict="olaph")
        cmd = ["crispasr"]
        b._append_params(cmd)
        self.assertIn("--hotwords", cmd)
        self.assertEqual(cmd[cmd.index("--hotwords-boost") + 1], "3.0")
        self.assertEqual(cmd[cmd.index("--g2p-dict") + 1], "olaph")

    def test_server_and_s2s_flags(self):
        b = self._make_backend(ws_port=8081, no_warmup=True, s2s=True, stream_keep=200)
        cmd = ["crispasr"]
        b._append_params(cmd)
        self.assertEqual(cmd[cmd.index("--ws-port") + 1], "8081")
        self.assertIn("--no-warmup", cmd)
        self.assertIn("--s2s", cmd)
        self.assertEqual(cmd[cmd.index("--stream-keep") + 1], "200")


class TestCrispASRRegistrySync(unittest.TestCase):
    """The config registries must use valid CrispASR 0.7.1 backend names."""

    def test_no_stale_backend_names(self):
        import config

        # These were invalid --backend values (resolved to 'unresolved').
        self.assertNotIn("vibevoice-asr", config.CRISPASR_SUB_BACKENDS)
        self.assertNotIn("chatterbox-tts", config.CRISPASR_TTS_BACKENDS)
        self.assertNotIn("vibevoice-tts", config.CRISPASR_TTS_BACKENDS)

    def test_new_backends_added(self):
        import config

        for name in ("mega-asr", "moss-audio", "sensevoice", "hubert"):
            self.assertIn(name, config.CRISPASR_SUB_BACKENDS)
        for name in ("zonos", "bark", "dia", "melotts", "piper"):
            self.assertIn(name, config.CRISPASR_TTS_BACKENDS)
        self.assertIn("m2m100-wmt21", config.CRISPASR_TRANSLATION_BACKENDS)

    def test_model_maps_have_keys_for_lists(self):
        import config

        for b in config.CRISPASR_SUB_BACKENDS:
            self.assertIn(
                f"crispasr:{b}", config.BACKEND_MODEL_MAP, f"BACKEND_MODEL_MAP missing crispasr:{b}"
            )
        for b in config.CRISPASR_TTS_BACKENDS:
            self.assertIn(
                f"crispasr:{b}", config.TTS_BACKEND_MAP, f"TTS_BACKEND_MAP missing crispasr:{b}"
            )


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

    @skip_no_crispasr
    def test_auto_model_gets_auto_download(self):
        from workers.transcription.backends.crispasr_backend import CrispasrBackend

        b = CrispasrBackend(model_id="auto", device="cpu")
        cmd, _ = b._build_base_cmd()
        self.assertIn("--auto-download", cmd)

    @skip_no_crispasr
    def test_auto_quant_model(self):
        from workers.transcription.backends.crispasr_backend import CrispasrBackend

        b = CrispasrBackend(model_id="auto:q8_0", device="cpu")
        cmd, _ = b._build_base_cmd()
        self.assertIn("auto:q8_0", cmd)


if __name__ == "__main__":
    unittest.main()
