"""Test EU AI Act compliance: provenance flags, watermark, C2PA."""

import unittest


class TestProvenanceCLIArgs(unittest.TestCase):
    """Test that CLI argparse accepts all provenance flags."""

    def test_i_have_rights_flag(self):
        import argparse

        from cli import _build_crispasr_kwargs

        ns = argparse.Namespace(
            backend="crispasr",
            i_have_rights=True,
            no_spoken_disclaimer=False,
            watermark_model=None,
            no_watermark=False,
            detect_watermark=None,
            c2pa_cert=None,
            c2pa_key=None,
            **self._defaults(),
        )
        kwargs = _build_crispasr_kwargs(ns)
        self.assertTrue(kwargs.get("i_have_rights"))

    def test_no_watermark_flag(self):
        import argparse

        from cli import _build_crispasr_kwargs

        ns = argparse.Namespace(
            backend="crispasr",
            no_watermark=True,
            i_have_rights=False,
            no_spoken_disclaimer=False,
            watermark_model=None,
            detect_watermark=None,
            c2pa_cert=None,
            c2pa_key=None,
            **self._defaults(),
        )
        kwargs = _build_crispasr_kwargs(ns)
        self.assertTrue(kwargs.get("no_watermark"))

    def test_c2pa_cert_key_flags(self):
        import argparse

        from cli import _build_crispasr_kwargs

        ns = argparse.Namespace(
            backend="crispasr",
            c2pa_cert="cert.pem",
            c2pa_key="key.pem",
            i_have_rights=False,
            no_spoken_disclaimer=False,
            watermark_model=None,
            no_watermark=False,
            detect_watermark=None,
            **self._defaults(),
        )
        kwargs = _build_crispasr_kwargs(ns)
        self.assertEqual(kwargs.get("c2pa_cert"), "cert.pem")
        self.assertEqual(kwargs.get("c2pa_key"), "key.pem")

    def test_watermark_model_flag(self):
        import argparse

        from cli import _build_crispasr_kwargs

        ns = argparse.Namespace(
            backend="crispasr",
            watermark_model="audioseal.gguf",
            i_have_rights=False,
            no_spoken_disclaimer=False,
            no_watermark=False,
            detect_watermark=None,
            c2pa_cert=None,
            c2pa_key=None,
            **self._defaults(),
        )
        kwargs = _build_crispasr_kwargs(ns)
        self.assertEqual(kwargs.get("watermark_model"), "audioseal.gguf")

    def test_all_provenance_flags_together(self):
        import argparse

        from cli import _build_crispasr_kwargs

        ns = argparse.Namespace(
            backend="crispasr",
            i_have_rights=True,
            no_spoken_disclaimer=True,
            watermark_model="audioseal.gguf",
            no_watermark=False,
            detect_watermark=None,
            c2pa_cert="cert.pem",
            c2pa_key="key.pem",
            **self._defaults(),
        )
        kwargs = _build_crispasr_kwargs(ns)
        self.assertTrue(kwargs["i_have_rights"])
        self.assertTrue(kwargs["no_spoken_disclaimer"])
        self.assertEqual(kwargs["watermark_model"], "audioseal.gguf")
        self.assertEqual(kwargs["c2pa_cert"], "cert.pem")
        self.assertEqual(kwargs["c2pa_key"], "key.pem")

    def _defaults(self):
        """Provide default values for all other _build_crispasr_kwargs fields."""
        return {
            "crispasr_backend": None,
            "vad": False,
            "split_on_punct": False,
            "temperature": None,
            "best_of": None,
            "beam_size": None,
            "seed": None,
            "max_new_tokens": None,
            "frequency_penalty": None,
            "prompt": None,
            "carry_initial_prompt": False,
            "auto_download": False,
            "prefix_text": None,
            "translate": False,
            "flash_attn": False,
            "no_gpu": False,
            "gpu_backend": None,
            "n_gpu_layers": None,
            "no_kv_offload": False,
            "vad_model": None,
            "vad_threshold": None,
            "vad_min_speech_ms": None,
            "vad_min_silence_ms": None,
            "vad_max_speech_s": None,
            "vad_pad_ms": None,
            "diarize": False,
            "diarize_method": None,
            "diarize_embedder": None,
            "diarize_cluster_threshold": None,
            "diarize_max_speakers": None,
            "diarize_speakers": False,
            "speaker_db_consent": False,
            "detect_language": False,
            "lid_backend": None,
            "lid_model": None,
            "aligner_model": None,
            "force_aligner": False,
            "text_file": None,
            "align_output": None,
            "align_format": None,
            "punc_model": None,
            "speaker_db": None,
            "enroll_speaker": None,
            "speaker_threshold": None,
            "titanet_model": None,
            "grammar": None,
            "grammar_rule": None,
            "grammar_penalty": None,
            "output_srt": False,
            "output_vtt": False,
            "output_json": False,
            "output_json_full": False,
            "output_csv": False,
            "output_lrc": False,
            "output_file": None,
            "mic": False,
            "live": False,
            "stream_step": None,
            "stream_length": None,
            "stream_json": False,
            "host": None,
            "port": None,
            "api_keys": None,
            "wyoming_port": None,
            "voice": None,
            "ref_text": None,
            "instruct": None,
            "codec_model": None,
            "tts_steps": None,
            "tts_play": False,
            "tts_play_device": None,
        }


class TestTTSBackendProvenance(unittest.TestCase):
    """Test that TTS backend accepts and uses provenance kwargs."""

    def test_no_watermark_kwarg(self):
        from workers.tts.backends.crispasr_tts_backend import CrispasrTTSBackend

        b = CrispasrTTSBackend(model_id="auto", no_watermark=True)
        self.assertTrue(b.no_watermark)

    def test_c2pa_kwargs(self):
        from workers.tts.backends.crispasr_tts_backend import CrispasrTTSBackend

        b = CrispasrTTSBackend(model_id="auto", c2pa_cert="cert.pem", c2pa_key="key.pem")
        self.assertEqual(b.c2pa_cert, "cert.pem")
        self.assertEqual(b.c2pa_key, "key.pem")

    def test_defaults_compliant(self):
        """Default TTS backend does not disable watermark or C2PA."""
        from workers.tts.backends.crispasr_tts_backend import CrispasrTTSBackend

        b = CrispasrTTSBackend(model_id="auto")
        self.assertFalse(b.no_watermark)
        self.assertFalse(b.i_have_rights)
        self.assertFalse(b.no_spoken_disclaimer)
        self.assertIsNone(b.c2pa_cert)
        self.assertIsNone(b.c2pa_key)

    def test_i_have_rights_required_pattern(self):
        """i_have_rights defaults to False — voice cloning gated by default."""
        from workers.tts.backends.crispasr_tts_backend import CrispasrTTSBackend

        b = CrispasrTTSBackend(model_id="auto")
        self.assertFalse(b.i_have_rights)
        b2 = CrispasrTTSBackend(model_id="auto", i_have_rights=True)
        self.assertTrue(b2.i_have_rights)


class TestProvenanceParamMap(unittest.TestCase):
    """Test PARAM_MAP has all EU AI Act compliance flags."""

    def test_all_provenance_keys(self):
        from workers.transcription.backends.crispasr_backend import PARAM_MAP

        required = {
            "i_have_rights": "--i-have-rights",
            "no_spoken_disclaimer": "--no-spoken-disclaimer",
            "no_watermark": "--no-watermark",
            "watermark_model": "--watermark-model",
            "detect_watermark": "--detect-watermark",
            "c2pa_cert": "--c2pa-cert",
            "c2pa_key": "--c2pa-key",
        }
        for key, flag in required.items():
            self.assertIn(key, PARAM_MAP, f"missing key: {key}")
            self.assertEqual(PARAM_MAP[key][0], flag, f"wrong flag for {key}")

    def test_no_watermark_is_bool(self):
        from workers.transcription.backends.crispasr_backend import PARAM_MAP

        self.assertIs(PARAM_MAP["no_watermark"][1], bool)

    def test_c2pa_cert_is_str(self):
        from workers.transcription.backends.crispasr_backend import PARAM_MAP

        self.assertIs(PARAM_MAP["c2pa_cert"][1], str)
        self.assertIs(PARAM_MAP["c2pa_key"][1], str)


if __name__ == "__main__":
    unittest.main()
