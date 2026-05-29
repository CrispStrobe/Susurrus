# workers/tts/backends/kokoro_onnx_tts_backend.py
"""Kokoro ONNX TTS backend — fast local speech synthesis.

License: Kokoro is Apache 2.0.
"""

import logging
import os

import numpy as np

from .base import TTSBackend


class KokoroOnnxTTSBackend(TTSBackend):
    """Local TTS using Kokoro ONNX models.

    Requires: ``pip install kokoro-onnx``
    """

    def synthesize(self, text, output_path="tts_output.wav", voice=None):
        try:
            from kokoro_onnx import Kokoro
        except ImportError:
            raise ImportError(
                "kokoro-onnx is required for the Kokoro ONNX TTS backend. "
                "Install with: pip install kokoro-onnx"
            )

        voice_id = voice or self.kwargs.get("voice") or "af_heart"
        speed = float(self.kwargs.get("speed", 1.0))
        lang = self.language or "en-us"

        kokoro = Kokoro(
            self.kwargs.get("model_path", "kokoro-v1.0.onnx"),
            self.kwargs.get("voices_path", "voices-v1.0.bin"),
        )

        samples, sample_rate = kokoro.create(text, voice=voice_id, speed=speed, lang=lang)

        # Write WAV
        import soundfile as sf
        sf.write(output_path, samples, sample_rate)

        logging.info(f"Kokoro ONNX TTS output: {output_path}")
        return output_path

    def list_voices(self):
        return [
            "af_heart", "af_sky", "af_bella", "af_nicole", "af_sarah",
            "am_adam", "am_michael",
            "bf_emma", "bf_isabella",
            "bm_george", "bm_lewis",
        ]
