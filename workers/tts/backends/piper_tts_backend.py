# workers/tts/backends/piper_tts_backend.py
"""Piper TTS backend — fast local ONNX-based speech synthesis.

License: Piper is MIT-licensed.
Voices are licensed individually (mostly CC-BY/CC-BY-SA).
"""

import logging
import os

from .base import TTSBackend

_PIPER_VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "susurrus", "piper")


class PiperTTSBackend(TTSBackend):
    """Local TTS using Piper ONNX models.

    Requires: ``pip install piper-tts``
    """

    def synthesize(self, text, output_path="tts_output.wav", voice=None):
        try:
            from piper import PiperVoice
        except ImportError:
            raise ImportError(
                "piper-tts is required for the Piper TTS backend. "
                "Install with: pip install piper-tts"
            )

        voice_name = voice or self.kwargs.get("voice") or "de_DE-thorsten-medium"
        model_path = self._ensure_model(voice_name)

        piper_voice = PiperVoice.load(model_path)

        import wave

        with wave.open(output_path, "wb") as wav_file:
            piper_voice.synthesize(text, wav_file)

        logging.info(f"Piper TTS output: {output_path}")
        return output_path

    def _ensure_model(self, voice_name):
        """Download Piper voice model if not cached."""
        os.makedirs(_CACHE_DIR, exist_ok=True)

        # Voice name format: lang_REGION-name-quality
        parts = voice_name.split("-")
        if len(parts) >= 2:
            lang_region = parts[0]  # e.g. de_DE
            lang = lang_region.split("_")[0]  # e.g. de
        else:
            lang = "de"
            lang_region = "de_DE"

        onnx_name = f"{voice_name}.onnx"
        onnx_path = os.path.join(_CACHE_DIR, onnx_name)
        json_path = onnx_path + ".json"

        if os.path.isfile(onnx_path) and os.path.isfile(json_path):
            return onnx_path

        # Download from HuggingFace
        import urllib.request

        for fname, local_path in [(onnx_name, onnx_path), (onnx_name + ".json", json_path)]:
            url = f"{_PIPER_VOICES_BASE}/{lang}/{lang_region}/{voice_name}/{fname}"
            logging.info(f"Downloading Piper voice: {url}")
            try:
                urllib.request.urlretrieve(url, local_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download Piper voice {voice_name}: {e}")

        return onnx_path

    def list_voices(self):
        return [
            "de_DE-thorsten-medium",
            "de_DE-thorsten-high",
            "de_DE-thorsten-low",
            "de_DE-eva_k-x_low",
            "de_DE-karlsson-low",
            "de_DE-kerstin-low",
            "de_DE-pavoque-low",
            "de_DE-ramona-low",
            "en_US-lessac-medium",
            "en_US-amy-medium",
            "en_GB-alba-medium",
        ]
