# workers/tts/backends/chatterbox_tts_backend.py
"""Chatterbox TTS backend — expressive speech synthesis with voice cloning.

License: Chatterbox TTS is Apache 2.0.
Some handler patterns adapted from CrispTTS (EUPL v1.2).
"""

import logging
import os

from .base import TTSBackend


class ChatterboxTTSBackend(TTSBackend):
    """Local TTS using Chatterbox with optional reference audio for voice cloning.

    Requires: ``pip install chatterbox-tts``

    Kwargs:
        reference_audio: str — path to reference WAV for voice cloning
        exaggeration: float — emotion intensity (default 0.5)
        cfg_weight: float — classifier-free guidance weight
    """

    def synthesize(self, text, output_path="tts_output.wav", voice=None):
        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError:
            raise ImportError(
                "chatterbox-tts is required for the Chatterbox TTS backend. "
                "Install with: pip install chatterbox-tts"
            )

        import torch
        import torchaudio

        # Device selection
        if self.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif self.device == "mps" and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        model = ChatterboxTTS.from_pretrained(device=device)

        reference_audio = voice or self.kwargs.get("reference_audio")
        exaggeration = float(self.kwargs.get("exaggeration", 0.5))
        cfg_weight = self.kwargs.get("cfg_weight")

        generate_kwargs = {"exaggeration": exaggeration}
        if cfg_weight is not None:
            generate_kwargs["cfg_weight"] = float(cfg_weight)

        if reference_audio and os.path.isfile(reference_audio):
            wav = model.generate(text, audio_prompt_path=reference_audio, **generate_kwargs)
        else:
            wav = model.generate(text, **generate_kwargs)

        torchaudio.save(output_path, wav, model.sr)

        # Cleanup GPU
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

        logging.info(f"Chatterbox TTS output: {output_path}")
        return output_path

    def list_voices(self):
        return []
