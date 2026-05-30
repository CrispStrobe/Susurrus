# workers/tts/backends/speecht5_tts_backend.py
"""SpeechT5 TTS backend — Microsoft's encoder-decoder speech synthesis.

License: SpeechT5 model is MIT-licensed.
"""

import logging

from .base import TTSBackend


class SpeechT5TTSBackend(TTSBackend):
    """Local TTS using Microsoft SpeechT5.

    Requires: ``pip install transformers torch soundfile datasets``
    """

    def synthesize(self, text, output_path="tts_output.wav", voice=None):
        try:
            import soundfile as sf
            import torch
            from datasets import load_dataset
            from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
        except ImportError:
            raise ImportError(
                "transformers, torch, soundfile, and datasets are required for SpeechT5. "
                "Install with: pip install transformers torch soundfile datasets"
            )

        model_id = self.model_id or "microsoft/speecht5_tts"

        processor = SpeechT5Processor.from_pretrained(model_id)
        model = SpeechT5ForTextToSpeech.from_pretrained(model_id)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        # Device
        if self.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif self.device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model = model.to(device)
        vocoder = vocoder.to(device)

        # Speaker embedding
        speaker_idx = int(self.kwargs.get("speaker_idx", 7306))
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embedding = (
            torch.tensor(embeddings_dataset[speaker_idx]["xvector"]).unsqueeze(0).to(device)
        )

        inputs = processor(text=text, return_tensors="pt").to(device)
        speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)

        sf.write(output_path, speech.cpu().numpy(), samplerate=16000)

        # Cleanup
        del model, vocoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

        logging.info(f"SpeechT5 TTS output: {output_path}")
        return output_path

    def list_voices(self):
        return []
