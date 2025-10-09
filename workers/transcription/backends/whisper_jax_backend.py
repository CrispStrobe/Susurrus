# workers/transcription/backends/whisper_jax_backend.py
"""Whisper-JAX backend"""
import logging
from .base import TranscriptionBackend

class WhisperJaxBackend(TranscriptionBackend):
    """Whisper-JAX backend for fast TPU/GPU inference"""
    
    def __init__(self, model_id, device, language=None, word_timestamps=False, **kwargs):
        super().__init__(model_id, device, language, **kwargs)
        self.word_timestamps = word_timestamps
        self.temp_files = []
    
    def preprocess_audio(self, audio_path):
        """Convert to WAV for best compatibility"""
        from utils.audio_utils import convert_audio_to_wav
        wav_path = convert_audio_to_wav(audio_path)
        if wav_path != audio_path:
            self.temp_files.append(wav_path)
        return wav_path
    
    def transcribe(self, audio_path):
        """Transcribe using Whisper-JAX"""
        try:
            import jax
            import jax.numpy as jnp
            from whisper_jax import FlaxWhisperPipeline
        except ImportError:
            raise ImportError("whisper-jax, jax, and jaxlib required for whisper-jax backend")
        
        # Set JAX device
        jax_device = None
        if self.device in ['cuda', 'gpu']:
            jax_device = 'gpu' if jax.devices('gpu') else 'cpu'
        else:
            jax_device = 'cpu'
        
        # Set data type based on device
        dtype = jnp.bfloat16 if jax_device == 'gpu' else jnp.float32
        
        logging.info(f"Loading whisper-jax model '{self.model_id}' with dtype={dtype}")
        pipeline = FlaxWhisperPipeline(self.model_id, dtype=dtype)
        
        logging.info("Starting transcription with whisper-jax")
        
        # Run transcription
        outputs = pipeline(audio_path, language=self.language, return_timestamps=self.word_timestamps)
        
        if self.word_timestamps and 'chunks' in outputs:
            for chunk in outputs['chunks']:
                text = chunk['text'].strip()
                start = chunk['timestamp'][0] if chunk['timestamp'][0] is not None else 0.0
                end = chunk['timestamp'][1] if chunk['timestamp'][1] is not None else 0.0
                yield (start, end, text)
        else:
            text = outputs.get('text', '').strip()
            yield (0.0, 0.0, text)
    
    def cleanup(self):
        """Cleanup temporary files"""
        import os
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logging.warning(f"Failed to remove temp file {temp_file}: {e}")
