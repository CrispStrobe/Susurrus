# workers/transcription/backends/voxtral_backend.py
"""Voxtral backends"""
import logging
import os
from .base import TranscriptionBackend

class VoxtralLocalBackend(TranscriptionBackend):
    """Voxtral local inference backend"""
    
    def __init__(self, model_id, device, language=None, temperature=0.0, 
                 max_chunk_length=1500, **kwargs):
        super().__init__(model_id, device, language, **kwargs)
        self.temperature = temperature
        self.max_chunk_length = max_chunk_length
        self.voxtral = None
    
    def transcribe(self, audio_path):
        """Transcribe using Voxtral local"""
        try:
            from backends.transcription.voxtral_local import VoxtralLocal
        except ImportError:
            raise ImportError("voxtral_local module required")
        
        logging.info(f"Using Voxtral (local) with model {self.model_id}")
        
        # Initialize Voxtral
        self.voxtral = VoxtralLocal(model_id=self.model_id, device=self.device)
        
        logging.info("Starting transcription with Voxtral...")
        segments = self.voxtral.transcribe(
            audio_path,
            language=self.language,
            temperature=self.temperature,
            chunk_length=self.max_chunk_length
        )
        
        # Yield segments
        for segment in segments:
            text = segment['text'].strip()
            start = segment['start']
            end = segment['end']
            yield (start, end, text)
    
    def cleanup(self):
        """Cleanup Voxtral model"""
        if self.voxtral:
            self.voxtral.unload_model()


class VoxtralAPIBackend(TranscriptionBackend):
    """Voxtral API backend"""
    
    def __init__(self, model_id, device, language=None, mistral_api_key=None, **kwargs):
        super().__init__(model_id, device, language, **kwargs)
        self.mistral_api_key = mistral_api_key or os.environ.get('MISTRAL_API_KEY')
        self.voxtral_api = None
    
    def transcribe(self, audio_path):
        """Transcribe using Voxtral API"""
        try:
            from backends.transcription.voxtral_api import VoxtralAPI
        except ImportError:
            raise ImportError("voxtral_api module required")
        
        if not self.mistral_api_key:
            raise ValueError("Mistral API key required for voxtral-api backend")
        
        logging.info("Using Voxtral (API)")
        
        # Initialize API client
        self.voxtral_api = VoxtralAPI(api_key=self.mistral_api_key)
        
        logging.info("Starting transcription with Voxtral API...")
        
        # Use chunking for long audio
        segments = self.voxtral_api.transcribe_with_chunking(
            audio_path,
            language=self.language,
            max_duration=600
        )
        
        if not segments:
            logging.warning("No segments returned from API!")
            yield (0.0, 0.0, "No transcription returned from API")
            return
        
        # Yield segments
        for segment in segments:
            text = segment.get('text', '').strip()
            start = segment.get('start', 0.0)
            end = segment.get('end', 0.0)
            
            if text:
                yield (start, end, text)