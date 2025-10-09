# workers/transcription/backends/base.py:
"""Base class for transcription backends"""
from abc import ABC, abstractmethod

class TranscriptionBackend(ABC):
    """Base class for all transcription backends"""
    
    def __init__(self, model_id, device, language=None, **kwargs):
        self.model_id = model_id
        self.device = device
        self.language = language
        self.kwargs = kwargs
    
    @abstractmethod
    def transcribe(self, audio_path):
        """Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Generator yielding (start, end, text) tuples or text lines
        """
        pass
    
    def preprocess_audio(self, audio_path):
        """Preprocess audio if needed"""
        return audio_path
    
    def cleanup(self):
        """Cleanup resources"""
        pass