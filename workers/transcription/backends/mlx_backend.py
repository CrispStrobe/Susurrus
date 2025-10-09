# workers/transcription/backends/mlx_backend.py
"""MLX Whisper backend"""
import logging
from .base import TranscriptionBackend

class MLXBackend(TranscriptionBackend):
    """MLX Whisper backend for Apple Silicon"""
    
    def __init__(self, model_id, device, language=None, word_timestamps=False, **kwargs):
        super().__init__(model_id, device, language, **kwargs)
        self.word_timestamps = word_timestamps
        self.model = None
    
    def transcribe(self, audio_path):
        """Transcribe using MLX"""
        try:
            import mlx_whisper
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError("mlx_whisper and huggingface_hub required for mlx-whisper backend")
        
        logging.info(f"Using mlx-whisper with model {self.model_id}")
        
        # Convert to WAV if needed
        from utils.audio_utils import convert_audio_to_wav
        wav_path = convert_audio_to_wav(audio_path)
        
        # Download model if needed
        try:
            model_path = snapshot_download(repo_id=self.model_id)
            logging.info(f"Downloaded model files to: {model_path}")
        except Exception as e:
            logging.error(f"Failed to download model files: {str(e)}")
            raise
        
        # Transcribe
        transcribe_options = {
            "path_or_hf_repo": self.model_id,
            "verbose": True,
            "word_timestamps": self.word_timestamps,
            "language": self.language,
        }
        
        try:
            result = mlx_whisper.transcribe(wav_path, **transcribe_options)
            
            # Yield segments
            if 'segments' in result:
                for segment in result['segments']:
                    text = segment['text'].strip()
                    start = segment['start']
                    end = segment['end']
                    yield (start, end, text)
            else:
                text = result.get('text', '').strip()
                yield (0.0, 0.0, text)
                
        except Exception as e:
            logging.error(f"Transcription failed: {str(e)}")
            raise