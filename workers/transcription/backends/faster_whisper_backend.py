# workers/transcription/backends/faster_whisper_backend.py
"""Faster Whisper backends"""
import logging
from .base import TranscriptionBackend

class FasterWhisperBatchedBackend(TranscriptionBackend):
    """Faster Whisper with batched inference"""
    
    def __init__(self, model_id, device, language=None, word_timestamps=False, 
                 quantization=None, **kwargs):
        super().__init__(model_id, device, language, **kwargs)
        self.word_timestamps = word_timestamps
        self.quantization = quantization
        self.model = None
    
    def transcribe(self, audio_path):
        """Transcribe using Faster Whisper batched"""
        try:
            from faster_whisper import WhisperModel, BatchedInferencePipeline
        except ImportError:
            raise ImportError("faster_whisper required for faster-batched backend")
        
        compute_type = self.quantization if self.quantization else 'int8'
        
        logging.info(f"Loading model {self.model_id} with compute_type={compute_type}")
        model = WhisperModel(self.model_id, device=self.device, compute_type=compute_type)
        pipeline = BatchedInferencePipeline(model=model)
        
        logging.info("Starting batched transcription")
        segments, info = pipeline.transcribe(
            audio_path,
            batch_size=4,
            language=self.language,
            word_timestamps=self.word_timestamps,
            vad_filter=True,
        )
        
        if hasattr(info, 'language') and info.language:
            logging.info(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
        
        for segment in segments:
            text = segment.text.strip()
            start = segment.start
            end = segment.end
            
            try:
                text = text.encode('utf-8', errors='replace').decode('utf-8')
                yield (start, end, text)
                
                # Yield word timestamps if requested
                if self.word_timestamps and segment.words:
                    for word in segment.words:
                        word_text = word.word.strip()
                        word_text = word_text.encode('utf-8', errors='replace').decode('utf-8')
                        word_start = word.start
                        word_end = word.end
                        yield (word_start, word_end, f"  {word_text}")
            except Exception as e:
                logging.error(f"Error encoding segment text: {e}")
                yield (start, end, "[Encoding issue with text]")


class FasterWhisperSequencedBackend(TranscriptionBackend):
    """Faster Whisper with sequential inference"""
    
    def __init__(self, model_id, device, language=None, word_timestamps=False,
                 quantization=None, **kwargs):
        super().__init__(model_id, device, language, **kwargs)
        self.word_timestamps = word_timestamps
        self.quantization = quantization
    
    def transcribe(self, audio_path):
        """Transcribe using Faster Whisper sequential"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("faster_whisper required for faster-sequenced backend")
        
        # Determine compute type based on device
        if self.device == 'cuda':
            compute_type = "float16"
        else:
            compute_type = "int8"
        
        # Override with user-specified quantization if provided
        if self.quantization:
            compute_type = self.quantization
        
        logging.info(f"Loading model {self.model_id} with compute_type={compute_type}")
        model = WhisperModel(self.model_id, device=self.device, compute_type=compute_type)
        
        options = {
            "language": self.language,
            "beam_size": 5,
            "best_of": 5,
            "word_timestamps": self.word_timestamps,
            "vad_filter": True,
        }
        
        segments, info = model.transcribe(audio_path, **options)
        
        if hasattr(info, 'language') and info.language:
            logging.info(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
        
        for segment in segments:
            text = segment.text.strip()
            start = segment.start
            end = segment.end
            
            try:
                text = text.encode('utf-8', errors='replace').decode('utf-8')
                yield (start, end, text)
                
                # Yield word timestamps if requested
                if self.word_timestamps and segment.words:
                    for word in segment.words:
                        word_text = word.word.strip()
                        word_text = word_text.encode('utf-8', errors='replace').decode('utf-8')
                        word_start = word.start
                        word_end = word.end
                        yield (word_start, word_end, f"  {word_text}")
            except Exception as e:
                logging.error(f"Error encoding segment text: {e}")
                yield (start, end, "[Encoding issue with text]")