# workers/transcription/backends/transformers_backend.py
"""Transformers backend"""
import logging
from .base import TranscriptionBackend

class TransformersBackend(TranscriptionBackend):
    """Hugging Face Transformers backend"""
    
    def __init__(self, model_id, device, language=None, word_timestamps=False,
                 max_chunk_length=30, **kwargs):
        super().__init__(model_id, device, language, **kwargs)
        self.word_timestamps = word_timestamps
        self.max_chunk_length = max_chunk_length if max_chunk_length > 0 else 30
        self.model = None
    
    def transcribe(self, audio_path):
        """Transcribe using Transformers"""
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError:
            raise ImportError("transformers and torch required for transformers backend")
        
        # Determine torch data type based on device
        if self.device == 'cpu':
            torch_dtype = torch.float32
        elif self.device == 'cuda':
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        logging.info(f"Loading model {self.model_id} with {torch_dtype}")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, device_map=self.device
        )
        processor = AutoProcessor.from_pretrained(self.model_id)
        
        # Create pipeline with chunking for long audio
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=self.max_chunk_length,
            return_timestamps="word" if self.word_timestamps else "chunk",
            device=self.device,
        )
        
        # Set language if specified
        if self.language:
            logging.info(f"Setting language to: {self.language}")
            asr_pipeline.model.config.forced_decoder_ids = (
                asr_pipeline.tokenizer.get_decoder_prompt_ids(
                    language=self.language, task="transcribe"
                )
            )
        
        logging.info("Starting transcription...")
        result = asr_pipeline(audio_path)
        
        if self.word_timestamps and 'chunks' in result:
            for chunk in result['chunks']:
                text = chunk['text'].strip()
                start = chunk['timestamp'][0] if chunk['timestamp'][0] is not None else 0.0
                end = chunk['timestamp'][1] if chunk['timestamp'][1] is not None else 0.0
                yield (start, end, text)
        else:
            text = result.get('text', '').strip()
            yield (0.0, 0.0, text)
            