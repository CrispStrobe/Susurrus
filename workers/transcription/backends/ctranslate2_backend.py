# workers/transcription/backends/ctranslate2_backend.py
"""CTranslate2 backend"""
import logging
import os
from .base import TranscriptionBackend

class CTranslate2Backend(TranscriptionBackend):
    """CTranslate2 backend for optimized inference"""
    
    def __init__(self, model_id, device, language=None, word_timestamps=False,
                 quantization=None, preprocessor_path='', original_model_id='', **kwargs):
        super().__init__(model_id, device, language, **kwargs)
        self.word_timestamps = word_timestamps
        self.quantization = quantization
        self.preprocessor_path = preprocessor_path or model_id
        self.original_model_id = original_model_id
        self.model = None
    
    def transcribe(self, audio_path):
        """Transcribe using CTranslate2"""
        try:
            import ctranslate2
            from transformers import WhisperProcessor, WhisperTokenizer
            import librosa
        except ImportError:
            raise ImportError("ctranslate2, transformers, and librosa required for ctranslate2 backend")
        
        # Check if model directory exists
        model_dir = self.model_id
        
        if not os.path.exists(os.path.join(model_dir, 'model.bin')):
            logging.error(f"model.bin not found in {model_dir}")
            raise FileNotFoundError(f"model.bin not found in {model_dir}")
        
        # Load tokenizer and processor
        preprocessor_files = ["tokenizer.json", "vocabulary.json", "tokenizer_config.json"]
        preprocessor_missing = not all(
            os.path.exists(os.path.join(self.preprocessor_path, f)) 
            for f in preprocessor_files
        )
        
        if preprocessor_missing:
            logging.info("Preprocessor files not found locally. Attempting to download.")
            
            if not self.original_model_id:
                logging.error("Original model ID not specified. Cannot load tokenizer and processor.")
                raise Exception("Original model ID not specified.")
            
            logging.info(f"Using original model ID: {self.original_model_id}")
            
            try:
                tokenizer = WhisperTokenizer.from_pretrained(self.original_model_id)
                processor = WhisperProcessor.from_pretrained(self.original_model_id)
                logging.info("Loaded tokenizer and processor from original model.")
            except Exception as e:
                logging.error(f"Failed to load tokenizer and processor: {str(e)}")
                raise
        else:
            try:
                tokenizer = WhisperTokenizer.from_pretrained(self.preprocessor_path)
                processor = WhisperProcessor.from_pretrained(self.preprocessor_path)
                logging.info("Loaded tokenizer and processor from local files.")
            except Exception as e:
                logging.error(f"Failed to load tokenizer and processor: {str(e)}")
                raise
        
        # Load the model
        try:
            logging.info(f"Loading CTranslate2 model from {model_dir} on {self.device}")
            model = ctranslate2.models.Whisper(model_dir, device=self.device)
            
            # Load and process audio
            logging.info(f"Loading audio from {audio_path}")
            audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Process audio
            inputs = processor(audio_array, return_tensors="np", sampling_rate=16000)
            features = ctranslate2.StorageView.from_array(inputs.input_features)
            
            # Detect language if not specified
            language = self.language
            if not language:
                results = model.detect_language(features)
                detected_language, probability = results[0][0]
                logging.info(f"Detected language: {detected_language} with probability {probability:.4f}")
                language = detected_language
            
            # Prepare prompt
            prompt = tokenizer.convert_tokens_to_ids([
                "<|startoftranscript|>",
                language,
                "<|transcribe|>",
                "<|notimestamps|>" if not self.word_timestamps else "",
            ])
            
            # Generate transcription
            logging.info("Running transcription...")
            results = model.generate(features, [prompt], beam_size=5)
            
            # Decode results
            transcription = tokenizer.decode(results[0].sequences_ids[0])
            
            # Clean up and yield
            transcription = transcription.replace("<|startoftranscript|>", "")
            transcription = transcription.replace(f"<|{language}|>", "")
            transcription = transcription.replace("<|transcribe|>", "")
            transcription = transcription.replace("<|notimestamps|>", "")
            transcription = transcription.replace("<|endoftext|>", "").strip()
            
            yield (0.0, 0.0, transcription)
            
            logging.info("Transcription completed successfully")
            
        except Exception as e:
            logging.error(f"CTranslate2 transcription failed: {str(e)}")
            raise