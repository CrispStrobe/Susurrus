# workers/transcription/backends/openai_whisper_backend.py
"""OpenAI Whisper backend"""
import logging
import os
import tempfile
from .base import TranscriptionBackend

class OpenAIWhisperBackend(TranscriptionBackend):
    """Original OpenAI Whisper backend"""
    
    def __init__(self, model_id, device, language=None, word_timestamps=False,
                 max_chunk_length=0, **kwargs):
        super().__init__(model_id, device, language, **kwargs)
        self.word_timestamps = word_timestamps
        self.max_chunk_length = max_chunk_length
        self.temp_files = []
    
    def preprocess_audio(self, audio_path):
        """Convert to WAV for best compatibility"""
        from utils.audio_utils import convert_audio_to_wav
        wav_path = convert_audio_to_wav(audio_path)
        if wav_path != audio_path:
            self.temp_files.append(wav_path)
        return wav_path
    
    def transcribe(self, audio_path):
        """Transcribe using OpenAI Whisper"""
        try:
            import whisper
            from pydub import AudioSegment
            from pydub.silence import split_on_silence
        except ImportError:
            raise ImportError("openai-whisper and pydub required for OpenAI Whisper backend")
        
        # Load the model
        logging.info(f"Loading OpenAI Whisper model: {self.model_id}")
        model = whisper.load_model(self.model_id, device=self.device)
        
        logging.info("Starting transcription")
        
        # Handle chunking for long files if specified
        if self.max_chunk_length > 0:
            logging.info(f"Processing audio in chunks of {self.max_chunk_length} seconds")
            audio_segment = AudioSegment.from_file(audio_path)
            
            # Split audio on silence for natural chunks
            chunks = split_on_silence(
                audio_segment,
                min_silence_len=500,
                silence_thresh=audio_segment.dBFS - 14,
                keep_silence=250
            )
            
            # Merge chunks to respect max_chunk_length
            merged_chunks = []
            current_chunk = AudioSegment.empty()
            for chunk in chunks:
                if len(current_chunk) + len(chunk) <= self.max_chunk_length * 1000:
                    current_chunk += chunk
                else:
                    merged_chunks.append(current_chunk)
                    current_chunk = chunk
            merged_chunks.append(current_chunk)
            
            # Process each chunk
            total_offset = 0.0
            for chunk_index, chunk in enumerate(merged_chunks):
                # Create temporary file for this chunk
                chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                self.temp_files.append(chunk_file.name)
                chunk_file.close()
                
                # Export chunk to file
                chunk.export(chunk_file.name, format="wav")
                
                # Transcribe chunk
                result = model.transcribe(
                    chunk_file.name,
                    language=self.language,
                    word_timestamps=self.word_timestamps
                )
                
                # Yield segments with adjusted timestamps
                for segment in result['segments']:
                    text = segment['text'].strip()
                    start = segment['start'] + total_offset
                    end = segment['end'] + total_offset
                    yield (start, end, text)
                    
                    # Yield word timestamps if requested
                    if self.word_timestamps and 'words' in segment:
                        for word in segment['words']:
                            word_text = word['word'].strip()
                            word_start = word['start'] + total_offset
                            word_end = word['end'] + total_offset
                            yield (word_start, word_end, f"  {word_text}")
                
                # Update offset for next chunk
                total_offset += len(chunk) / 1000.0
        else:
            # Process the entire file at once
            result = model.transcribe(
                audio_path,
                language=self.language,
                word_timestamps=self.word_timestamps
            )
            
            # Yield segments
            for segment in result['segments']:
                text = segment['text'].strip()
                start = segment['start']
                end = segment['end']
                yield (start, end, text)
                
                # Yield word timestamps if requested
                if self.word_timestamps and 'words' in segment:
                    for word in segment['words']:
                        word_text = word['word'].strip()
                        word_start = word['start']
                        word_end = word['end']
                        yield (word_start, word_end, f"  {word_text}")
    
    def cleanup(self):
        """Cleanup temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logging.warning(f"Failed to remove temp file {temp_file}: {e}")