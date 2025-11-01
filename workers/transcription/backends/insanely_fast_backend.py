# workers/transcription/backends/insanely_fast_backend.py:
"""Insanely Fast Whisper backend"""
import logging
import os
import subprocess
import tempfile
import threading
import json
from .base import TranscriptionBackend

class InsanelyFastBackend(TranscriptionBackend):
    """Insanely Fast Whisper backend"""
    
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
        """Transcribe using Insanely Fast Whisper"""
        try:
            # Create temporary file for output
            output_json = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
            self.temp_files.append(output_json)
            
            cmd = [
                "insanely-fast-whisper",
                "--file-name", audio_path,
                "--device-id", "0" if self.device == "cuda" else self.device,
                "--model-name", self.model_id,
                "--task", "transcribe",
                "--batch-size", "24",
                "--timestamp", "chunk" if not self.word_timestamps else "word",
                "--transcript-path", output_json
            ]
            
            if self.language:
                cmd.extend(["--language", self.language])
            
            logging.info(f"Running insanely-fast-whisper: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            
            # Read stderr in real-time and log it
            def read_stderr():
                for line in iter(process.stderr.readline, ''):
                    logging.info(f"insanely-fast-whisper: {line.strip()}")
            
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()
            
            # Wait for process to complete
            process.wait()
            stderr_thread.join(timeout=1)
            
            if process.returncode != 0:
                error_output = process.stderr.read() if process.stderr else "Unknown error"
                raise Exception(f"Insanely Fast Whisper failed with error: {error_output}")
            
            # Parse the output JSON file
            with open(output_json, 'r') as json_file:
                transcription_data = json.load(json_file)
            
            if self.word_timestamps and 'words' in transcription_data:
                # Word-level timestamps
                for word in transcription_data['words']:
                    text = word['text'].strip()
                    start = word['timestamp'][0] if word['timestamp'][0] is not None else 0.0
                    end = word['timestamp'][1] if word['timestamp'][1] is not None else 0.0
                    yield (start, end, text)
            elif 'chunks' in transcription_data:
                # Chunk-level timestamps
                for chunk in transcription_data['chunks']:
                    text = chunk['text'].strip()
                    start = chunk['timestamp'][0] if chunk['timestamp'][0] is not None else 0.0
                    end = chunk['timestamp'][1] if chunk['timestamp'][1] is not None else 0.0
                    yield (start, end, text)
            else:
                # Fall back to full text
                text = transcription_data.get('text', '').strip()
                yield (0.0, 0.0, text)
                
        except FileNotFoundError:
            raise ImportError("insanely-fast-whisper not found. Install with: pip install insanely-fast-whisper")
        except Exception as e:
            logging.error(f"Error with insanely-fast-whisper: {str(e)}")
            raise
    
    def cleanup(self):
        """Cleanup temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logging.warning(f"Failed to remove temp file {temp_file}: {e}")