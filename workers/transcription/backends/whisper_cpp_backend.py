# workers/transcription/backends/whisper_cpp_backend.py
"""Whisper.cpp backend"""
import logging
import os
import subprocess
import tempfile
import threading
import re
from .base import TranscriptionBackend

class WhisperCppBackend(TranscriptionBackend):
    """Whisper.cpp backend for lightweight inference"""
    
    def __init__(self, model_id, device, language=None, word_timestamps=False,
                 output_format='txt', **kwargs):
        super().__init__(model_id, device, language, **kwargs)
        self.word_timestamps = word_timestamps
        self.output_format = output_format
        self.temp_files = []
    
    def preprocess_audio(self, audio_path):
        """Convert to WAV format for whisper.cpp"""
        from utils.audio_utils import convert_audio_to_wav
        wav_path = convert_audio_to_wav(audio_path)
        if wav_path != audio_path:
            self.temp_files.append(wav_path)
        return wav_path
    
    def transcribe(self, audio_path):
        """Transcribe using whisper.cpp"""
        logging.info("=== Starting whisper.cpp pipeline ===")
        
        # Find or download the model
        from models.model_config import find_or_download_whisper_cpp_model, find_whisper_cpp_executable
        
        try:
            model_path = find_or_download_whisper_cpp_model(self.model_id)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            model_size = os.path.getsize(model_path) / 1024 / 1024
            logging.info(f"Using model file: {os.path.basename(model_path)} ({model_size:.2f} MB)")
        except Exception as e:
            logging.error(f"Model preparation failed: {str(e)}")
            raise
        
        # Locate the whisper.cpp executable
        whisper_cpp_executable = find_whisper_cpp_executable()
        if not whisper_cpp_executable:
            raise FileNotFoundError("Could not find whisper.cpp executable")
        logging.info(f"Using executable: {whisper_cpp_executable}")
        
        # Prepare output file path if needed
        output_file = None
        if self.output_format in ('srt', 'vtt'):
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{self.output_format}').name
            self.temp_files.append(output_file)
        
        # Build the command
        cmd = [
            whisper_cpp_executable,
            '-m', model_path,
            '-f', audio_path,
            '-t', str(min(os.cpu_count() or 4, 8)),
        ]
        
        # Add options
        if self.language:
            cmd.extend(['-l', self.language])
        else:
            cmd.extend(['-l', 'auto'])
        
        if self.word_timestamps:
            cmd.append('--word-timestamps')
        
        # Add output format options
        if self.output_format == 'srt':
            cmd.append('--output-srt')
        elif self.output_format == 'vtt':
            cmd.append('--output-vtt')
        elif self.output_format == 'txt':
            cmd.append('--output-txt')
        
        # Add file paths if specified
        if output_file:
            cmd.append(output_file)
        
        # Run the command
        logging.info(f"Running whisper.cpp: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Process output in real-time
        def read_output(stream, is_stderr=False):
            for line in iter(stream.readline, ''):
                line = line.strip()
                if not line:
                    continue
                
                if is_stderr:
                    logging.info(f"whisper.cpp: {line}")
                else:
                    # Try to extract timestamps and text
                    timestamp_match = re.match(r'\[(\d+:\d+:\d+\.\d+) --> (\d+:\d+:\d+\.\d+)\]\s+(.+)', line)
                    if timestamp_match:
                        # Parse timestamp format to seconds
                        start_str, end_str, text = timestamp_match.groups()
                        start = self._parse_timestamp(start_str)
                        end = self._parse_timestamp(end_str)
                        yield (start, end, text)
                    else:
                        # For lines without timestamps
                        yield (0.0, 0.0, line)
        
        # Start separate threads for stdout and stderr
        output_lines = []
        
        def collect_stdout():
            for result in read_output(process.stdout, False):
                output_lines.append(result)
        
        def collect_stderr():
            for line in iter(process.stderr.readline, ''):
                line = line.strip()
                if line:
                    logging.info(f"whisper.cpp: {line}")
        
        stdout_thread = threading.Thread(target=collect_stdout, daemon=True)
        stderr_thread = threading.Thread(target=collect_stderr, daemon=True)
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete
        return_code = process.wait()
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        
        # Handle the process result
        if return_code != 0:
            logging.error(f"whisper.cpp failed with return code {return_code}")
            raise Exception("Transcription process failed")
        
        # Yield collected output
        for result in output_lines:
            yield result
        
        # If we have a separate output file (srt/vtt), read and yield its contents
        if output_file and os.path.exists(output_file):
            logging.info(f"OUTPUT FILE: {output_file}")
            with open(output_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
                if not file_content.strip():
                    logging.warning("Output file is empty")
                else:
                    # Parse the file content and yield
                    for line in file_content.split('\n'):
                        if line.strip():
                            yield (0.0, 0.0, line)
    
    def _parse_timestamp(self, timestamp_str):
        """Parse timestamp string to seconds"""
        parts = timestamp_str.split(':')
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    def cleanup(self):
        """Cleanup temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logging.warning(f"Failed to remove temp file {temp_file}: {e}")