import sys
import os
import tempfile
import time
import logging
from urllib.parse import urlparse
import subprocess
import shlex
import re

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QTextEdit,
    QFileDialog, QComboBox, QCheckBox, QHBoxLayout, QVBoxLayout, QSpinBox,
    QGroupBox, QPlainTextEdit, QMessageBox, QProgressBar, QInputDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Import the necessary modules for transcription
import torch
import requests
from pydub import AudioSegment

# Import mlx_whisper for transcription
import mlx_whisper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Force device to "cpu" to avoid MPS-related errors
device = "cpu"
#logging.info(f"Using device: {device}")

class TranscriptionThread(QThread):
    progress_signal = pyqtSignal(str, str)  # metrics, transcription
    error_signal = pyqtSignal(str)
    transcription_replace_signal = pyqtSignal(str)  # for whisper.cpp output files

    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        try:
            model_id = self.args['model_id']
            word_timestamps = self.args['word_timestamps']
            language = self.args['language']
            backend = self.args['backend']
            device_arg = self.args['device_arg']
            pipeline_type = self.args['pipeline_type']
            max_chunk_length = float(self.args['max_chunk_length']) if self.args['max_chunk_length'] else 0.0
            output_format = self.args['output_format']
            quantization = self.args.get('quantization', None)
            
            python_executable = sys.executable

            cmd = [
                python_executable,
                '-u',
                'transcribe_worker.py',
                '--model-id', model_id,
                '--backend', backend,
                '--device', device_arg,
            ]

            if self.args['audio_input']:
                cmd.extend(['--audio-input', self.args['audio_input']])
            if self.args['audio_url']:
                cmd.extend(['--audio-url', self.args['audio_url']])

            if word_timestamps:
                cmd.append('--word-timestamps')
            if language:
                cmd.extend(['--language', language])
            if pipeline_type != 'default':
                cmd.extend(['--pipeline-type', pipeline_type])
            if max_chunk_length > 0:
                cmd.extend(['--max-chunk-length', str(max_chunk_length)])
            if backend == 'whisper.cpp' and output_format:
                cmd.extend(['--output-format', output_format])
            if backend == 'ctranslate2' and quantization:
                cmd.extend(['--quantization', quantization])

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
            )

            timecode_pattern = re.compile(
                r'^\['
                r'(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{3}|\d+\.\d+)'
                r'\s-->\s'
                r'(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{3}|\d+\.\d+)'
                r'\]\s*(.*)'
            )
            in_transcription = False
            output_file = None
            is_whisper_jax = self.args['backend'] == 'whisper-jax'
            whisper_jax_output = ""

            for line in process.stdout:
                line = line.rstrip()
                
                if line.startswith('OUTPUT FILE: '):
                    output_file = line[len('OUTPUT FILE: '):].strip()
                    continue

                if is_whisper_jax:
                    if line.startswith("Transcription time:") or line.startswith("Audio file size:"):
                        self.progress_signal.emit(line, '')
                    else:
                        whisper_jax_output += line + "\n"
                        self.progress_signal.emit('', line)
                elif 'Starting transcription' in line or 'Detected language' in line or 'Transcription time' in line:
                    self.progress_signal.emit(line, '')
                    in_transcription = False
                elif timecode_pattern.match(line):
                    match = timecode_pattern.match(line)
                    text = match.group(3)
                    self.progress_signal.emit('', text)
                    in_transcription = True
                elif in_transcription and line.strip() != '':
                    self.progress_signal.emit('', line)
                elif line.strip() != '':
                    self.progress_signal.emit(line, '')
                    in_transcription = False

            # Wait for the process to complete
            process.stdout.close()
            process.wait()

            if process.returncode != 0:
                error_msg = process.stderr.read()
                self.error_signal.emit(error_msg)
            else:
                if is_whisper_jax:
                    self.transcription_replace_signal.emit(whisper_jax_output)
                elif self.args['backend'] == 'whisper.cpp' and self.args['output_format'] in ('srt', 'vtt'):
                    if output_file and os.path.exists(output_file):
                        with open(output_file, 'r', encoding='utf-8') as f:
                            output_content = f.read()
                            self.transcription_replace_signal.emit(output_content)
                    else:
                        self.error_signal.emit(f"Whisper.cpp output file not found: {output_file}")

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            self.error_signal.emit(error_msg)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Susurrus: Audio Transcription")
        self.init_ui()

    def check_transcribe_button_state(self):
        if self.audio_input_path.text().strip() or self.audio_url.text().strip():
            self.transcribe_button.setEnabled(True)
        else:
            self.transcribe_button.setEnabled(False)
    
    def init_ui(self):

        self.backend_model_map = {
            'mlx-whisper': [
                "mlx-community/whisper-large-v3-turbo",
                "mlx-community/whisper-large-v3-turbo-q4",
                "mlx-community/whisper-tiny-mlx-4bit",
                "mlx-community/whisper-base-mlx-4bit",
                "mlx-community/whisper-small-mlx-q4",
                "mlx-community/whisper-medium-mlx-4bit",
                "mlx-community/whisper-large-v3-mlx-4bit",
                "mlx-community/whisper-large-v3-mlx"
            ],
            'faster-batched': [
                "cstr/whisper-large-v3-turbo-int8_float32",
                "SYSTRAN/faster-whisper-large-v1",
                "GalaktischeGurke/primeline-whisper-large-v3-german-ct2"
            ],
            'faster-sequenced': [
                "cstr/whisper-large-v3-turbo-int8_float32",
                "SYSTRAN/faster-whisper-large-v1",
                "GalaktischeGurke/primeline-whisper-large-v3-german-ct2"
            ],
            'transformers': [
                "openai/whisper-large-v3",
                "openai/whisper-large-v2",
                "openai/whisper-medium",
                "openai/whisper-small"
            ],
            'OpenAI Whisper': [
                "large-v2",
                "medium",
                "small",
                "base",
                "tiny"
            ],
            'whisper.cpp': [
                'large-v3-turbo-q5_0',
                'large-v3-turbo',
                'small',
                'base',
                'tiny',
                'tiny.en'
            ],
            'ctranslate2': [
                # Include models compatible with ctranslate2
                "cstr/whisper-large-v3-turbo-int8_float32",
                "SYSTRAN/faster-whisper-large-v1",
                "GalaktischeGurke/primeline-whisper-large-v3-german-ct2",
            ],
            'whisper-jax': [
                "openai/whisper-tiny",
                "openai/whisper-medium",
                "tiny.en",
                "base.en",
                "small.en",
                "medium.en",
                "large-v2",
            ],
        }

        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel("<h1>Susurus: Audio Transcription</h1>")
        subtitle_label = QLabel("Transcribe audio using various (Whisper-) backends.")
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)

        # Input Row
        input_row = QHBoxLayout()

        # Audio Input (file path display and button to select file)
        self.audio_input_path = QLineEdit()
        self.audio_input_path.setPlaceholderText("Upload or Record Audio")
        self.audio_input_button = QPushButton("Browse")
        self.audio_input_button.clicked.connect(self.select_audio_file)

        # Audio URL
        self.audio_url = QLineEdit()
        self.audio_url.setPlaceholderText("Or Enter URL of audio file or YouTube link")

        input_row.addWidget(QLabel("Audio File:"))
        input_row.addWidget(self.audio_input_path)
        input_row.addWidget(self.audio_input_button)
        input_row.addWidget(QLabel("or URL:"))
        input_row.addWidget(self.audio_url)

        main_layout.addLayout(input_row)

        # Transcribe Button
        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.setEnabled(False)
        self.transcribe_button.clicked.connect(self.start_transcription)

        main_layout.addWidget(self.transcribe_button)

        self.audio_input_path.textChanged.connect(self.check_transcribe_button_state)
        self.audio_url.textChanged.connect(self.check_transcribe_button_state)
        
        # Advanced Options Accordion
        advanced_options_group = QGroupBox("Advanced Options")
        advanced_options_group.setCheckable(True)
        advanced_options_group.setChecked(False)

        advanced_layout = QVBoxLayout()

        # Proxy Row
        proxy_row = QHBoxLayout()
        self.proxy_url = QLineEdit()
        self.proxy_url.setPlaceholderText("Enter proxy URL if needed")
        self.proxy_username = QLineEdit()
        self.proxy_username.setPlaceholderText("Proxy username (optional)")
        self.proxy_password = QLineEdit()
        self.proxy_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.proxy_password.setPlaceholderText("Proxy password (optional)")

        proxy_row.addWidget(QLabel("Proxy URL:"))
        proxy_row.addWidget(self.proxy_url)
        proxy_row.addWidget(QLabel("Username:"))
        proxy_row.addWidget(self.proxy_username)
        proxy_row.addWidget(QLabel("Password:"))
        proxy_row.addWidget(self.proxy_password)

        advanced_layout.addLayout(proxy_row)

        # Transcription Backend Selection Row
        backend_row = QHBoxLayout()
        self.backend_selection = QComboBox()
        self.backend_selection.addItems([
            "mlx-whisper",
            "OpenAI Whisper",
            "faster-batched",
            "faster-sequenced",
            "transformers",
            "whisper.cpp",
            "ctranslate2",
            "whisper-jax",  
        ])

        self.backend_selection.setCurrentText("mlx-whisper")

        backend_row.addWidget(QLabel("Backend:"))
        backend_row.addWidget(self.backend_selection)

        advanced_layout.addLayout(backend_row)

        # Connect the backend selection change to the update_model_options method
        self.backend_selection.currentTextChanged.connect(self.update_model_options)

        # Combine Model, Device, and Language selection into one row
        model_row = QHBoxLayout()

        model_row.addWidget(QLabel("Model:"))
        self.model_id = QComboBox()
        # Set initial models for the default backend
        self.model_id.addItems(self.backend_model_map[self.backend_selection.currentText()])
        self.model_id.setEditable(True)
        model_row.addWidget(self.model_id)

        model_row.addWidget(QLabel("Device:"))
        self.device_selection = QComboBox()
        self.device_selection.addItems([
            "Auto",  # Let the script decide the best device
            "CPU",
            "GPU",  # CUDA
            "MPS"   # Apple Silicon
        ])
        self.device_selection.setCurrentText("Auto")
        model_row.addWidget(self.device_selection)

        model_row.addWidget(QLabel("Language:"))
        self.language = QLineEdit()
        self.language.setPlaceholderText("en")
        model_row.addWidget(self.language)

        advanced_layout.addLayout(model_row)

        # Output Format Row (initially hidden)
        output_format_row = QHBoxLayout()
        self.output_format_selection = QComboBox()
        self.output_format_selection.addItems(['txt', 'srt', 'vtt'])
        self.output_format_selection.setCurrentText('txt')

        output_format_row.addWidget(QLabel("Output Format:"))
        output_format_row.addWidget(self.output_format_selection)

        self.output_format_row_widget = QWidget()
        self.output_format_row_widget.setLayout(output_format_row)
        self.output_format_row_widget.setVisible(False)
        advanced_layout.addWidget(self.output_format_row_widget)
        
        # Max Chunk Length Row (initially hidden)
        chunk_row = QHBoxLayout()
        self.max_chunk_length = QLineEdit()
        self.max_chunk_length.setPlaceholderText("Max Chunk Length (seconds, 0=No Chunking, default=0)")
        self.max_chunk_length.setText("0")  # Default value

        chunk_row.addWidget(QLabel("Max Chunk Length:"))
        chunk_row.addWidget(self.max_chunk_length)

        # Store the widget to control visibility
        self.chunk_row_widget = QWidget()
        self.chunk_row_widget.setLayout(chunk_row)
        self.chunk_row_widget.setVisible(False)
        advanced_layout.addWidget(self.chunk_row_widget)
        
        
        # Start Time, End Time, Include Timecodes Row
        misc_row = QHBoxLayout()
        self.start_time = QLineEdit()
        self.start_time.setPlaceholderText("Start Time (seconds)")
        self.end_time = QLineEdit()
        self.end_time.setPlaceholderText("End Time (seconds)")

        self.word_timestamps = QCheckBox("Word-level timestamps")
        self.word_timestamps.setChecked(False)

        misc_row.addWidget(QLabel("Start Time (s):"))
        misc_row.addWidget(self.start_time)
        misc_row.addWidget(QLabel("End Time (s):"))
        misc_row.addWidget(self.end_time)
        #misc_row.addWidget(self.word_timestamps)

        advanced_layout.addLayout(misc_row)

        # Set layout for advanced options group
        advanced_options_group.setLayout(advanced_layout)
        main_layout.addWidget(advanced_options_group)

        # Output Row
        output_layout = QHBoxLayout()

        # Metrics Output
        metrics_layout = QVBoxLayout()
        metrics_label = QLabel("Transcription Metrics and Verbose Messages")
        self.metrics_output = QPlainTextEdit()
        self.metrics_output.setReadOnly(True)
        metrics_layout.addWidget(metrics_label)
        metrics_layout.addWidget(self.metrics_output)

        # Transcription Output
        transcription_layout = QVBoxLayout()
        transcription_label = QLabel("Transcription")
        self.transcription_output = QPlainTextEdit()
        self.transcription_output.setReadOnly(True)
        transcription_layout.addWidget(transcription_label)
        transcription_layout.addWidget(self.transcription_output)

        # Save Transcription Button
        self.save_transcription_button = QPushButton("Download Transcription")
        self.save_transcription_button.setEnabled(False)
        self.save_transcription_button.clicked.connect(self.save_transcription)

        output_layout.addLayout(metrics_layout)
        output_layout.addLayout(transcription_layout)
        output_layout.addWidget(self.save_transcription_button)

        main_layout.addLayout(output_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.setLayout(main_layout)

    def select_audio_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.mp3 *.wav *.flac)")
        if file_name:
            self.audio_input_path.setText(file_name)

    def update_model_options(self, backend):
        models = self.backend_model_map.get(backend, [])
        self.model_id.clear()
        self.model_id.addItems(models)

        # Show/hide chunking selection based on backend
        if backend in ['OpenAI Whisper', 'transformers']:
            self.chunk_row_widget.setVisible(True)
        else:
            self.chunk_row_widget.setVisible(False)

        # Show/hide output format selection based on backend
        if backend == 'whisper.cpp':
            self.output_format_row_widget.setVisible(True)
        else:
            self.output_format_row_widget.setVisible(False)

    def replace_transcription_output(self, text):
        self.transcription_output.clear()
        self.transcription_output.setPlainText(text)
        self.save_transcription_button.setEnabled(True)
        self.transcription_text = text  # Update the transcription text used for saving
    
    def start_transcription(self):
        # Collect arguments from UI
        args = {
            'audio_input': self.audio_input_path.text(),
            'audio_url': self.audio_url.text(),
            'proxy_url': self.proxy_url.text(),
            'proxy_username': self.proxy_username.text(),
            'proxy_password': self.proxy_password.text(),
            'model_id': self.model_id.currentText(),
            'start_time': self.start_time.text(),
            'end_time': self.end_time.text(),
            'word_timestamps': False,  # Set accordingly
            'language': self.language.text().strip() or "",
            'backend': self.backend_selection.currentText(),
            'device_arg': self.device_selection.currentText(),
            'pipeline_type': 'default',  # Or get it from the UI if applicable
            'max_chunk_length': self.max_chunk_length.text(),
            'output_format': self.output_format_selection.currentText(),
        }
        
        # Handle ctranslate2 specific logic
        quantization = None
        if args['backend'] == 'ctranslate2':
            args['quantization'] = quantization
            model_dir = os.path.join(os.getcwd(), 'ctranslate2_models', args['model_id'].replace('/', '_'))
            if not os.path.exists(model_dir):
                # Ask for quantization
                quantization, ok_pressed = QInputDialog.getItem(
                    self, 
                    "Select Quantization",
                    "Choose quantization for model conversion:",
                    ["float32", "int8_float16", "int16", "int8"], 
                    1, 
                    False
                )
                if not ok_pressed:
                    self.progress_bar.setVisible(False)
                    self.transcribe_button.setEnabled(True)
                    return
                reply = QMessageBox.question(
                    self,
                    "Model Conversion Required",
                    f"The model {args['model_id']} needs to be converted to ctranslate2 format with quantization '{quantization}'. This may take several minutes. Do you want to proceed?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    self.progress_bar.setVisible(False)
                    self.transcribe_button.setEnabled(True)
                    return
                # Pass quantization to the worker script
                args['quantization'] = quantization

        self.thread = TranscriptionThread(args)
        self.thread.progress_signal.connect(self.update_outputs)
        self.thread.error_signal.connect(self.show_error)
        self.thread.finished.connect(self.on_transcription_finished)
        
        # must we replace the content of the transcription with a file from whisper.cpp?
        self.thread.transcription_replace_signal.connect(self.replace_transcription_output)  # signal for when we read from whisper.cpp output file
        
        self.thread.start()
        self.transcribe_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.transcription_output.clear()
        self.metrics_output.clear()
        self.transcription_text = ""

    def on_transcription_finished(self):
        self.transcribe_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def show_error(self, error_msg):
        self.metrics_output.appendPlainText(error_msg)
        QMessageBox.critical(self, "Error", error_msg)
        self.transcribe_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def update_outputs(self, metrics, transcription):
        if metrics:
            self.metrics_output.appendPlainText(metrics)
        if transcription:
            self.transcription_output.appendPlainText(transcription)
            self.save_transcription_button.setEnabled(True)
            self.transcription_text += transcription + '\n'

    def save_transcription(self):
        if hasattr(self, 'transcription_text'):
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Transcription", "", "Text Files (*.txt)")
            if save_path:
                try:
                    with open(save_path, 'w', encoding='utf-8') as dst:
                        dst.write(self.transcription_text)
                    QMessageBox.information(self, "Success", f"Transcription saved to: {save_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save transcription: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "No transcription available to save.")

def transcribe_audio(
    audio_input, audio_url, proxy_url, proxy_username, proxy_password,
    model_id, start_time=None, end_time=None, verbose=False, word_timestamps=False,
    language=None
):
    try:
        if verbose:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.WARNING)

        logging.info(f"Transcription parameters: model_id={model_id}")

        # Determine the audio source
        audio_path = None
        is_temp_file = False

        if audio_input and len(audio_input) > 0:
            # audio_input is a filepath to uploaded or recorded audio
            audio_path = audio_input
            is_temp_file = False
        elif audio_url and len(audio_url.strip()) > 0:
            # audio_url is provided
            audio_path, is_temp_file = download_audio(audio_url, proxy_url, proxy_username, proxy_password)
            if not audio_path:
                error_msg = f"Error downloading audio from {audio_url}. Check logs for details."
                logging.error(error_msg)
                yield f"Error: {error_msg}", ""
                return
        else:
            error_msg = "No audio source provided. Please upload an audio file or enter a URL."
            logging.error(error_msg)
            yield f"Error: {error_msg}", ""
            return

        # Convert start_time and end_time to float or None
        start_time = float(start_time) if start_time else None
        end_time = float(end_time) if end_time else None

        if start_time is not None or end_time is not None:
            audio_path = trim_audio(audio_path, start_time, end_time)
            is_temp_file = True  # The trimmed audio is a temporary file
            logging.info(f"Audio trimmed from {start_time} to {end_time}")

        # Perform the transcription
        start_time_perf = time.time()

        # Load the model (mlx-whisper handles caching)
        transcribe_options = {
            "path_or_hf_repo": model_id,
            "verbose": verbose,
            "word_timestamps": word_timestamps,
            "language": language,  # Specify the language to avoid auto-detection
        }

        logging.info("Starting transcription")
        result = mlx_whisper.transcribe(audio_path, **transcribe_options)
        transcription = result["text"]

        end_time_perf = time.time()

        # Calculate metrics
        transcription_time = end_time_perf - start_time_perf
        audio_file_size = os.path.getsize(audio_path) / (1024 * 1024)

        metrics_output = (
            f"Transcription time: {transcription_time:.2f} seconds\n"
            f"Audio file size: {audio_file_size:.2f} MB\n"
        )

        # Optionally include word-level timestamps
        if word_timestamps:
            word_transcriptions = []
            for segment in result["segments"]:
                words_info = segment["words"]
                for word_info in words_info:
                    word_text = word_info["word"]
                    start = word_info["start"]
                    end = word_info["end"]
                    word_transcriptions.append(f"[{start:.2f}s -> {end:.2f}s] {word_text}")
            transcription = "\n".join(word_transcriptions)

        yield metrics_output, transcription

    except Exception as e:
        error_msg = f"An error occurred during transcription: {str(e)}"
        logging.error(error_msg)
        yield f"Error: {error_msg}", ""

    finally:
        # Clean up temporary audio files
        if audio_path and is_temp_file and os.path.exists(audio_path):
            os.remove(audio_path)

def trim_audio(audio_path, start_time, end_time):
    try:
        logging.info(f"Trimming audio from {start_time} to {end_time}")
        audio = AudioSegment.from_file(audio_path)
        audio_duration = len(audio) / 1000  # Duration in seconds

        # Default start and end times if None
        start_time = max(0, start_time) if start_time is not None else 0
        end_time = min(audio_duration, end_time) if end_time is not None else audio_duration

        # Validate times
        if start_time >= end_time:
            raise Exception("End time must be greater than start time.")

        trimmed_audio = audio[int(start_time * 1000):int(end_time * 1000)]
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
            trimmed_audio.export(temp_audio_file.name, format="wav")
            logging.info(f"Trimmed audio saved to: {temp_audio_file.name}")
        return temp_audio_file.name
    except Exception as e:
        logging.error(f"Error trimming audio: {str(e)}")
        raise Exception(f"Error trimming audio: {str(e)}")

def download_audio(url, proxy_url, proxy_username, proxy_password):
    parsed_url = urlparse(url)
    logging.info(f"Downloading audio from URL: {url}")
    try:
        if 'youtube.com' in parsed_url.netloc or 'youtu.be' in parsed_url.netloc:
            audio_file = download_youtube_audio(url, proxy_url, proxy_username, proxy_password)
            if not audio_file:
                logging.error(f"Failed to download audio from {url}. Ensure yt-dlp is installed and up to date.")
                return None, False
        else:
            audio_file = download_direct_audio(url, proxy_url, proxy_username, proxy_password)
            if not audio_file:
                logging.error(f"Failed to download audio from {url}")
                return None, False
        return audio_file, True
    except Exception as e:
        logging.error(f"Error downloading audio from {url}: {str(e)}")
        return None, False

def download_youtube_audio(url, proxy_url, proxy_username, proxy_password):
    import yt_dlp
    logging.info(f"Using yt-dlp {yt_dlp.version.__version__} to download YouTube audio")
    temp_dir = tempfile.mkdtemp()
    output_template = os.path.join(temp_dir, '%(id)s.%(ext)s')
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
        'no_warnings': False,
        'logger': MyLogger(),
    }
    if proxy_url and len(proxy_url.strip()) > 0:
        ydl_opts['proxy'] = proxy_url
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if 'entries' in info:
                # Can be a playlist or a list of videos
                info = info['entries'][0]
            output_file = ydl.prepare_filename(info)
            output_file = os.path.splitext(output_file)[0] + '.mp3'
            if os.path.exists(output_file):
                logging.info(f"Downloaded YouTube audio: {output_file}")
                return output_file
            else:
                logging.error("yt-dlp did not produce an output file.")
                return None
    except Exception as e:
        logging.error(f"yt-dlp failed to download audio: {str(e)}")
        return None

def download_direct_audio(url, proxy_url, proxy_username, proxy_password):
    try:
        proxies = None
        auth = None
        if proxy_url and len(proxy_url.strip()) > 0:
            proxies = {
                "http": proxy_url,
                "https": proxy_url
            }
            if proxy_username and proxy_password:
                auth = (proxy_username, proxy_password)
        response = requests.get(url, stream=True, proxies=proxies, auth=auth)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
            logging.info(f"Downloaded direct audio to: {temp_file.name}")
            return temp_file.name
        else:
            logging.error(f"Failed to download audio from {url} with status code {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Error in download_direct_audio: {str(e)}")
        return None

class MyLogger(object):
    def debug(self, msg):
        logging.debug(msg)
    def info(self, msg):
        logging.info(msg)
    def warning(self, msg):
        logging.warning(msg)
    def error(self, msg):
        logging.error(msg)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
