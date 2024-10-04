import sys
import os
import tempfile
import time
import logging
from urllib.parse import urlparse
import subprocess
import shlex
import re
import shutil

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QTextEdit,
    QFileDialog, QComboBox, QCheckBox, QHBoxLayout, QVBoxLayout, QSpinBox,
    QGroupBox, QPlainTextEdit, QMessageBox, QProgressBar, QInputDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QAction

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

import os

from PyQt6.QtWidgets import QWidget, QPushButton, QVBoxLayout

class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)

        self.toggle_button = QPushButton()
        
        #self.toggle_button.setStyleSheet("text-align: left; background-color: #3a3a3a; color: white; padding: 5px; border: none;")
        self.toggle_button.setStyleSheet("""
            text-align: left;
            background-color: #3a3a3a;
            color: white;
            padding: 5px;
            font-size: 14px;
            border: none;
        """)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.clicked.connect(self.toggle)
        self.title = title
        self.update_toggle_button_text()

        self.content_area = QWidget()
        self.content_area.setVisible(False)

        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.content_area)
        self.setLayout(self.main_layout)

    def toggle(self):
        checked = self.toggle_button.isChecked()
        self.content_area.setVisible(checked)
        self.update_toggle_button_text()

    def update_toggle_button_text(self):
        arrow = "▼" if self.toggle_button.isChecked() else "►"
        self.toggle_button.setText(f"{arrow} {self.title}")

    def setContentLayout(self, layout):
        self.content_area.setLayout(layout)

from ctranslate2.converters.transformers import TransformersConverter
from transformers import WhisperForConditionalGeneration, WhisperTokenizer

def check_model_in_cache(model_id):
    
    try:
        from transformers.utils import cached_file
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(model_id)
        # Attempt to get the path to the model file in the cache
        model_file = cached_file(model_id, 'pytorch_model.bin')
        if os.path.exists(model_file):
            print(f"Model found in cache: {model_file}")
            return True
    except Exception as e:
        print(f"Model not found in cache: {e}")
    return False

class TranscriptionThread(QThread):
    progress_signal = pyqtSignal(str, str)  # metrics, transcription
    error_signal = pyqtSignal(str)
    transcription_replace_signal = pyqtSignal(str)  # for whisper.cpp output files

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._is_running = True
        self.process = None

    def format_time(self, time_str):
        if not time_str:
            return ''
        try:
            # Replace comma with period if present
            time_str = time_str.replace(',', '.')
            # Convert to float and format to 3 decimal places
            return f"{float(time_str):.3f}"
        except ValueError:
            return None
    
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
            start_time = self.args.get('start_time', '')
            end_time = self.args.get('end_time', '')
            
            start_time = self.format_time(self.args.get('start_time', ''))
            end_time = self.format_time(self.args.get('end_time', ''))
            
            if start_time is None or end_time is None:
                print ("no valid trim times (in seconds) provided, defaulting to none.")
                start_time = ''
                end_time = ''

            python_executable = sys.executable

            cmd = [
                python_executable,
                '-u',
                'transcribe_worker.py',
                '--model-id', model_id,
                '--backend', backend,
                '--device', device_arg,
            ]

            if 'preprocessor_path' in self.args and self.args['preprocessor_path']:
                cmd.extend(['--preprocessor-path', self.args['preprocessor_path']])
            
            if 'original_model_id' in self.args and self.args['original_model_id']:
                cmd.extend(['--original-model-id', self.args['original_model_id']])

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
            if start_time:
                cmd.extend(['--start-time', start_time])
            if end_time:
                cmd.extend(['--end-time', end_time])

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
            )

            timecode_pattern_old = re.compile(
                r'^\['
                r'(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{3}|\d+\.\d+)'
                r'\s-->\s'
                r'(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{3}|\d+\.\d+)'
                r'\]\s*(.*)'
            )
            timecode_pattern = re.compile(
                r'^\['
                r'([^\]]+?)'  # Match any characters up to the closing bracket
                r'\]\s*(.*)'  # Capture the rest of the line as transcription text
            )

            in_transcription = False
            output_file = None
            is_whisper_jax = self.args['backend'] == 'whisper-jax'
            whisper_jax_output = ""

            for line in self.process.stdout:
                if not self._is_running:
                    break
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
                    text = match.group(2)
                    self.progress_signal.emit('', text)
                    in_transcription = True
                elif in_transcription and line.strip() != '':
                    self.progress_signal.emit('', line)
                elif line.strip() != '':
                    self.progress_signal.emit(line, '')
                    in_transcription = False

            # Wait for the process to complete
            self.process.stdout.close()
            self.process.wait()

            if self.process.returncode != 0:
                error_msg = self.process.stderr.read()
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
            pass
    
    def stop(self):
        self._is_running = False
        if self.process:
            self.process.terminate()
            self.process.wait()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Susurrus: Audio Transcription")
        self.setMinimumSize(800, 600)
        self.setAcceptDrops(True)  # Enable drag and drop
        self.init_ui()
        # Keep track of the transcription thread
        self.thread = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            file_path = urls[0].toLocalFile()
            self.audio_input_path.setText(file_path)
    
    def get_original_model_id(self, model_id):
        # Search the backend_model_map for the original model ID
        for backend_models in self.backend_model_map.values():
            for model_tuple in backend_models:
                if isinstance(model_tuple, tuple) and model_tuple[0] == model_id:
                    return model_tuple[1]
                elif model_tuple == model_id:
                    return model_id  # It's already the original model

        # If not found, use improved heuristics
        model_id_lower = model_id.lower()

        # Handle special cases first
        if model_id.startswith("openai/whisper-"):
            return model_id  # It's already an OpenAI Whisper model
        
        if "endpoint" in model_id_lower:
            return "openai/whisper-large-v2"  # Default for endpoints

        # Check for specific version numbers
        if "v3_turbo" in model_id_lower:
            base = "openai/whisper-large-v3"
        elif "v3" in model_id_lower:
            base = "openai/whisper-large-v3"
        elif "v2" in model_id_lower:
            base = "openai/whisper-large-v2"
        elif "v1" in model_id_lower:
            base = "openai/whisper-large-v1"
        else:
            base = "openai/whisper"

        # Check for model size
        if "large" in model_id_lower:
            size = "large"
        elif "medium" in model_id_lower:
            size = "medium"
        elif "small" in model_id_lower:
            size = "small"
        elif "base" in model_id_lower:
            size = "base"
        elif "tiny" in model_id_lower:
            size = "tiny"
        else:
            # Default to large if size is not specified
            size = "large"

        # Check for language specificity
        if "_en" in model_id_lower or ".en" in model_id_lower:
            lang = ".en"
        else:
            lang = ""

        # Construct the original model ID
        if "v3" in base or "v2" in base or "v1" in base:
            return f"{base}{lang}"
        else:
            return f"{base}-{size}{lang}"

    def find_or_convert_ctranslate2_model(self, model_id):
        # Extract the original model ID
        original_model_id = self.get_original_model_id(model_id)
        print(f"Original model ID determined as: {original_model_id}")

        model_dir_name = model_id.replace('/', '_')
        local_model_dir = os.path.join(os.getcwd(), 'ctranslate2_models', model_dir_name)
        local_model_bin_path = os.path.join(local_model_dir, 'model.bin')

        # First, check if model.bin exists locally and is not empty
        if os.path.exists(local_model_bin_path) and os.path.getsize(local_model_bin_path) > 0:
            if os.path.islink(local_model_bin_path):
                # If it's a symlink, replace it with the actual file
                real_path = os.path.realpath(local_model_bin_path)
                os.remove(local_model_bin_path)
                shutil.copy(real_path, local_model_bin_path)
                print(f"Replaced symlink with actual file for model.bin in: {local_model_dir}")
            else:
                print(f"Model already converted and exists locally in: {local_model_dir}")
            return local_model_dir, original_model_id

        else:
            # If the directory exists but model.bin is missing or empty, remove the directory
            if os.path.exists(local_model_dir):
                shutil.rmtree(local_model_dir)
            os.makedirs(local_model_dir, exist_ok=True)

        # Check if the model is already in CTranslate2 format in the Hugging Face repo
        try:
            from huggingface_hub import hf_hub_download, HfApi
            
            api = HfApi()
            model_files = api.list_repo_files(model_id)
            if 'model.bin' in model_files:
                print(f"Found pre-converted CTranslate2 model in Hugging Face repo: {model_id}")
                # Download the model.bin and other necessary files
                for file in ['model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json', 'preprocessor_config.json']:
                    if file in model_files:
                        file_path = hf_hub_download(repo_id=model_id, filename=file, local_dir=local_model_dir)
                        # If the downloaded file is a symlink, replace it with the actual file
                        if os.path.islink(file_path):
                            real_path = os.path.realpath(file_path)
                            os.remove(file_path)
                            shutil.copy(real_path, file_path)
                return local_model_dir, local_model_dir
            
        except Exception as e:
            print(f"Error while checking Hugging Face repo: {e}")
            print("Proceeding with local model conversion...")

        # If we reach here, we need to convert the model
        print(f"Model not found in CTranslate2 format or couldn't be downloaded. Converting {model_id} to CTranslate2 format...")

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
            raise Exception("Quantization selection cancelled by user.")

        reply = QMessageBox.question(
            self,
            "Model Conversion Required",
            f"The model {model_id} needs to be converted to CTranslate2 format with quantization '{quantization}'. This may take several minutes. Do you want to proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.No:
            raise Exception("Model conversion cancelled by user.")

        # Perform the model conversion
        try:
            self.convert_model_to_ctranslate2(model_id, local_model_dir, quantization)
            print(f"Model converted and saved to: {local_model_dir}")
        except Exception as e:
            print(f"Error during model conversion: {e}")
            raise

        # Download the preprocessor files and save them in the same directory
        try:
            from transformers import WhisperProcessor
            print(f"Downloading preprocessor files for model: {model_id}")
            preprocessor = WhisperProcessor.from_pretrained(model_id)
            preprocessor.save_pretrained(local_model_dir)
            print(f"Preprocessor files saved to: {local_model_dir}")
        except Exception as e:
            print(f"Error downloading preprocessor files: {e}")
            raise

        # Download the preprocessor files and save them in the same directory
        try:
            from transformers import WhisperProcessor
            print(f"Downloading preprocessor files for original model: {original_model_id}")
            preprocessor = WhisperProcessor.from_pretrained(original_model_id)
            preprocessor.save_pretrained(local_model_dir)
            print(f"Preprocessor files saved to: {local_model_dir}")
        except Exception as e:
            print(f"Error downloading preprocessor files: {e}")
            raise

        return local_model_dir, original_model_id  # Return both values

    def convert_model_to_ctranslate2(self, model_id, output_dir, quantization):
        from ctranslate2.converters.transformers import TransformersConverter
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        
        print(f"Loading model {model_id} for conversion...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        
        print(f"Converting model to CTranslate2 format...")
        converter = TransformersConverter(model, processor)
        converter.convert(output_dir, quantization=quantization, force=True)
        
        # Verify that the model.bin file was created and is not a symlink
        model_bin_path = os.path.join(output_dir, 'model.bin')
        if not os.path.exists(model_bin_path) or os.path.getsize(model_bin_path) == 0:
            raise Exception(f"Failed to convert model. model.bin not found or empty in {output_dir}")
        if os.path.islink(model_bin_path):
            real_path = os.path.realpath(model_bin_path)
            os.remove(model_bin_path)
            shutil.copy(real_path, model_bin_path)
        
        print(f"Model successfully converted and saved to: {output_dir}")
    
    def check_transcribe_button_state(self):
        if self.audio_input_path.text().strip() or self.audio_url.text().strip():
            self.transcribe_button.setEnabled(True)
        else:
            self.transcribe_button.setEnabled(False)
    
    def init_ui(self):

        self.backend_model_map = {
            'mlx-whisper': [
                ("mlx-community/whisper-large-v3-turbo", "openai/whisper-large-v3"),
                ("mlx-community/whisper-large-v3-turbo-q4", "openai/whisper-large-v3"),
                ("mlx-community/whisper-tiny-mlx-4bit", "openai/whisper-tiny"),
                ("mlx-community/whisper-base-mlx-4bit", "openai/whisper-base"),
                ("mlx-community/whisper-small-mlx-q4", "openai/whisper-small"),
                ("mlx-community/whisper-medium-mlx-4bit", "openai/whisper-medium"),
                ("mlx-community/whisper-large-v3-mlx-4bit", "openai/whisper-large-v3"),
                ("mlx-community/whisper-large-v3-mlx", "openai/whisper-large-v3")
            ],
            'faster-batched': [
                ("cstr/whisper-large-v3-turbo-int8_float32", "openai/whisper-large-v3"),
                ("SYSTRAN/faster-whisper-large-v1", "openai/whisper-large-v2"),
                ("GalaktischeGurke/primeline-whisper-large-v3-german-ct2", "openai/whisper-large-v3")
            ],
            'faster-sequenced': [
                ("cstr/whisper-large-v3-turbo-int8_float32", "openai/whisper-large-v3"),
                ("SYSTRAN/faster-whisper-large-v1", "openai/whisper-large-v2"),
                ("GalaktischeGurke/primeline-whisper-large-v3-german-ct2", "openai/whisper-large-v3")
            ],
            'transformers': [
                ("openai/whisper-large-v3", "openai/whisper-large-v3"),
                ("openai/whisper-large-v2", "openai/whisper-large-v2"),
                ("openai/whisper-medium", "openai/whisper-medium"),
                ("openai/whisper-small", "openai/whisper-small")
            ],
            'OpenAI Whisper': [
                ("large-v2", "openai/whisper-large-v2"),
                ("medium", "openai/whisper-medium"),
                ("small", "openai/whisper-small"),
                ("base", "openai/whisper-base"),
                ("tiny", "openai/whisper-tiny")
            ],
            'whisper.cpp': [
                ("large-v3-turbo-q5_0", "openai/whisper-large-v3"),
                ("large-v3-turbo", "openai/whisper-large-v3"),
                ("small", "openai/whisper-small"),
                ("base", "openai/whisper-base"),
                ("tiny", "openai/whisper-tiny"),
                ("tiny.en", "openai/whisper-tiny.en")
            ],
            'ctranslate2': [
                ("cstr/whisper-large-v3-turbo-int8_float32", "openai/whisper-large-v3"),
                ("SYSTRAN/faster-whisper-large-v1", "openai/whisper-large-v2"),
                ("GalaktischeGurke/primeline-whisper-large-v3-german-ct2", "openai/whisper-large-v3")
            ],
            'whisper-jax': [
                ("openai/whisper-tiny", "openai/whisper-tiny"),
                ("openai/whisper-medium", "openai/whisper-medium"),
                ("tiny.en", "openai/whisper-tiny.en"),
                ("base.en", "openai/whisper-base.en"),
                ("small.en", "openai/whisper-small.en"),
                ("medium.en", "openai/whisper-medium.en"),
                ("large-v2", "openai/whisper-large-v2")
            ],
            'insanely-fast-whisper': [
                ("openai/whisper-large-v3", "openai/whisper-large-v3"),
                ("openai/whisper-medium", "openai/whisper-medium"),
                ("openai/whisper-small", "openai/whisper-small"),
                ("openai/whisper-base", "openai/whisper-base"),
                ("openai/whisper-tiny", "openai/whisper-tiny"),
            ],
        }
        
        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel("<h1 style='color: #FFFFFF;'>Susurrus: Audio Transcription</h1>")
        subtitle_label = QLabel("<p style='color: #666666;'>Transcribe audio using various (Whisper-) backends.</p>")
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
        self.audio_url.textChanged.connect(self.toggle_proxy_settings)  # Connect the function

        input_row.addWidget(QLabel("Audio File:"))
        input_row.addWidget(self.audio_input_path)
        input_row.addWidget(self.audio_input_button)
        input_row.addWidget(QLabel("or URL:"))
        input_row.addWidget(self.audio_url)

        main_layout.addLayout(input_row)

        # Transcribe and Abort Buttons
        button_layout = QHBoxLayout()

        self.transcribe_button = QPushButton(QIcon.fromTheme("media-playback-start"), "Transcribe")
        self.transcribe_button.setEnabled(False)
        self.transcribe_button.clicked.connect(self.start_transcription)

        self.abort_button = QPushButton(QIcon.fromTheme("media-playback-stop"), "Abort")
        self.abort_button.setEnabled(False)
        self.abort_button.clicked.connect(self.abort_transcription)

        self.save_button = QPushButton(QIcon.fromTheme("document-save"), "")
        self.save_button.setToolTip("Save Transcription")
        self.save_button.clicked.connect(self.save_transcription)

        button_layout.addStretch()

        button_layout.addWidget(self.transcribe_button)
        button_layout.addWidget(self.abort_button)
        button_layout.addWidget(self.save_button)

        main_layout.addLayout(button_layout)

        self.audio_input_path.textChanged.connect(self.check_transcribe_button_state)
        self.audio_url.textChanged.connect(self.check_transcribe_button_state)

        # Advanced Options Group
        # Advanced Options Collapsible Box
        self.advanced_options_box = CollapsibleBox("Advanced Options")
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

        self.proxy_row_widget = QWidget()
        self.proxy_row_widget.setLayout(proxy_row)
        self.proxy_row_widget.setVisible(False)  # Hide initially

        advanced_layout.addWidget(self.proxy_row_widget)

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
            "insanely-fast-whisper", 
        ])

        self.backend_selection.setCurrentText("mlx-whisper")

        backend_row.addWidget(QLabel("Backend:"))
        backend_row.addWidget(self.backend_selection)

        advanced_layout.addLayout(backend_row)

        # Connect the backend selection change to the update_model_options method
        self.backend_selection.currentTextChanged.connect(self.update_model_options)

        # Model, Device, and Language selection row
        model_row = QHBoxLayout()

        model_row.addWidget(QLabel("Model:"))
        self.model_id = QComboBox()
        # Set initial models for the default backend
        models = self.backend_model_map[self.backend_selection.currentText()]
        self.model_id.addItems([model_tuple[0] for model_tuple in models])
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

        self.chunk_row_widget = QWidget()
        self.chunk_row_widget.setLayout(chunk_row)
        self.chunk_row_widget.setVisible(False)
        advanced_layout.addWidget(self.chunk_row_widget)

        # Start Time and End Time Row
        misc_row = QHBoxLayout()
        self.start_time = QLineEdit()
        self.start_time.setPlaceholderText("Start Time (seconds)")
        self.end_time = QLineEdit()
        self.end_time.setPlaceholderText("End Time (seconds)")

        misc_row.addWidget(QLabel("Start Time (s):"))
        misc_row.addWidget(self.start_time)
        misc_row.addWidget(QLabel("End Time (s):"))
        misc_row.addWidget(self.end_time)

        advanced_layout.addLayout(misc_row)

        # Set the content layout for the collapsible box
        self.advanced_options_box.setContentLayout(advanced_layout)
        main_layout.addWidget(self.advanced_options_box)

        # Output Row
        output_layout = QHBoxLayout()
        metrics_layout = QVBoxLayout()
        metrics_label = QLabel("Metrics")
        self.metrics_output = QPlainTextEdit()
        self.metrics_output.setReadOnly(True)
        self.metrics_output.setMaximumWidth(300)  # Limit width of metrics
        metrics_layout.addWidget(metrics_label)
        metrics_layout.addWidget(self.metrics_output)

        transcription_layout = QVBoxLayout()
        transcription_label = QLabel("Transcription")
        self.transcription_output = QPlainTextEdit()
        self.transcription_output.setReadOnly(True)
        transcription_layout.addWidget(transcription_label)
        transcription_layout.addWidget(self.transcription_output)

        output_layout.addLayout(metrics_layout, 1)
        output_layout.addLayout(transcription_layout, 3)

        main_layout.addLayout(output_layout)

        # Set a modern style
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                font-size: 14px;
            }
            QPushButton {
                background-color: #4a4a4a;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QLineEdit, QComboBox, QPlainTextEdit {
                background-color: #3a3a3a;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
                font-size: 14px;
            }
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 4px;
                margin-top: 20px;
                padding-top: 15px;
                padding-bottom: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 7px;
                padding: 0 5px 0 5px;
                background-color: #2b2b2b;
                font-size: 16px;
            }
        """)

        # Set window size
        self.setMinimumSize(800, 600)

        # Adjust main layout margins and spacing
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        self.toggle_proxy_settings()  # Ensure proxy settings are correctly shown or hidden

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress bar
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        self.setLayout(main_layout)

    def toggle_proxy_settings(self):
        if self.audio_url.text().strip():
            self.proxy_row_widget.setVisible(True)
        else:
            self.proxy_row_widget.setVisible(False)
            
    def select_audio_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.mp3 *.wav *.flac)")
        if file_name:
            self.audio_input_path.setText(file_name)

    def update_model_options(self, backend):
        models = self.backend_model_map.get(backend, [])
        self.model_id.clear()
        for model_tuple in models:
            model_id = model_tuple[0]
            self.model_id.addItem(model_id)

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
        self.save_button.setEnabled(True)
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
        
        # Normalize backend and device arguments
        args['backend'] = args['backend'].strip().lower()
        args['device_arg'] = args['device_arg'].strip().lower()
        print(f"Backend selected: '{args['backend']}'")
        print(f"Device selected: '{args['device_arg']}'")
        print(f"Model selected: '{args['model_id']}'")

        # Handle ctranslate2 specific logic
        quantization = None
        if args['backend'] == 'ctranslate2':
            print("ctranslate2 selected. checking...")
            # Force device to CPU if device is MPS
            if args['device_arg'] == 'mps':
                args['device_arg'] = 'cpu'
            
            # Find or convert the model
            try:
                model_dir, original_model_id = self.find_or_convert_ctranslate2_model(args['model_id'])
                args['model_id'] = model_dir  # Update model_id to point to the model directory
                args['original_model_id'] = original_model_id
                args['preprocessor_path'] = model_dir  # Set the preprocessor path to the same directory
                print(f"Using model directory: {args['model_id']}")

                # Check if preprocessor files exist
                preprocessor_files = ["tokenizer.json", "vocabulary.json", "tokenizer_config.json"]
                preprocessor_missing = not all(os.path.exists(os.path.join(model_dir, f)) for f in preprocessor_files)

                if preprocessor_missing:
                    print(f"Preprocessor files missing. Downloading from original model ID: {original_model_id}")
                    # No need to determine original_model_id again; use the one retrieved earlier
                    try:
                        from transformers import WhisperProcessor
                        preprocessor = WhisperProcessor.from_pretrained(original_model_id)
                        preprocessor.save_pretrained(model_dir)
                        print(f"Preprocessor files saved to: {model_dir}")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Failed to download preprocessor files: {str(e)}")
                        self.progress_bar.setVisible(False)
                        self.transcribe_button.setEnabled(True)
                        return
                else:
                    print("Preprocessor files already exist.")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Model preparation failed: {str(e)}")
                self.progress_bar.setVisible(False)
                self.transcribe_button.setEnabled(True)
                return
            
            # Check if model.bin exists
            model_bin_path = os.path.join(model_dir, 'model.bin')
            if not os.path.exists(model_bin_path):
                # Proceed with model conversion
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
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    self.progress_bar.setVisible(False)
                    self.transcribe_button.setEnabled(True)
                    return
                # Pass quantization to the worker script
                args['quantization'] = quantization
                # Perform the model conversion here
                try:
                    self.convert_model_to_ctranslate2(args['model_id'], model_dir, quantization)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Model conversion failed: {str(e)}")
                    self.progress_bar.setVisible(False)
                    self.transcribe_button.setEnabled(True)
                    return
            else:
                print(f"Model already converted and exists in: {model_dir}")
            args['model_id'] = model_dir  # Update model_id to point to the converted model directory


        self.thread = TranscriptionThread(args)
        self.thread.progress_signal.connect(self.update_outputs)
        self.thread.error_signal.connect(self.show_error)
        self.thread.finished.connect(self.on_transcription_finished)
        
        # must we replace the content of the transcription with a file from whisper.cpp?
        self.thread.transcription_replace_signal.connect(self.replace_transcription_output)  # signal for when we read from whisper.cpp output file
        
        self.thread.start()
        self.transcribe_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.abort_button.setEnabled(True)

        self.transcription_output.clear()
        self.metrics_output.clear()
        self.transcription_text = ""

    def abort_transcription(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None
        self.transcribe_button.setEnabled(True)
        self.abort_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.metrics_output.appendPlainText("Transcription aborted by user.")
    
    def on_transcription_finished(self):
        self.transcribe_button.setEnabled(True)
        self.abort_button.setEnabled(False)
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
            self.save_button.setEnabled(True)
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
