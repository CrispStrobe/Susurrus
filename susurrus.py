import sys
import os
import logging
import subprocess
import re
import shutil
import platform
import threading
import json

def safe_create_symlink(src, dst):
    try:
        os.symlink(src, dst)
    except OSError as e:
        if e.winerror == 1314:
            logging.warning(f"Insufficient privileges to create symlink from {src} to {dst}. Copying file instead.")
            shutil.copy(src, dst)
        else:
            logging.error(f"Failed to create symlink from {src} to {dst}: {e}")
            raise

def diagnose_pytorch():
    import sys
    import platform
    import logging

    logging.info(f"Python version: {sys.version}")
    logging.info(f"Platform: {platform.platform()}")

    try:
        import torch
        logging.info(f"PyTorch version: {torch.__version__}")

        # Check CUDA availability
        logging.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

        # Check CUDA version PyTorch was built with
        logging.info(f"PyTorch CUDA version: {torch.version.cuda}")

        # Get NVIDIA driver version
        if hasattr(torch.version, 'cuda') and torch.cuda.is_available():
            logging.info(f"NVIDIA driver version: {torch.cuda.get_device_properties(0).name}")
        else:
            logging.info("No CUDA driver found")

        # Try importing CUDA toolkit
        try:
            import nvidia.cuda
            logging.info("CUDA toolkit is installed")
        except ImportError:
            logging.info("CUDA toolkit not found in Python environment")

    except ImportError:
        logging.error("PyTorch is not installed")
        return False

    return True

def is_diarization_available():
    """Check if diarization functionality is available without importing pyannote.audio directly"""
    try:
        # Use importlib.util to avoid actually loading the module
        import importlib.util
        import sys
        
        # Check if the module is available
        spec = importlib.util.find_spec("pyannote.audio")
        if spec is None:
            return False
            
        # Check for HF_TOKEN which is required
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Error checking diarization availability: {str(e)}")
        return False
    
def check_ffmpeg_installation():
    """Check if ffmpeg is properly installed and working."""
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        if result.returncode == 0:
            version_str = result.stdout.split('\n')[0]
            logging.info(f"ffmpeg is installed: {version_str}")
            return True
        else:
            logging.warning("ffmpeg command failed. Output formats may be limited.")
            return False
    except FileNotFoundError:
        logging.warning("ffmpeg not found. Some audio formats may not be supported.")
        return False
    except Exception as e:
        logging.warning(f"Error checking ffmpeg: {str(e)}")
        return False

def check_developer_mode():
    if platform.system() == 'Windows':
        try:
            import ctypes
            # Check if process has admin rights
            is_admin = ctypes.windll.shell32.IsUserAnAdmin()
            if not is_admin:
                logging.warning("Python is not running with administrator privileges")

            # Check Developer Mode
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock",
                    0, winreg.KEY_READ)
                value, _ = winreg.QueryValueEx(key, "AllowDevelopmentWithoutDevMode")
                if value != 1:
                    logging.warning("Developer Mode is not enabled")
                    QMessageBox.warning(None, "Developer Mode Not Enabled",
                        "Please enable Developer Mode in Windows Settings:\n"
                        "1. Open Windows Settings\n"
                        "2. Navigate to Privacy & security > For developers\n"
                        "3. Enable 'Developer Mode'\n\n"
                        "This will improve cache performance.")
            except WindowsError as e:
                logging.warning(f"Could not check Developer Mode registry: {e}")

        except Exception as e:
            logging.warning(f"Could not check Developer Mode status: {e}")

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QComboBox, QHBoxLayout, QVBoxLayout, QCheckBox,
    QPlainTextEdit, QMessageBox, QProgressBar, QInputDialog,
    QHeaderView, QTableWidget, QTableWidgetItem, QDialogButtonBox,
    QDialog, QMenuBar, QMenu
)
from PyQt6.QtCore import QThread, pyqtSignal, QSettings, Qt
from PyQt6.QtGui import QIcon, QAction, QColor

# Set up logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_default_device():
    cuda_available = check_cuda()
    if cuda_available:
        return "GPU"
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "MPS"
    except ImportError:
        pass
    return "CPU"

def check_cuda():
    try:
        import torch

        # Force CUDA initialization
        if torch.cuda.is_available():
            # Get CUDA device count
            device_count = torch.cuda.device_count()

            # Get CUDA device properties
            if device_count > 0:
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda

                logging.info(f"CUDA is available with {device_count} device(s)")
                logging.info(f"Primary GPU: {device_name}")
                logging.info(f"CUDA Version: {cuda_version}")

                # Test CUDA by creating a small tensor
                try:
                    test_tensor = torch.tensor([1.0], device='cuda')
                    logging.info("CUDA test successful - GPU is working")
                    return True
                except RuntimeError as e:
                    logging.error(f"CUDA test failed: {e}")
                    return False
            else:
                logging.warning("CUDA is available but no GPU devices found")
                return False
        else:
            logging.warning("CUDA is not available")
            return False

    except ImportError as e:
        logging.warning(f"PyTorch import failed: {e}")
        return False
    except Exception as e:
        logging.warning(f"CUDA check failed: {e}")
        return False

def check_dependencies():
    """Check for required dependencies with safer import handling"""
    
    dependencies = {
        "PyTorch": {
            "required": True,
            "installed": False,
            "version": None,
            "module": "torch",
            "message": "PyTorch is essential for audio processing and transcription"
        },
        "Transformers": {
            "required": True,
            "installed": False,
            "version": None,
            "module": "transformers",
            "message": "Transformers is required for Whisper models"
        },
        "PyAnnote Audio": {
            "required": False,
            "installed": False,
            "version": None,
            "module": "pyannote.audio",
            "message": "Required for speaker diarization"
        },
        "Pydub": {
            "required": True,
            "installed": False,
            "version": None,
            "module": "pydub",
            "message": "Required for audio file processing"
        },
        "Hugging Face Hub": {
            "required": False,
            "installed": False,
            "version": None,
            "module": "huggingface_hub",
            "message": "Required for model downloading"
        },
        "NumPy": {
            "required": True,
            "installed": False,
            "version": None,
            "module": "numpy",
            "message": "Required for audio processing"
        }
    }
    
    # Check each dependency with safer approach
    for name, info in dependencies.items():
        try:
            # Use importlib for safer importing
            import importlib
            module = importlib.import_module(info["module"])
            info["installed"] = True
            
            # Get version if available
            if hasattr(module, "__version__"):
                info["version"] = module.__version__
            elif hasattr(module, "version"):
                info["version"] = module.version
            
            version_str = f" v{info['version']}" if info["version"] else ""
            logging.info(f"{name}{version_str} is available")
            
            # Special checks for specific modules
            if name == "PyTorch":
                # Safer CUDA check that won't crash
                try:
                    cuda_available = module.cuda.is_available()
                    logging.info(f"CUDA available: {cuda_available}")
                except:
                    logging.info("Could not check CUDA availability")
                
                # Safer MPS check that won't crash
                try:
                    if hasattr(module.backends, 'mps'):
                        mps_available = module.backends.mps.is_available()
                        logging.info(f"MPS available: {mps_available}")
                except:
                    logging.info("Could not check MPS availability")
                
        except ImportError:
            info["installed"] = False
            if info["required"]:
                logging.warning(f"{name} not found. {info['message']}")
            else:
                logging.info(f"{name} not found. {info['message']} (optional)")
        except Exception as e:
            # Instead of allowing the exception to propagate, just mark as not installed
            # and log the error
            info["installed"] = False
            logging.warning(f"Error checking {name}: {str(e)}")
            if info["required"]:
                logging.warning(f"{name} might not be usable. {info['message']}")
            else:
                logging.info(f"{name} might not be usable. {info['message']} (optional)")
    
    # Check for pyannote.audio version specifically while avoiding the full import
    # which could cause the dependency conflict
    try:
        import importlib.metadata
        pyannote_version = importlib.metadata.version("pyannote.audio")
        dependencies["PyAnnote Audio"]["installed"] = True
        dependencies["PyAnnote Audio"]["version"] = pyannote_version
        logging.info(f"PyAnnote Audio v{pyannote_version} is installed")
    except:
        # Already handled in the main loop
        pass
    
    # Check for ffmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        if result.returncode == 0:
            version_str = result.stdout.split('\n')[0]
            logging.info(f"ffmpeg is installed: {version_str}")
            dependencies["ffmpeg"] = {
                "required": True,
                "installed": True,
                "version": version_str.split(' ')[2],
                "message": "Required for audio format conversion"
            }
        else:
            logging.warning("ffmpeg command failed. Audio format support may be limited.")
            dependencies["ffmpeg"] = {
                "required": True,
                "installed": False,
                "message": "Required for audio format conversion"
            }
    except FileNotFoundError:
        logging.warning("ffmpeg not found. Audio format support may be limited.")
        dependencies["ffmpeg"] = {
            "required": True,
            "installed": False,
            "message": "Required for audio format conversion"
        }
    except Exception as e:
        logging.warning(f"Error checking ffmpeg: {str(e)}")
        dependencies["ffmpeg"] = {
            "required": True,
            "installed": False,
            "message": "Required for audio format conversion"
        }
    
    # Check for HF_TOKEN in environment
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        logging.info("Hugging Face token found in environment.")
        dependencies["HF_TOKEN"] = {
            "required": False,
            "installed": True,
            "message": "Required for speaker diarization"
        }
    else:
        logging.info("No Hugging Face token found in environment. You'll need to provide one for speaker diarization.")
        dependencies["HF_TOKEN"] = {
            "required": False,
            "installed": False,
            "message": "Required for speaker diarization"
        }
    
    # Return the dependency dict for potential UI display
    return dependencies

def get_default_backend():
    system = platform.system().lower()

    if system == 'windows':
        cuda_available = check_cuda()
        if cuda_available:
            logging.info("Using CUDA GPU with faster-batched backend")
            return 'faster-batched'
        else:
            logging.info("No working CUDA GPU detected, falling back to CPU with whisper.cpp")
            return 'whisper.cpp'
    elif system == 'darwin':
        try:
            import mlx
            return 'mlx-whisper'
        except ImportError:
            return 'faster-batched'
    else:  # Linux and others
        cuda_available = check_cuda()
        if cuda_available:
            return 'faster-batched'
        else:
            return 'whisper.cpp'

def get_default_model_for_backend(backend):
    defaults = {
        'mlx-whisper': "mlx-community/whisper-tiny-mlx-4bit",
        'faster-batched': "cstr/whisper-large-v3-turbo-int8_float32",
        'faster-sequenced': "cstr/whisper-large-v3-turbo-int8_float32",
        'whisper.cpp': "tiny",
        'transformers': "openai/whisper-tiny",
        'OpenAI Whisper': "tiny",
        'ctranslate2': "cstr/whisper-large-v3-turbo-int8_float32",
        'whisper-jax': "openai/whisper-tiny",
        'insanely-fast-whisper': "openai/whisper-tiny"
    }
    return defaults.get(backend, "tiny")

class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.toggle_button = QPushButton()
        self.toggle_button.setStyleSheet(self.get_button_style())
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self.toggle)
        self.title = title
        self.update_toggle_button_text()

        self.content_area = QWidget()
        self.content_area.setVisible(False)
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.content_area)
        self.setLayout(self.main_layout)

    def get_button_style(self):
        return """
            text-align: left;
            background-color: #3a3a3a;
            color: white;
            padding: 5px;
            font-size: 14px;
            border: none;
        """

    def toggle(self):
        self.content_area.setVisible(self.toggle_button.isChecked())
        self.update_toggle_button_text()

    def update_toggle_button_text(self):
        arrow = "▼" if self.toggle_button.isChecked() else "►"
        self.toggle_button.setText(f"{arrow} {self.title}")

    def setContentLayout(self, layout):
        self.content_area.setLayout(layout)


class TranscriptionThread(QThread):
    progress_signal = pyqtSignal(str, str)  # metrics, transcription
    error_signal = pyqtSignal(str)
    transcription_replace_signal = pyqtSignal(str)  # For whisper.cpp output files and diarization
    diarization_signal = pyqtSignal(str)  # For diarization status updates

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
            # Check if diarization is enabled
            if self.args.get('diarization_enabled', False):
                self._run_diarization()
            else:
                self._run_standard_transcription()
                
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            logging.error(error_msg)
            self.error_signal.emit(error_msg)

    def _run_diarization(self):
        """Run transcription with speaker diarization"""
        try:
            # Check if diarize_audio module exists
            try:
                import importlib.util
                spec = importlib.util.find_spec("diarize_audio")
                if spec is None:
                    raise ImportError("No module named 'diarize_audio'")
            except ImportError as e:
                # Module not found, we need to show an error and abort
                error_msg = f"Failed to import diarize_audio module: {str(e)}"
                logging.error(error_msg)
                self.error_signal.emit(error_msg)
                self.error_signal.emit("Please create diarize_audio.py and diarize_worker.py files as described in the documentation")
                return
            
            # Extract arguments
            audio_input = self.args['audio_input']
            hf_token = self.args['hf_token']
            model_id = self.args['model_id']
            language = self.args['language']
            backend = self.args['backend']
            device_arg = self.args['device_arg']
            min_speakers = self.args.get('min_speakers')
            max_speakers = self.args.get('max_speakers')
            output_format = self.args.get('output_format', 'txt')
            diarization_model = self.args.get('diarization_model', 'Default')
            
            # Convert to valid format for command line
            min_speakers_arg = f"--min-speakers {min_speakers}" if min_speakers else ""
            max_speakers_arg = f"--max-speakers {max_speakers}" if max_speakers else ""
            language_arg = f"--language {language}" if language else ""
            
            python_executable = sys.executable
            
            # Check if diarize_worker.py exists
            diarize_worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diarize_worker.py')
            if not os.path.exists(diarize_worker_path):
                # Try looking in the current directory
                diarize_worker_path = 'diarize_worker.py'
                if not os.path.exists(diarize_worker_path):
                    self.error_signal.emit("diarize_worker.py not found. Please create this file.")
                    return
            
            # Build the command to run diarize_worker.py
            cmd = [
                python_executable,
                '-u',
                diarize_worker_path,
                '--audio-input', audio_input,
                '--hf-token', hf_token,
                '--transcribe',
                '--model-id', model_id,
                '--backend', backend,
                '--device', device_arg,
                '--output-formats', output_format,
                '--diarization-model', diarization_model
            ]
            
            # Add optional arguments
            if min_speakers:
                cmd.extend(['--min-speakers', str(min_speakers)])
            
            if max_speakers:
                cmd.extend(['--max-speakers', str(max_speakers)])
                
            if language:
                cmd.extend(['--language', language])
            
            self.progress_signal.emit("Starting speaker diarization...", "")
            self.progress_signal.emit(f"Using diarization model: {diarization_model}", "")
            
            # Run the diarization worker
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=True
            )
            
            # Track output files found in the output
            output_files = {}
            diarization_json = None
            
            # Read from stderr and emit metrics
            def read_stderr():
                for line in iter(self.process.stderr.readline, ''):
                    if not self._is_running:
                        break
                    self.progress_signal.emit(line.strip(), '')  # Emit logs to metrics
            
            # Start a separate thread to read stderr
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()
            
            # Process stdout
            for line in iter(self.process.stdout.readline, ''):
                if not self._is_running:
                    break
                
                line = line.strip()
                
                # Check for output file markers
                if line.startswith("OUTPUT FILE ("):
                    # Parse output file path
                    match = re.match(r"OUTPUT FILE \((.+)\): (.+)", line)
                    if match:
                        format_type, file_path = match.groups()
                        output_files[format_type.lower()] = file_path
                        self.progress_signal.emit(f"Generated {format_type} transcript: {file_path}", "")
                
                elif line.startswith("DIARIZATION JSON:"):
                    diarization_json = line.split(":", 1)[1].strip()
                    self.progress_signal.emit(f"Diarization data saved to: {diarization_json}", "")
                
                else:
                    self.progress_signal.emit(line, "")
            
            # Wait for process to complete
            self.process.wait()
            stderr_thread.join(timeout=1)
            
            if self.process.returncode != 0:
                self.error_signal.emit("Speaker diarization process failed")
                return
            
            # Load and display the transcript if available
            if output_format.lower() in output_files:
                file_path = output_files[output_format.lower()]
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        transcript = f.read()
                        self.transcription_replace_signal.emit(transcript)
                except Exception as e:
                    self.error_signal.emit(f"Failed to read transcript file: {str(e)}")
            
            # Load and display diarization information
            if diarization_json:
                try:
                    with open(diarization_json, 'r', encoding='utf-8') as f:
                        diarization_data = json.load(f)
                        speaker_count = len(set(segment["speaker"] for segment in diarization_data))
                        segment_count = len(diarization_data)
                        self.diarization_signal.emit(
                            f"Diarization successful. Found {speaker_count} speakers in {segment_count} segments."
                        )
                except Exception as e:
                    self.error_signal.emit(f"Failed to read diarization data: {str(e)}")
            
        except Exception as e:
            error_msg = f"Diarization error: {str(e)}"
            logging.error(error_msg)
            self.error_signal.emit(error_msg)

    def _run_standard_transcription(self):
        """Run standard transcription without diarization"""
        try:
            # Extract arguments
            model_id = self.args['model_id']
            word_timestamps = self.args['word_timestamps']
            language = self.args['language']
            backend = self.args['backend']
            device_arg = self.args['device_arg']
            pipeline_type = self.args['pipeline_type']
            max_chunk_length = float(self.args['max_chunk_length']) if self.args['max_chunk_length'] else 0.0
            output_format = self.args['output_format']
            quantization = self.args.get('quantization', None)
            start_time = self.format_time(self.args.get('start_time', ''))
            end_time = self.format_time(self.args.get('end_time', ''))

            if start_time is None or end_time is None:
                logging.info("No valid trim times provided, defaulting to none.")
                start_time = ''
                end_time = ''

            python_executable = sys.executable

            # Build the command to run transcribe_worker.py
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

            if self.args['proxy_url']:
                cmd.extend(['--proxy-url', self.args['proxy_url']])
            if self.args['proxy_username']:
                cmd.extend(['--proxy-username', self.args['proxy_username']])
            if self.args['proxy_password']:
                cmd.extend(['--proxy-password', self.args['proxy_password']])

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
                bufsize=1,
            )

            # read from stderr and emit metrics
            def read_stderr():
                for line in self.process.stderr:
                    if not self._is_running:
                        break
                    line = line.decode('utf-8', errors='replace').rstrip()
                    if line:
                        self.progress_signal.emit(line, '')  # Emit logs to metrics

            # Start a separate thread to read stderr
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()

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
                line = line.decode('utf-8', errors='replace').rstrip()

                if line.startswith('OUTPUT FILE: '):
                    output_file = line[len('OUTPUT FILE: '):].strip()
                    continue

                # Handle download progress or other special lines
                if line.startswith("\rProgress:"):
                    self.progress_signal.emit(line, '')  # Show in metrics window
                    continue

                if is_whisper_jax:
                    if line.startswith("Transcription time:") or line.startswith("Audio file size:") or line.startswith("Total transcription time:"):
                        self.progress_signal.emit(line, '')
                    else:
                        whisper_jax_output += line + "\n"
                        self.progress_signal.emit('', line)
                elif 'Starting transcription' in line or 'Detected language' in line or 'Total transcription time' in line:
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
            stderr_thread.join(timeout=1)

            if self.process.returncode != 0:
                error_msg = "Transcription process failed."
                self.error_signal.emit(error_msg)
            else:
                # Handle the transcription output
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
            logging.error(error_msg)
            self.error_signal.emit(error_msg)

    def stop(self):
        self._is_running = False
        if self.process:
            self.process.terminate()
            self.process.wait()

class DependenciesDialog(QDialog):
    """Dialog to display the status of dependencies"""
    def __init__(self, dependencies, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Susurrus Dependencies")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("<h2>Dependency Status</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Info text
        info = QLabel("The following dependencies are required or recommended for Susurrus:")
        layout.addWidget(info)
        
        # Create a table for dependencies
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Dependency", "Status", "Details"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        
        # Populate table
        table.setRowCount(len(dependencies))
        row = 0
        
        for name, info in dependencies.items():
            # Name
            name_item = QTableWidgetItem(name)
            table.setItem(row, 0, name_item)
            
            # Status
            if info["installed"]:
                status_text = "✓ Installed"
                if info.get("version"):
                    status_text += f" (v{info['version']})"
                status_item = QTableWidgetItem(status_text)
                status_item.setForeground(QColor("green"))
            else:
                if info["required"]:
                    status_text = "❌ Missing"
                    status_item = QTableWidgetItem(status_text)
                    status_item.setForeground(QColor("red"))
                else:
                    status_text = "⚠ Optional"
                    status_item = QTableWidgetItem(status_text)
                    status_item.setForeground(QColor("orange"))
            
            table.setItem(row, 1, status_item)
            
            # Details
            details_item = QTableWidgetItem(info["message"])
            table.setItem(row, 2, details_item)
            
            row += 1
        
        layout.addWidget(table)
        
        # Add installation instructions for missing dependencies
        if any(not info["installed"] and info["required"] for info in dependencies.values()):
            instructions_label = QLabel("<b>Installation Instructions:</b>")
            layout.addWidget(instructions_label)
            
            instructions_text = QLabel(
                "Missing required dependencies can be installed with pip:<br>"
                "<code>pip install torch transformers pydub numpy</code><br><br>"
                "For speaker diarization, also install:<br>"
                "<code>pip install pyannote.audio huggingface_hub</code><br><br>"
                "For ffmpeg, visit <a href='https://ffmpeg.org/download.html'>ffmpeg.org/download.html</a>"
            )
            instructions_text.setTextFormat(Qt.TextFormat.RichText)
            instructions_text.setOpenExternalLinks(True)
            layout.addWidget(instructions_text)
        
        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)

class DiarizationSettingsBox(CollapsibleBox):
    """Collapsible box for speaker diarization settings"""
    
    def __init__(self, parent=None):
        super().__init__("Speaker Diarization", parent)
        layout = QVBoxLayout()
        
        # Enable diarization checkbox
        self.enable_diarization = QCheckBox("Enable Speaker Diarization")
        self.enable_diarization.setToolTip("Identify different speakers in the audio")
        layout.addWidget(self.enable_diarization)
        
        # Hugging Face token input
        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("Hugging Face Token:"))
        self.hf_token = QLineEdit()
        self.hf_token.setPlaceholderText("Enter your Hugging Face API token")
        self.hf_token.setToolTip("Required for speaker diarization. Get it from https://huggingface.co/settings/tokens")
        token_layout.addWidget(self.hf_token)
        
        # Token help button
        self.token_help_button = QPushButton("?")
        self.token_help_button.setMaximumWidth(30)
        self.token_help_button.clicked.connect(self.show_token_help)
        token_layout.addWidget(self.token_help_button)
        
        layout.addLayout(token_layout)
        
        # Diarization model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Diarization Model:"))
        self.diarization_model = QComboBox()
        
        # Add models - will be populated when DiarizationManager is available
        self.diarization_model.addItems([
            "Default",
            "English",
            "Chinese",
            "German", 
            "Spanish",
            "Japanese"
        ])
        
        model_layout.addWidget(self.diarization_model)
        
        # Model help button
        self.model_help_button = QPushButton("?")
        self.model_help_button.setMaximumWidth(30)
        self.model_help_button.clicked.connect(self.show_model_help)
        model_layout.addWidget(self.model_help_button)
        
        layout.addLayout(model_layout)
        
        # Min/Max speakers row
        speakers_layout = QHBoxLayout()
        speakers_layout.addWidget(QLabel("Min. Speakers:"))
        self.min_speakers = QLineEdit()
        self.min_speakers.setPlaceholderText("Auto")
        self.min_speakers.setMaximumWidth(60)
        speakers_layout.addWidget(self.min_speakers)
        
        speakers_layout.addWidget(QLabel("Max. Speakers:"))
        self.max_speakers = QLineEdit()
        self.max_speakers.setPlaceholderText("Auto")
        self.max_speakers.setMaximumWidth(60)
        speakers_layout.addWidget(self.max_speakers)
        
        speakers_layout.addStretch()
        layout.addLayout(speakers_layout)
        
        # Add layout to content area
        self.setContentLayout(layout)

    def show_token_help(self):
        """Show help dialog for Hugging Face token"""
        QMessageBox.information(
            self,
            "Hugging Face Token Help",
            "A Hugging Face API token is required for speaker diarization.\n\n"
            "1. Create a free account at https://huggingface.co\n"
            "2. Go to https://huggingface.co/settings/tokens\n"
            "3. Create a new token with 'read' access\n"
            "4. Copy and paste the token here\n\n"
            "Note: You need to accept the user agreement for the diarization models at "
            "https://huggingface.co/pyannote/speaker-diarization"
        )
    
    def show_model_help(self):
        """Show help dialog for diarization model selection"""
        QMessageBox.information(
            self,
            "Diarization Model Selection",
            "Choose the appropriate diarization model for your audio:\n\n"
            "• Default: General purpose diarization model\n"
            "• English: Optimized for English conversations\n"
            "• Chinese: Optimized for Mandarin Chinese conversations\n"
            "• German: Optimized for German conversations\n"
            "• Spanish: Optimized for Spanish conversations\n"
            "• Japanese: Optimized for Japanese conversations\n\n"
            "Language-specific models may provide better results for their respective languages, "
            "especially for phone calls and naturalistic conversations."
        )
    

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Susurrus: Whisper Audio Transcription with Speaker Diarization")
        self.setMinimumSize(800, 600)
        self.setAcceptDrops(True)  # Enable drag and drop
        self.thread = None
        
        # Initialize QSettings
        self.settings = QSettings("Susurrus", "AudioTranscription")
        
        # Check for token in environment
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            self.has_env_token = True
        else:
            self.has_env_token = False

        # Check system configuration - use a try-except to prevent crashes
        try:
            diagnose_pytorch()
        except Exception as e:
            logging.warning(f"Error during PyTorch diagnosis: {str(e)}")
        
        try:
            check_developer_mode()
        except Exception as e:
            logging.warning(f"Error checking developer mode: {str(e)}")
        
        try:
            check_cuda()
        except Exception as e:
            logging.warning(f"Error checking CUDA: {str(e)}")
        
        try:
            # Use improved dependency checking
            self.dependencies = check_dependencies()
        except Exception as e:
            logging.warning(f"Error checking dependencies: {str(e)}")
            self.dependencies = {}

        self.init_ui()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    

    def show_diarization_help(self):
        """Show help dialog for speaker diarization"""
        QMessageBox.information(
            self,
            "Speaker Diarization Help",
            "<h2>Speaker Diarization in Susurrus</h2>"
            "<p>Speaker diarization identifies different speakers in your audio recordings "
            "and creates transcriptions with speaker labels.</p>"
            "<h3>Requirements</h3>"
            "<ul>"
            "<li>A Hugging Face account and API token</li>"
            "<li>The pyannote.audio library installed</li>"
            "<li>Acceptance of the model license agreements</li>"
            "</ul>"
            "<h3>Tips for best results</h3>"
            "<ul>"
            "<li>Use clean audio with minimal background noise</li>"
            "<li>Choose language-specific models for non-English content</li>"
            "<li>Set min/max speakers if you know how many speakers to expect</li>"
            "<li>Recordings where speakers don't talk over each other work better</li>"
            "</ul>"
            "<p>Language-specific models are available for English, Chinese, German, Spanish, and Japanese.</p>"
        )


    def show_dependencies_dialog(self):
        """Show the dependencies check dialog"""
        try:
            # Run a fresh check of dependencies to get the latest status
            dependencies = check_dependencies()
        except Exception as e:
            logging.error(f"Error checking dependencies: {str(e)}")
            QMessageBox.warning(
                self,
                "Dependency Check Error",
                f"There was an error checking dependencies: {str(e)}"
            )
            dependencies = self.dependencies  # Use the stored dependencies as fallback
        
        dialog = DependenciesDialog(dependencies, self)
        dialog.exec()

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
        logging.info(f"Original model ID determined as: {original_model_id}")

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
                logging.info(f"Replaced symlink with actual file for model.bin in: {local_model_dir}")
            else:
                logging.info(f"Model already converted and exists locally in: {local_model_dir}")
            return local_model_dir, original_model_id

        else:
            # If the directory exists but model.bin is missing or empty, remove the directory
            if os.path.exists(local_model_dir):
                shutil.rmtree(local_model_dir)
            os.makedirs(local_model_dir, exist_ok=True)

        # Check if the model is already in CTranslate2 format in the Hugging Face repo
        try:
            try:
                from huggingface_hub import hf_hub_download, HfApi
            except ImportError:
                QMessageBox.critical(self, "Error", "huggingface_hub package is not installed. Cannot proceed.")
                logging.error("huggingface_hub package is not installed.")
                return None, None

            api = HfApi()
            model_files = api.list_repo_files(model_id)
            if 'model.bin' in model_files:
                logging.info(f"Found pre-converted CTranslate2 model in Hugging Face repo: {model_id}")
                # Download the model.bin and other necessary files
                for file in ['model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json', 'preprocessor_config.json']:
                    if file in model_files:
                        file_path = hf_hub_download(repo_id=model_id, filename=file, local_dir=local_model_dir)
                        # If the downloaded file is a symlink, replace it with the actual file
                        if os.path.islink(file_path):
                            real_path = os.path.realpath(file_path)
                            os.remove(file_path)
                            shutil.copy(real_path, file_path)
                return local_model_dir, original_model_id

        except Exception as e:
            logging.error(f"Error while checking Hugging Face repo: {e}")
            logging.info("Proceeding with local model conversion...")

        # If we reach here, we need to convert the model
        logging.info(f"Model not found in CTranslate2 format or couldn't be downloaded. Converting {model_id} to CTranslate2 format...")

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
            QMessageBox.information(self, "Information", "Quantization selection cancelled by user.")
            logging.info("Quantization selection cancelled by user.")
            return None, None

        reply = QMessageBox.question(
            self,
            "Model Conversion Required",
            f"The model {model_id} needs to be converted to CTranslate2 format with quantization '{quantization}'. This may take several minutes. Do you want to proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.No:
            QMessageBox.information(self, "Information", "Model conversion cancelled by user.")
            logging.info("Model conversion cancelled by user.")
            return None, None

        # Perform the model conversion
        try:
            self.convert_model_to_ctranslate2(model_id, local_model_dir, quantization)
            logging.info(f"Model converted and saved to: {local_model_dir}")
        except Exception as e:
            logging.error(f"Error during model conversion: {e}")
            QMessageBox.critical(self, "Error", f"Error during model conversion: {str(e)}")
            return None, None

        # Download the preprocessor files and save them in the same directory
        try:
            try:
                from transformers import WhisperProcessor
            except ImportError:
                QMessageBox.critical(self, "Error", "transformers package is not installed. Cannot load the preprocessor files.")
                logging.error("transformers package is not installed.")
                return None, None

            logging.info(f"Downloading preprocessor files for original model: {original_model_id}")
            preprocessor = WhisperProcessor.from_pretrained(original_model_id)
            preprocessor.save_pretrained(local_model_dir)
            logging.info(f"Preprocessor files saved to: {local_model_dir}")
        except Exception as e:
            logging.error(f"Error downloading preprocessor files: {e}")
            QMessageBox.critical(self, "Error", f"Error downloading preprocessor files: {str(e)}")
            return None, None

        return local_model_dir, original_model_id  # Return both values


    def convert_model_to_ctranslate2(self, model_id, output_dir, quantization):
        try:
            from ctranslate2.converters.transformers import TransformersConverter
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError:
            QMessageBox.critical(self, "Error", "Required packages for model conversion are not installed. Please install 'ctranslate2' and 'transformers'.")
            logging.error("Required packages 'ctranslate2' and 'transformers' are not installed.")
            return

        logging.info(f"Loading model {model_id} for conversion...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)

        logging.info(f"Converting model to CTranslate2 format...")
        converter = TransformersConverter(model, processor)
        converter.convert(output_dir, quantization=quantization, force=True)

        # Verify that the model.bin file was created and is not a symlink
        model_bin_path = os.path.join(output_dir, 'model.bin')
        if not os.path.exists(model_bin_path) or os.path.getsize(model_bin_path) == 0:
            error_msg = f"Failed to convert model. model.bin not found or empty in {output_dir}"
            logging.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            return
        if os.path.islink(model_bin_path):
            real_path = os.path.realpath(model_bin_path)
            os.remove(model_bin_path)
            shutil.copy(real_path, model_bin_path)

        logging.info(f"Model successfully converted and saved to: {output_dir}")


    def check_transcribe_button_state(self):
        if self.audio_input_path.text().strip() or self.audio_url.text().strip():
            self.transcribe_button.setEnabled(True)
        else:
            self.transcribe_button.setEnabled(False)

    def init_thread_connections(self):
        """Initialize connections for the transcription thread"""
        self.thread.progress_signal.connect(self.update_outputs)
        self.thread.error_signal.connect(self.show_error)
        self.thread.finished.connect(self.on_transcription_finished)
        self.thread.transcription_replace_signal.connect(self.replace_transcription_output)
        
        # Additional connection for diarization status updates
        if hasattr(self.thread, 'diarization_signal'):
            self.thread.diarization_signal.connect(self.update_diarization_status)

    def save_diarization_settings(self):
        """Save diarization settings to QSettings"""
        enabled = self.diarization_box.enable_diarization.isChecked()
        model = self.diarization_box.diarization_model.currentText()
        
        self.settings.setValue("diarization_enabled", enabled)
        self.settings.setValue("diarization_model", model)
        
        # Optional: save min/max speakers if needed
        min_speakers = self.diarization_box.min_speakers.text().strip()
        max_speakers = self.diarization_box.max_speakers.text().strip()
        
        if min_speakers:
            self.settings.setValue("min_speakers", min_speakers)
        if max_speakers:
            self.settings.setValue("max_speakers", max_speakers)
        
        # Sync settings to disk
        self.settings.sync()

    def create_menu_bar(self):
        """Create application menu bar"""
        menu_bar = QMenuBar(self)
        self.layout().setMenuBar(menu_bar)
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        # Open audio file action
        open_action = QAction("&Open Audio File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.select_audio_file)
        file_menu.addAction(open_action)
        
        # Save transcript action
        save_action = QAction("&Save Transcript...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_transcription)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menu_bar.addMenu("&Tools")
        
        # Transcribe action
        transcribe_action = QAction("&Transcribe", self)
        transcribe_action.setShortcut("F5")
        transcribe_action.triggered.connect(self.start_transcription)
        tools_menu.addAction(transcribe_action)
        
        # Abort action
        abort_action = QAction("&Abort Transcription", self)
        abort_action.setShortcut("Esc")
        abort_action.triggered.connect(self.abort_transcription)
        tools_menu.addAction(abort_action)
        
        tools_menu.addSeparator()
        
        # Check dependencies action
        dependencies_action = QAction("Check &Dependencies...", self)
        dependencies_action.triggered.connect(self.show_dependencies_dialog)
        tools_menu.addAction(dependencies_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        
        # About action
        about_action = QAction("&About Susurrus", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
        # Diarization help action
        diarization_help_action = QAction("Speaker &Diarization Help", self)
        diarization_help_action.triggered.connect(self.show_diarization_help)
        help_menu.addAction(diarization_help_action)

    def show_about_dialog(self):
        """Show the about dialog"""
        QMessageBox.about(
            self,
            "About Susurrus",
            "<h1>Susurrus</h1>"
            "<p>Whisper Audio Transcription with Speaker Diarization</p>"
            "<p>Version 1.1.0</p>"
            "<p>A tool for transcribing audio files with speaker identification.</p>"
            "<p>Supports multiple Whisper backends and language-specific diarization models.</p>"
            "<p>Features:</p>"
            "<ul>"
            "<li>Multiple transcription backends</li>"
            "<li>Speaker diarization with pyannote.audio</li>"
            "<li>Support for many audio formats</li>"
            "<li>Language-specific diarization models</li>"
            "<li>Multiple output formats</li>"
            "</ul>"
        )
        
    def init_ui(self):

        self.backend_model_map = {
            'mlx-whisper': [
                ("mlx-community/whisper-large-v3-turbo", "openai/whisper-large-v3-turbo"),
                ("mlx-community/whisper-large-v3-turbo-q4", "openai/whisper-large-v3-turbo"),
                ("mlx-community/whisper-tiny-mlx-4bit", "openai/whisper-tiny"),
                ("mlx-community/whisper-base-mlx-4bit", "openai/whisper-base"),
                ("mlx-community/whisper-small-mlx-q4", "openai/whisper-small"),
                ("mlx-community/whisper-medium-mlx-4bit", "openai/whisper-medium"),
                ("mlx-community/whisper-large-v3-mlx-4bit", "openai/whisper-large-v3"),
                ("mlx-community/whisper-large-v3-mlx", "openai/whisper-large-v3")
            ],
            'faster-batched': [
                ("cstr/whisper-large-v3-turbo-german-int8_float32", "openai/whisper-large-v3-turbo"),
                ("cstr/whisper-large-v3-turbo-int8_float32", "openai/whisper-large-v3-turbo"),
                ("SYSTRAN/faster-whisper-large-v1", "openai/whisper-large-v2"),
                ("GalaktischeGurke/primeline-whisper-large-v3-german-ct2", "openai/whisper-large-v3")
            ],
            'faster-sequenced': [
                ("cstr/whisper-large-v3-turbo-german-int8_float32", "openai/whisper-large-v3-turbo"),
                ("cstr/whisper-large-v3-turbo-int8_float32", "openai/whisper-large-v3-turbo"),
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
                ("large-v3-q5_0", "openai/whisper-large-v3"),
                ("medium-q5_0", "openai/whisper-medium"),
                ("small-q5_1", "openai/whisper-small"),
                ("base", "openai/whisper-base"),
                ("tiny", "openai/whisper-tiny"),
                ("tiny-q5_1", "openai/whisper-tiny"),
                ("tiny.en", "openai/whisper-tiny.en")
            ],
            'ctranslate2': [
                ("cstr/whisper-large-v3-turbo-german-int8_float32", "openai/whisper-large-v3-turbo"),
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
        self.setLayout(main_layout)

        # Create menu bar
        self.create_menu_bar()

        # Title
        title_label = QLabel("<h1 style='color: #FFFFFF;'>Susurrus: Whisper Audio Transcription</h1>")
        subtitle_label = QLabel("<p style='color: #666666;'>Transcribe audio using various (Whisper-) backends.</p>")
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)

        # Input Row
        input_row = QHBoxLayout()

        # Audio Input (file path display and button to select file)
        self.audio_input_path = QLineEdit()
        self.audio_input_path.setPlaceholderText("Select (or Drop) Audio file")
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

        # Advanced Options Collapsible Box
        self.advanced_options_box = CollapsibleBox("Advanced Options")
        advanced_layout = QVBoxLayout()

        self.diarization_box = DiarizationSettingsBox()
        main_layout.addWidget(self.diarization_box)

        # Pre-fill Hugging Face token from environment if available
        if self.has_env_token:
            self.diarization_box.hf_token.setText("Using HF_TOKEN from environment")
            self.diarization_box.hf_token.setEnabled(False)
            env_token_note = QLabel("Token loaded from environment variable")
            env_token_note.setStyleSheet("color: green; font-style: italic;")
            env_token_layout = QHBoxLayout()
            env_token_layout.addWidget(env_token_note)
            env_token_layout.addStretch()
            # Add this note to the diarization box's layout
            self.diarization_box.layout().addLayout(env_token_layout)
        
        # Load previous diarization settings from QSettings
        if self.settings.contains("diarization_enabled"):
            enabled = self.settings.value("diarization_enabled", type=bool)
            self.diarization_box.enable_diarization.setChecked(enabled)
        
        if self.settings.contains("diarization_model"):
            model = self.settings.value("diarization_model")
            index = self.diarization_box.diarization_model.findText(model)
            if index >= 0:
                self.diarization_box.diarization_model.setCurrentIndex(index)
        
        # Connect diarization settings changes to save function
        self.diarization_box.enable_diarization.toggled.connect(self.save_diarization_settings)
        self.diarization_box.diarization_model.currentTextChanged.connect(self.save_diarization_settings)

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


        # Transcription Backend Selection Row
        backend_row = QHBoxLayout()
        self.backend_selection = QComboBox()

        # Get the platform-appropriate default backend
        default_backend = get_default_backend()

        available_backends = [
            "faster-batched",
            "faster-sequenced",
            "whisper.cpp",
            "transformers",
            "OpenAI Whisper",
            "ctranslate2",
            "whisper-jax",
            "insanely-fast-whisper"
        ]

        # Add mlx-whisper only on macOS
        if platform.system().lower() == 'darwin':
            available_backends.insert(0, "mlx-whisper")

        self.backend_selection.addItems(available_backends)
        backend_row.addWidget(QLabel("Backend:"))
        backend_row.addWidget(self.backend_selection)
        advanced_layout.addLayout(backend_row)

        # Connect the backend selection change to the update_model_options method
        self.backend_selection.currentTextChanged.connect(self.update_model_options)

        # Model, Device, and Language selection row
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_id = QComboBox()
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
        default_device = get_default_device()
        self.device_selection.setCurrentText(default_device)
        model_row.addWidget(self.device_selection)

        model_row.addWidget(QLabel("Language:"))
        self.language = QLineEdit()
        self.language.setPlaceholderText("en")
        model_row.addWidget(self.language)

        # Max Chunk Length Row
        chunk_row = QHBoxLayout()
        self.max_chunk_length = QLineEdit()
        self.max_chunk_length.setPlaceholderText("Max Chunk Length (seconds, 0=No Chunking, default=0)")
        self.max_chunk_length.setText("0")  # Default value

        chunk_row.addWidget(QLabel("Max Chunk Length:"))
        chunk_row.addWidget(self.max_chunk_length)

        self.chunk_row_widget = QWidget()
        self.chunk_row_widget.setLayout(chunk_row)
        self.chunk_row_widget.setVisible(False)  # Hide initially
        advanced_layout.addWidget(self.chunk_row_widget)

        # Output Format Row
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

        advanced_layout.addLayout(model_row)

        # Now set the default backend and update models
        self.backend_selection.setCurrentText(default_backend)
        # Set initial models for the default backend
        models = self.backend_model_map[default_backend]
        self.model_id.clear()
        self.model_id.addItems([model_tuple[0] for model_tuple in models])
        default_model = get_default_model_for_backend(default_backend)
        self.model_id.setCurrentText(default_model)

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
        """)

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
        """Open file dialog with updated support for more audio formats"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Audio File", 
            "", 
            "Audio Files (*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.opus *.webm *.mp4 *.wma)"
        )
        if file_name:
            self.audio_input_path.setText(file_name)

    def update_model_options(self, backend):
        backend = backend.lower()
        models = self.backend_model_map.get(backend, [])
        self.model_id.clear()
        for model_tuple in models:
            model_id = model_tuple[0]
            self.model_id.addItem(model_id)

        # Show/hide chunking selection based on backend
        if backend in ['openai whisper', 'transformers']:
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
        logging.info(f"Backend selected: '{args['backend']}'")
        logging.info(f"Device selected: '{args['device_arg']}'")
        logging.info(f"Model selected: '{args['model_id']}'")

        # Check if diarization is enabled
        diarization_enabled = self.diarization_box.enable_diarization.isChecked()

        if diarization_enabled:
            # Check if diarization is available
            if not is_diarization_available():
                QMessageBox.critical(
                    self,
                    "Diarization Not Available",
                    "Speaker diarization is not available. Please ensure you have:\n\n"
                    "1. Installed pyannote.audio\n"
                    "2. Set a valid Hugging Face token in the HF_TOKEN environment variable\n\n"
                    "If you still see this message, there may be a version conflict between packages."
                )
                self.progress_bar.setVisible(False)
                self.transcribe_button.setEnabled(True)
                return
                
            # Get diarization parameters
            hf_token = self.diarization_box.hf_token.text().strip()
            min_speakers = self.diarization_box.min_speakers.text().strip()
            max_speakers = self.diarization_box.max_speakers.text().strip()
            diarization_model = self.diarization_box.diarization_model.currentText()
            
            # Validate token
            if not hf_token:
                # Try to get token from environment
                hf_token = os.environ.get("HF_TOKEN", "")
                
                if not hf_token:
                    QMessageBox.critical(
                        self,
                        "Missing Token",
                        "A Hugging Face token is required for speaker diarization.\n\n"
                        "Please enter your token or disable speaker diarization."
                    )
                    self.progress_bar.setVisible(False)
                    self.transcribe_button.setEnabled(True)
                    return
            
            # Store diarization settings in args
            args['diarization_enabled'] = True
            args['hf_token'] = hf_token
            args['min_speakers'] = min_speakers if min_speakers else None
            args['max_speakers'] = max_speakers if max_speakers else None
            args['diarization_model'] = diarization_model
            
            # Show a warning for first-time users
            if not self.settings.value("diarization_warning_shown", False):
                reply = QMessageBox.information(
                    self,
                    "Speaker Diarization Information",
                    "You are using speaker diarization for the first time.\n\n"
                    "Important notes:\n"
                    "• The first run will download the diarization model (approx. 1GB)\n"
                    "• Processing may take longer than standard transcription\n"
                    "• For language-specific content, consider using the matching language model\n\n"
                    "Do you want to continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.No:
                    self.progress_bar.setVisible(False)
                    self.transcribe_button.setEnabled(True)
                    return
                    
                # Don't show this warning again
                self.settings.setValue("diarization_warning_shown", True)
                    
        else:
            args['diarization_enabled'] = False

        # Handle ctranslate2 specific logic
        if args['backend'] == 'ctranslate2':
            logging.info("ctranslate2 selected. Checking...")
            # Force device to CPU if device is MPS
            if args['device_arg'] == 'mps':
                args['device_arg'] = 'cpu'

            # Find or convert the model
            try:
                model_dir, original_model_id = self.find_or_convert_ctranslate2_model(args['model_id'])
                if model_dir is None:
                    self.progress_bar.setVisible(False)
                    self.transcribe_button.setEnabled(True)
                    return

                args['model_id'] = model_dir  # Update model_id to point to the model directory
                args['original_model_id'] = original_model_id
                args['preprocessor_path'] = model_dir  # Set the preprocessor path to the same directory
                logging.info(f"Using model directory: {args['model_id']}")

                # Check if preprocessor files exist
                preprocessor_files = ["tokenizer.json", "vocabulary.json", "tokenizer_config.json"]
                preprocessor_missing = not all(os.path.exists(os.path.join(model_dir, f)) for f in preprocessor_files)

                if preprocessor_missing:
                    logging.info(f"Preprocessor files missing. Downloading from original model ID: {original_model_id}")
                    try:
                        from transformers import WhisperProcessor
                    except ImportError:
                        QMessageBox.critical(self, "Error", "transformers package is not installed. Cannot load the preprocessor files.")
                        logging.error("transformers package is not installed.")
                        self.progress_bar.setVisible(False)
                        self.transcribe_button.setEnabled(True)
                        return
                    # No need to determine original_model_id again; use the one retrieved earlier
                    try:
                        preprocessor = WhisperProcessor.from_pretrained(original_model_id)
                        preprocessor.save_pretrained(model_dir)
                        logging.info(f"Preprocessor files saved to: {model_dir}")
                    except Exception as e:
                        error_msg = f"Failed to download preprocessor files: {str(e)}"
                        logging.error(error_msg)
                        QMessageBox.critical(self, "Error", error_msg)
                        self.progress_bar.setVisible(False)
                        self.transcribe_button.setEnabled(True)
                        return
                else:
                    logging.info("Preprocessor files already exist.")

            except Exception as e:
                error_msg = f"Model preparation failed: {str(e)}"
                logging.error(error_msg)
                QMessageBox.critical(self, "Error", error_msg)
                self.progress_bar.setVisible(False)
                self.transcribe_button.setEnabled(True)
                self.abort_button.setEnabled(False)
                return

            args['model_id'] = model_dir  # Update model_id to point to the converted model directory

        self.thread = TranscriptionThread(args)
        self.thread.progress_signal.connect(self.update_outputs)
        self.thread.error_signal.connect(self.show_error)
        self.thread.finished.connect(self.on_transcription_finished)
        self.thread.transcription_replace_signal.connect(self.replace_transcription_output)  # signal for whisper.cpp output

        # connection for diarization status updates
        if hasattr(self.thread, 'diarization_signal'):
            self.thread.diarization_signal.connect(self.update_diarization_status)

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
        logging.info("Transcription aborted by user.")

    def on_transcription_finished(self):
        self.transcribe_button.setEnabled(True)
        self.abort_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        logging.info("Transcription process finished.")

    def show_error(self, error_msg):
        self.metrics_output.appendPlainText(error_msg)
        logging.error(error_msg)
        QMessageBox.critical(self, "Error", error_msg)
        self.transcribe_button.setEnabled(True)
        self.abort_button.setEnabled(False)
        self.progress_bar.setVisible(False)


    def update_outputs(self, metrics, transcription):
        if metrics:
            self.metrics_output.appendPlainText(metrics)
        if transcription:
            self.transcription_output.appendPlainText(transcription)
            self.save_button.setEnabled(True)
            #self.transcription_text += transcription + '\n' # check!
            #alternatively:
            self.transcription_text = self.transcription_output.toPlainText() # check!

    def update_diarization_status(self, status_msg):
        """Update UI with diarization status"""
        self.metrics_output.appendPlainText(status_msg)

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

def main():
    app = QApplication(sys.argv)
    
    # Check required dependencies for extended audio format support
    check_ffmpeg_installation()
    
    try:
        from pydub import AudioSegment
        logging.info("pydub is installed and available.")
    except ImportError:
        logging.warning("pydub is not installed. Installing it would improve audio format support.")
        # Show a warning dialog
        QMessageBox.warning(
            None, 
            "Missing Dependency", 
            "The pydub library is not installed. This application will have limited audio format support.\n\n"
            "To enable support for more audio formats, please install pydub:\n"
            "pip install pydub"
        )
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
