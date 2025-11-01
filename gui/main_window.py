# main_window.py:
"""Main application window"""
import os
import sys
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QPushButton, QLineEdit, QFileDialog, QPlainTextEdit, 
                              QProgressBar, QMessageBox, QMenuBar, QMenu,
                              QTextBrowser, QDialog, QDialogButtonBox)
from PyQt6.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QAction
import subprocess

from config import get_settings, APP_NAME, APP_VERSION, BACKEND_MODEL_MAP
from utils.device_detection import get_default_device, check_nvidia_installation
from utils.dependency_check import check_dependencies, check_ffmpeg_installation, is_diarization_available
from workers.transcription_thread import TranscriptionThread
from .widgets import (CollapsibleBox, DiarizationSettingsBox, 
                      VoxtralSettingsBox, AdvancedOptionsBox)
from .dialogs import (DependenciesDialog, InstallerDialog, 
                      CUDADiagnosticsDialog)
from models.model_config import CTranslate2ModelConverter, get_original_model_id

class MainWindow(QWidget):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}: Whisper Audio Transcription")
        self.setMinimumSize(800, 600)
        self.setAcceptDrops(True)
        
        self.thread = None
        self.settings = get_settings()
        self.backend_model_map = BACKEND_MODEL_MAP
        
        # Check environment tokens
        self.has_env_token = bool(os.environ.get("HF_TOKEN"))
        self.has_env_mistral_key = bool(os.environ.get("MISTRAL_API_KEY"))
        
        # Run diagnostics
        self._run_diagnostics()
        
        # Initialize UI
        self.init_ui()
        
        # Show warnings if needed
        self._show_startup_warnings()
    
    def _run_diagnostics(self):
        """Run startup diagnostics"""
        logging.info("Starting system diagnostics...")
        
        # Check FFMPEG
        ffmpeg_available = check_ffmpeg_installation()
        self.ffmpeg_available = ffmpeg_available
        
        # Check dependencies
        self.dependencies = check_dependencies()
        
        # Check for PyTorch CUDA issues
        cuda_diagnostics = check_nvidia_installation()
        self.pytorch_needs_cuda = (
            cuda_diagnostics['nvidia_driver']['detected'] and
            not cuda_diagnostics['pytorch'].get('cuda_available', False)
        )
    
    def _show_startup_warnings(self):
        """Show warnings about missing dependencies"""
        if not self.ffmpeg_available:
            QMessageBox.warning(
                self,
                "FFMPEG Not Found",
                "FFMPEG is not installed or not in your PATH. "
                "Audio format support will be limited.\n\n"
                "You can install FFMPEG through the Tools > Install Dependencies menu."
            )
        
        if self.pytorch_needs_cuda:
            reply = QMessageBox.warning(
                self,
                "PyTorch CUDA Support Missing",
                "An NVIDIA GPU was detected, but PyTorch was installed without CUDA support.\n\n"
                "Would you like to reinstall PyTorch with CUDA support now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.install_dependencies()
    
    def init_ui(self):
        """Initialize user interface"""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Title
        title_label = QLabel(
            "<h1 style='color: #FFFFFF;'>Susurrus: Whisper Audio Transcription</h1>"
        )
        subtitle_label = QLabel(
            "<p style='color: #666666;'>Transcribe audio using various (Whisper-) backends.</p>"
        )
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)
        
        # Input row
        input_row = self._create_input_row()
        main_layout.addLayout(input_row)
        
        # Action buttons
        button_layout = self._create_button_row()
        main_layout.addLayout(button_layout)
        
        # Settings boxes
        self.diarization_box = DiarizationSettingsBox()
        self._configure_diarization_box()
        main_layout.addWidget(self.diarization_box)
        
        self.voxtral_box = VoxtralSettingsBox()
        self._configure_voxtral_box()
        main_layout.addWidget(self.voxtral_box)
        
        self.advanced_options_box = AdvancedOptionsBox()
        self._configure_advanced_options()
        main_layout.addWidget(self.advanced_options_box)
        
        # Output area
        output_layout = self._create_output_area()
        main_layout.addLayout(output_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Apply styling
        self._apply_styling()
    
    def _create_input_row(self):
        """Create audio input row"""
        layout = QHBoxLayout()
        
        self.audio_input_path = QLineEdit()
        self.audio_input_path.setPlaceholderText("Select (or Drop) Audio file")
        self.audio_input_path.textChanged.connect(self.check_transcribe_button_state)
        
        self.audio_input_button = QPushButton("Browse")
        self.audio_input_button.clicked.connect(self.select_audio_file)
        
        self.audio_url = QLineEdit()
        self.audio_url.setPlaceholderText("Or Enter URL of audio file or video link")
        self.audio_url.textChanged.connect(self.check_transcribe_button_state)
        self.audio_url.textChanged.connect(self.toggle_proxy_settings)
        
        layout.addWidget(QLabel("Audio File:"))
        layout.addWidget(self.audio_input_path)
        layout.addWidget(self.audio_input_button)
        layout.addWidget(QLabel("or URL:"))
        layout.addWidget(self.audio_url)
        
        return layout
    
    def _create_button_row(self):
        """Create action button row"""
        layout = QHBoxLayout()
        layout.addStretch()
        
        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.setEnabled(False)
        self.transcribe_button.clicked.connect(self.start_transcription)
        
        self.abort_button = QPushButton("Abort")
        self.abort_button.setEnabled(False)
        self.abort_button.clicked.connect(self.abort_transcription)
        
        self.save_button = QPushButton("💾")
        self.save_button.setToolTip("Save Transcription")
        self.save_button.clicked.connect(self.save_transcription)
        
        layout.addWidget(self.transcribe_button)
        layout.addWidget(self.abort_button)
        layout.addWidget(self.save_button)
        
        return layout
    
    def _create_output_area(self):
        """Create output display area"""
        layout = QHBoxLayout()
        
        # Metrics output
        metrics_layout = QVBoxLayout()
        metrics_label = QLabel("Metrics")
        self.metrics_output = QPlainTextEdit()
        self.metrics_output.setReadOnly(True)
        self.metrics_output.setMaximumWidth(300)
        metrics_layout.addWidget(metrics_label)
        metrics_layout.addWidget(self.metrics_output)
        
        # Transcription output
        transcription_layout = QVBoxLayout()
        transcription_label = QLabel("Transcription")
        self.transcription_output = QPlainTextEdit()
        self.transcription_output.setReadOnly(True)
        transcription_layout.addWidget(transcription_label)
        transcription_layout.addWidget(self.transcription_output)
        
        layout.addLayout(metrics_layout, 1)
        layout.addLayout(transcription_layout, 3)
        
        return layout
    
    def _configure_diarization_box(self):
        """Configure diarization settings from saved settings"""
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
            self.diarization_box.content_area.layout().addLayout(env_token_layout)
        
        # Load previous diarization settings from QSettings
        if self.settings.contains("diarization_enabled"):
            enabled = self.settings.value("diarization_enabled", type=bool)
            self.diarization_box.enable_diarization.setChecked(enabled)
        
        if self.settings.contains("diarization_model"):
            model = self.settings.value("diarization_model")
            index = self.diarization_box.diarization_model.findText(model)
            if index >= 0:
                self.diarization_box.diarization_model.setCurrentIndex(index)
        
        if self.settings.contains("min_speakers"):
            min_speakers = self.settings.value("min_speakers")
            if min_speakers:
                self.diarization_box.min_speakers.setText(str(min_speakers))
        
        if self.settings.contains("max_speakers"):
            max_speakers = self.settings.value("max_speakers")
            if max_speakers:
                self.diarization_box.max_speakers.setText(str(max_speakers))
        
        # Connect diarization settings changes to save function
        self.diarization_box.enable_diarization.toggled.connect(self.save_diarization_settings)
        self.diarization_box.diarization_model.currentTextChanged.connect(self.save_diarization_settings)
        self.diarization_box.min_speakers.textChanged.connect(self.save_diarization_settings)
        self.diarization_box.max_speakers.textChanged.connect(self.save_diarization_settings)

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
        
    def _configure_voxtral_box(self):
        """Configure Voxtral settings from saved settings"""
        # Pre-fill Mistral API key from environment if available
        mistral_api_key = os.environ.get("MISTRAL_API_KEY", "")
        if mistral_api_key:
            self.voxtral_box.mistral_api_key.setText("Using MISTRAL_API_KEY from environment")
            self.voxtral_box.mistral_api_key.setEnabled(False)
            env_key_note = QLabel("API key loaded from environment variable")
            env_key_note.setStyleSheet("color: green; font-style: italic;")
            env_key_layout = QHBoxLayout()
            env_key_layout.addWidget(env_key_note)
            env_key_layout.addStretch()
            # Add this note to the Voxtral box's layout
            self.voxtral_box.content_area.layout().addLayout(env_key_layout)
        else:
            self.has_env_mistral_key = False
            # Load previous Voxtral settings from QSettings
            if self.settings.contains("mistral_api_key"):
                saved_key = self.settings.value("mistral_api_key")
                if saved_key:
                    self.voxtral_box.mistral_api_key.setText(saved_key)
        
        # Connect Voxtral settings changes to save function
        self.voxtral_box.mistral_api_key.textChanged.connect(self.save_voxtral_settings)

    def save_voxtral_settings(self):
        """Save Voxtral settings to QSettings"""
        api_key = self.voxtral_box.mistral_api_key.text().strip()
        
        # Only save if not using environment variable
        if not self.has_env_mistral_key and api_key:
            self.settings.setValue("mistral_api_key", api_key)
        
        # Sync settings to disk
        self.settings.sync()

    
    def _configure_advanced_options(self):
        """Configure advanced options with defaults"""
        from config import get_default_backend, get_default_model_for_backend
        import platform
        
        # Store reference to proxy row widget for toggle
        self.proxy_row_widget = self.advanced_options_box.proxy_row
        self.chunk_row_widget = self.advanced_options_box.chunk_row
        self.output_format_row_widget = self.advanced_options_box.output_format_row
        
        # Store references to input widgets for easy access
        self.model_id = self.advanced_options_box.model_id
        self.backend_selection = self.advanced_options_box.backend_selection
        self.device_selection = self.advanced_options_box.device_selection
        self.language = self.advanced_options_box.language
        
        # Set up available backends
        available_backends = [
            "faster-batched",
            "faster-sequenced",
            "whisper.cpp",
            "transformers",
            "OpenAI Whisper",
            "ctranslate2",
            "whisper-jax",
            "insanely-fast-whisper",
            "voxtral-local",
            "voxtral-api"
        ]
        
        # Add mlx-whisper only on macOS
        if platform.system().lower() == 'darwin':
            available_backends.insert(0, "mlx-whisper")
        
        self.backend_selection.addItems(available_backends)
        
        # Get default backend and set it
        default_backend = get_default_backend()
        self.backend_selection.setCurrentText(default_backend)
        
        # Set default device
        default_device = get_default_device()
        self.device_selection.setCurrentText(default_device)
        
        # Connect backend selection to update models
        self.backend_selection.currentTextChanged.connect(self.update_model_options)
        
        # Initialize models for default backend
        self.update_model_options(default_backend)
        
        # Set default model
        default_model = get_default_model_for_backend(default_backend)
        self.model_id.setCurrentText(default_model)
    
    def _apply_styling(self):
        """Apply application styling"""
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
        
        # Install dependencies action
        install_action = QAction("&Install Dependencies...", self)
        install_action.triggered.connect(self.install_dependencies)
        tools_menu.addAction(install_action)
        
        # CUDA diagnostics action
        cuda_diagnostics_action = QAction("CUDA &Diagnostics...", self)
        cuda_diagnostics_action.triggered.connect(self.show_cuda_diagnostics)
        tools_menu.addAction(cuda_diagnostics_action)
        
        yt_deps_action = QAction("Install &yt-dlp Dependencies...", self)
        yt_deps_action.triggered.connect(self.install_yt_dependencies)
        tools_menu.addAction(yt_deps_action)
        
        voxtral_deps_action = QAction("Install &Voxtral Dependencies...", self)
        voxtral_deps_action.triggered.connect(self.install_voxtral_dependencies)
        tools_menu.addAction(voxtral_deps_action)
        
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
    
    # Event handlers
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
    
    # Action methods
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

    def check_transcribe_button_state(self):
        if self.audio_input_path.text().strip() or self.audio_url.text().strip():
            self.transcribe_button.setEnabled(True)
        else:
            self.transcribe_button.setEnabled(False)
    
    def toggle_proxy_settings(self):
        if self.audio_url.text().strip():
            self.proxy_row_widget.setVisible(True)
        else:
            self.proxy_row_widget.setVisible(False)

    
    def start_transcription(self):
        """Start transcription process"""
        # we use TranscriptionThread]
        from workers.transcription_thread import TranscriptionThread
        
        # Collect arguments
        args = self._collect_transcription_args()
        
        # Validate arguments
        if not self._validate_transcription_args(args):
            return
        
        # Create and start thread
        self.thread = TranscriptionThread(args)
        self.thread.progress_signal.connect(self.update_outputs)
        self.thread.error_signal.connect(self.show_error)
        self.thread.finished.connect(self.on_transcription_finished)
        self.thread.transcription_replace_signal.connect(self.replace_transcription_output)
        
        if hasattr(self.thread, 'diarization_signal'):
            self.thread.diarization_signal.connect(self.update_diarization_status)
        
        self.thread.start()
        
        # Update UI state
        self.transcribe_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.abort_button.setEnabled(True)
        self.transcription_output.clear()
        self.metrics_output.clear()
    
    def _collect_transcription_args(self):
        """Collect transcription arguments from UI"""
        args = {
            'audio_input': self.audio_input_path.text().strip(),
            'audio_url': self.audio_url.text().strip(),
            'proxy_url': self.advanced_options_box.proxy_url.text().strip(),
            'proxy_username': self.advanced_options_box.proxy_username.text().strip(),
            'proxy_password': self.advanced_options_box.proxy_password.text().strip(),
            'model_id': self.model_id.currentText(),
            'start_time': self.advanced_options_box.start_time.text().strip(),
            'end_time': self.advanced_options_box.end_time.text().strip(),
            'word_timestamps': False,
            'language': self.language.text().strip() or "",
            'backend': self.backend_selection.currentText(),
            'device_arg': self.device_selection.currentText(),
            'pipeline_type': 'default',
            'max_chunk_length': self.advanced_options_box.max_chunk_length.text().strip(),
            'output_format': self.advanced_options_box.output_format_selection.currentText(),
        }
        
        # Normalize backend and device arguments
        args['backend'] = args['backend'].strip().lower()
        args['device_arg'] = args['device_arg'].strip().lower()
        
        # Check if diarization is enabled
        diarization_enabled = self.diarization_box.enable_diarization.isChecked()
        args['diarization_enabled'] = diarization_enabled
        
        if diarization_enabled:
            # Get diarization parameters
            hf_token = self.diarization_box.hf_token.text().strip()
            
            # Try to get token from environment if not provided
            if not hf_token or hf_token == "Using HF_TOKEN from environment":
                hf_token = os.environ.get("HF_TOKEN", "")
            
            args['hf_token'] = hf_token
            args['min_speakers'] = self.diarization_box.min_speakers.text().strip() or None
            args['max_speakers'] = self.diarization_box.max_speakers.text().strip() or None
            args['diarization_model'] = self.diarization_box.diarization_model.currentText()
        
        # Voxtral-specific arguments
        if args['backend'] == 'voxtral-api':
            mistral_api_key = self.voxtral_box.mistral_api_key.text().strip()
            
            # Check if using environment variable
            if self.has_env_mistral_key or mistral_api_key == "Using MISTRAL_API_KEY from environment":
                mistral_api_key = os.environ.get("MISTRAL_API_KEY", "")
            
            args['mistral_api_key'] = mistral_api_key
        
        # Handle ctranslate2 specific logic
        if args['backend'] == 'ctranslate2':
            logging.info("ctranslate2 selected. Checking...")
            # Force device to CPU if device is MPS
            if args['device_arg'] == 'mps':
                args['device_arg'] = 'cpu'
            
            # Use the modularized converter - PASS parent_widget (self)
            try:
                model_dir, original_model_id = CTranslate2ModelConverter.find_or_convert_model(
                    args['model_id'],
                    parent_widget=self  # <-- ADD THIS PARAMETER
                )
                if model_dir is None:
                    self.progress_bar.setVisible(False)
                    self.transcribe_button.setEnabled(True)
                    return None
                
                args['model_id'] = model_dir
                args['original_model_id'] = original_model_id
                args['preprocessor_path'] = model_dir
                logging.info(f"Using model directory: {args['model_id']}")
                
            except Exception as e:
                error_msg = f"Model preparation failed: {str(e)}"
                logging.error(error_msg)
                QMessageBox.critical(self, "Error", error_msg)
                self.progress_bar.setVisible(False)
                self.transcribe_button.setEnabled(True)
                self.abort_button.setEnabled(False)
                return None
        
        return args
    
    def _validate_transcription_args(self, args):
        """Validate transcription arguments"""
        if args is None:
            return False
        
        # Check if we have audio input
        if not args['audio_input'] and not args['audio_url']:
            QMessageBox.warning(
                self,
                "No Audio Input",
                "Please provide either an audio file or URL."
            )
            self.transcribe_button.setEnabled(True)
            return False
        
        # Block YouTube URLs to be absolutely sure we comply with ToS
        if args['audio_url'] and ('youtube.com' in args['audio_url'] or 'youtu.be' in args['audio_url']):
            QMessageBox.critical(
                self,
                "Unsupported URL",
                "Downloading from YouTube is not supported due to Terms of Service restrictions.\n\n"
                "Please use a local file or a direct URL to a non-YouTube media file."
            )
            self.transcribe_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            return False
        
        # Validate Voxtral API key if using voxtral-api backend
        if args['backend'] == 'voxtral-api':
            mistral_api_key = args.get('mistral_api_key', '')
            if not mistral_api_key:
                QMessageBox.critical(
                    self,
                    "Missing API Key",
                    "Mistral AI API key is required for voxtral-api backend.\n\n"
                    "Please enter your API key in the Voxtral API Settings section,\n"
                    "or set the MISTRAL_API_KEY environment variable.\n\n"
                    "Get your API key from: https://console.mistral.ai/"
                )
                self.transcribe_button.setEnabled(True)
                return False
        
        # Validate Voxtral local backend availability
        if args['backend'] == 'voxtral-local':
            try:
                from backends.transcription import VoxtralLocal
            except ImportError:
                reply = QMessageBox.question(
                    self,
                    "Voxtral Dependencies Missing",
                    "The voxtral-local backend requires the development version of transformers.\n\n"
                    "Would you like to view installation instructions?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    QMessageBox.information(
                        self,
                        "Voxtral Installation Instructions",
                        "<h3>Installing Voxtral Dependencies</h3>"
                        "<p>Run these commands:</p>"
                        "<p><b>Windows:</b></p>"
                        "<pre>install_voxtral.bat</pre>"
                        "<p><b>Linux/Mac:</b></p>"
                        "<pre>./install_voxtral.sh</pre>"
                        "<p><b>Or manually:</b></p>"
                        "<pre>pip uninstall transformers -y\n"
                        "pip install git+https://github.com/huggingface/transformers.git\n"
                        "pip install mistral-common[audio] soundfile</pre>"
                    )
                
                self.transcribe_button.setEnabled(True)
                return False
        
        # Validate diarization settings if enabled
        if args.get('diarization_enabled', False):
            # from utils.dependency_check import is_diarization_available
            
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
                return False
            
            # Validate token
            hf_token = args.get('hf_token', '')
            if not hf_token:
                QMessageBox.critical(
                    self,
                    "Missing Token",
                    "A Hugging Face token is required for speaker diarization.\n\n"
                    "Please enter your token or disable speaker diarization."
                )
                self.progress_bar.setVisible(False)
                self.transcribe_button.setEnabled(True)
                return False
            
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
                    return False
                
                # Don't show this warning again
                self.settings.setValue("diarization_warning_shown", True)
        
        return True
    
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
    
    def update_outputs(self, metrics, transcription):
        if metrics:
            self.metrics_output.appendPlainText(metrics)
        if transcription:
            self.transcription_output.appendPlainText(transcription)
            self.save_button.setEnabled(True)
            #self.transcription_text += transcription + '\n' # check!
            #alternatively:
            self.transcription_text = self.transcription_output.toPlainText() # check!
    
    def replace_transcription_output(self, text):
        self.transcription_output.clear()
        self.transcription_output.setPlainText(text)
        self.save_button.setEnabled(True)
        self.transcription_text = text  # Update the transcription text used for saving
    
    def update_diarization_status(self, status_msg):
        """Update UI with diarization status"""
        self.metrics_output.appendPlainText(status_msg)
    
    def show_error(self, error_msg):
        self.metrics_output.appendPlainText(error_msg)
        logging.error(error_msg)
        QMessageBox.critical(self, "Error", error_msg)
        self.transcribe_button.setEnabled(True)
        self.abort_button.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def save_transcription(self):
        if hasattr(self, 'transcription_text'):
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Transcription", "", "Text Files (*.txt)")
            if save_path:
                try:
                    # Ensure text is in proper encoding before saving
                    text_to_save = self.transcription_text
                    with open(save_path, 'w', encoding='utf-8', errors='replace') as dst:
                        dst.write(text_to_save)
                    QMessageBox.information(self, "Success", f"Transcription saved to: {save_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save transcription: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "No transcription available to save.")
    
    # Menu actions
    def show_cuda_diagnostics(self):
        """Show a dialog with detailed CUDA diagnostics"""
        dialog = CUDADiagnosticsDialog(self)
        dialog.exec()
        
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
    
    def install_dependencies(self):
        """Open the dependencies installer dialog"""
        from .dialogs import InstallerDialog
        dialog = InstallerDialog(self)
        dialog.exec()
    
    def _run_diagnostics(self):
        """Run startup diagnostics"""
        logging.info("Starting system diagnostics...")
        
        # Check FFMPEG
        ffmpeg_available = check_ffmpeg_installation()
        self.ffmpeg_available = ffmpeg_available
        
        # Check dependencies
        self.dependencies = check_dependencies()
        
        # CHANGED - import now at top
        # Check for PyTorch CUDA issues
        cuda_diagnostics = check_nvidia_installation()
        self.pytorch_needs_cuda = (
            cuda_diagnostics['nvidia_driver']['detected'] and
            not cuda_diagnostics['pytorch'].get('cuda_available', False)
        )
    
    def install_yt_dependencies(self):
        """Install dependencies needed for YouTube downloading"""
        from .dialogs import InstallerDialog
        dialog = InstallerDialog(self)
        dialog.install_yt()  # Call the YouTube installation directly
    
    def install_voxtral_dependencies(self):
        """Install Voxtral dependencies"""
        from .dialogs import InstallerDialog
        dialog = InstallerDialog(self)
        dialog.install_voxtral()  # Call the Voxtral installation directly
    
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
    
    # Model management
    def update_model_options(self, backend):
        backend_lower = backend.lower()
        models = self.backend_model_map.get(backend_lower, [])
        self.model_id.clear()
        for model_tuple in models:
            model_id = model_tuple[0]
            self.model_id.addItem(model_id)

        # Show/hide chunking selection based on backend
        if backend_lower in ['openai whisper', 'transformers', 'voxtral-local']:
            self.chunk_row_widget.setVisible(True)
        else:
            self.chunk_row_widget.setVisible(False)

        # Show/hide output format selection based on backend
        if backend_lower == 'whisper.cpp':
            self.output_format_row_widget.setVisible(True)
        else:
            self.output_format_row_widget.setVisible(False)
        
        # Show/hide Voxtral settings box based on backend
        if backend_lower in ['voxtral-local', 'voxtral-api']:
            self.voxtral_box.setVisible(True)
            # Expand the box if using voxtral-api
            if backend_lower == 'voxtral-api':
                self.voxtral_box.toggle_button.setChecked(True)
                self.voxtral_box.content_area.setVisible(True)
                self.voxtral_box.update_toggle_button_text()
        else:
            self.voxtral_box.setVisible(False)
    
    