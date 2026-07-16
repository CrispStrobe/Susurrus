# main_window.py:
"""Main application window"""

import logging
import os

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenuBar,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config import APP_NAME, APP_VERSION, BACKEND_MODEL_MAP, get_settings
from models.model_config import CTranslate2ModelConverter
from utils.dependency_check import (
    check_dependencies,
    check_ffmpeg_installation,
    is_diarization_available,
)
from utils.device_detection import check_nvidia_installation, get_default_device
from workers.transcription_thread import TranscriptionThread
from workers.tts_thread import TranslationThread, TTSThread

from .dialogs import CUDADiagnosticsDialog, DependenciesDialog, InstallerDialog
from .widgets import (
    AdvancedOptionsBox,
    CrispASRAdvancedSettingsBox,
    DiarizationSettingsBox,
    TranslationSettingsWidget,
    TTSSettingsWidget,
    VoxtralSettingsBox,
)
from .widgets.collapsible_box import CollapsibleBox


class MainWindow(QWidget):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}: Audio Transcription & Speech")
        self.setMinimumSize(900, 700)
        self.setAcceptDrops(True)

        self.thread = None
        self.tts_thread = None
        self.translation_thread = None
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
        self.pytorch_needs_cuda = cuda_diagnostics["nvidia_driver"][
            "detected"
        ] and not cuda_diagnostics["pytorch"].get("cuda_available", False)

    def _show_startup_warnings(self):
        """Show warnings about missing dependencies"""
        if not self.ffmpeg_available:
            QMessageBox.warning(
                self,
                "FFMPEG Not Found",
                "FFMPEG is not installed or not in your PATH. "
                "Audio format support will be limited.\n\n"
                "You can install FFMPEG through the Tools > Install Dependencies menu.",
            )

        if self.pytorch_needs_cuda:
            reply = QMessageBox.warning(
                self,
                "PyTorch CUDA Support Missing",
                "An NVIDIA GPU was detected, but PyTorch was installed without CUDA support.\n\n"
                "Would you like to reinstall PyTorch with CUDA support now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
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
            f"<h1 style='color: #FFFFFF;'>{APP_NAME}: Audio Transcription & Speech</h1>"
        )
        subtitle_label = QLabel(
            "<p style='color: #666666;'>Transcribe, synthesize, and translate audio using "
            "multiple backends.</p>"
        )
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)

        # --- Tab widget for modes ---
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Transcription tab
        transcription_tab = self._create_transcription_tab()
        self.tab_widget.addTab(transcription_tab, "Transcription")

        # TTS tab
        tts_tab = self._create_tts_tab()
        self.tab_widget.addTab(tts_tab, "Text-to-Speech")

        # Translation tab
        translation_tab = self._create_translation_tab()
        self.tab_widget.addTab(translation_tab, "Translation")

        # History tab
        history_tab = self._create_history_tab()
        self.tab_widget.addTab(history_tab, "History")

        # Auto-refresh history when switching to that tab
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Apply styling
        self._apply_styling()

    # ---- Transcription Tab ----

    def _create_transcription_tab(self):
        """Create the transcription tab (preserves original layout)."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Input row
        input_row = self._create_input_row()
        layout.addLayout(input_row)

        # Waveform display (below input, above buttons)
        from gui.widgets.waveform_widget import WaveformWidget

        self.waveform_widget = WaveformWidget()
        self.waveform_widget.setVisible(False)
        layout.addWidget(self.waveform_widget)
        self.audio_input_path.textChanged.connect(self._on_audio_file_changed)

        # Action buttons
        button_layout = self._create_button_row()
        layout.addLayout(button_layout)

        # Settings boxes
        self.diarization_box = DiarizationSettingsBox()
        self._configure_diarization_box()
        layout.addWidget(self.diarization_box)

        self.voxtral_box = VoxtralSettingsBox()
        self._configure_voxtral_box()
        layout.addWidget(self.voxtral_box)

        # CrispASR advanced settings
        self.crispasr_box = CrispASRAdvancedSettingsBox()
        self.crispasr_box.setVisible(False)
        layout.addWidget(self.crispasr_box)

        self.advanced_options_box = AdvancedOptionsBox()
        self._configure_advanced_options()
        layout.addWidget(self.advanced_options_box)

        # Output area
        output_layout = self._create_output_area()
        layout.addLayout(output_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Batch panel (collapsible) with wired queue
        from gui.widgets.batch_panel import BatchPanel
        from workers.batch_queue import BatchQueue

        self._batch_queue = BatchQueue(
            on_job_done=self._on_batch_job_done,
            on_job_error=self._on_batch_job_error,
        )
        batch_box = CollapsibleBox("Batch Queue")
        self.batch_panel = BatchPanel()
        self.batch_panel.set_queue(self._batch_queue)
        batch_layout = QVBoxLayout()
        batch_layout.addWidget(self.batch_panel)
        batch_box.setContentLayout(batch_layout)
        layout.addWidget(batch_box)

        return tab

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

        self.save_button = QPushButton("Save")
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

        # Transcription output (segment list with fallback to plain text)
        transcription_layout = QVBoxLayout()
        transcription_label = QLabel("Transcription")
        from gui.widgets.segment_list_widget import SegmentListWidget

        self.transcription_output = SegmentListWidget()
        self.transcription_output.segment_edited.connect(self._on_segment_edited)
        transcription_layout.addWidget(transcription_label)
        transcription_layout.addWidget(self.transcription_output)

        layout.addLayout(metrics_layout, 1)
        layout.addLayout(transcription_layout, 3)

        return layout

    # ---- TTS Tab ----

    def _create_tts_tab(self):
        """Create the TTS tab."""
        self.tts_widget = TTSSettingsWidget()
        self.tts_widget.synthesize_btn.clicked.connect(self.start_synthesis)
        self.tts_widget.play_btn.clicked.connect(self._play_tts_output)
        return self.tts_widget

    # ---- Translation Tab ----

    def _create_translation_tab(self):
        """Create the translation tab."""
        self.translation_widget = TranslationSettingsWidget()
        self.translation_widget.translate_btn.clicked.connect(self.start_translation)
        return self.translation_widget

    # ---- History Tab ----

    def _create_history_tab(self):
        """Create the history browser tab."""
        from gui.widgets.history_panel import HistoryPanel

        self.history_panel = HistoryPanel()
        self.history_panel.load_entry_signal.connect(self._on_load_history_entry)
        return self.history_panel

    def _on_tab_changed(self, index):
        """Auto-refresh panels when switching tabs."""
        # History is tab index 3
        if index == 3 and hasattr(self, "history_panel"):
            self.history_panel.refresh()

    def _switch_to_history(self):
        """Switch to the History tab and refresh."""
        # History is the 4th tab (index 3)
        self.tab_widget.setCurrentIndex(3)
        if hasattr(self, "history_panel"):
            self.history_panel.refresh()

    def _on_load_history_entry(self, entry):
        """Load a history entry into the transcription output."""
        self.transcription_output.clear()
        text_parts = []
        segments = []
        for seg in entry.segments:
            start, end, text = seg[0], seg[1], seg[2]
            segments.append((start, end, text))
            if start > 0 or end > 0:
                text_parts.append(f"[{start:.2f} --> {end:.2f}]  {text}")
            else:
                text_parts.append(text)
        output = "\n".join(text_parts)
        self.transcription_output.setPlainText(output)
        self.transcription_text = output
        self._transcription_segments = segments
        self.save_button.setEnabled(True)
        self.tab_widget.setCurrentIndex(0)  # Switch to transcription tab

    # ---- Configuration ----

    def _configure_diarization_box(self):
        """Configure diarization settings from saved settings"""
        if self.has_env_token:
            self.diarization_box.hf_token.setText("Using HF_TOKEN from environment")
            self.diarization_box.hf_token.setEnabled(False)
            env_token_note = QLabel("Token loaded from environment variable")
            env_token_note.setStyleSheet("color: green; font-style: italic;")
            env_token_layout = QHBoxLayout()
            env_token_layout.addWidget(env_token_note)
            env_token_layout.addStretch()
            self.diarization_box.content_area.layout().addLayout(env_token_layout)

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

        self.diarization_box.enable_diarization.toggled.connect(self.save_diarization_settings)
        self.diarization_box.diarization_model.currentTextChanged.connect(
            self.save_diarization_settings
        )
        self.diarization_box.min_speakers.textChanged.connect(self.save_diarization_settings)
        self.diarization_box.max_speakers.textChanged.connect(self.save_diarization_settings)

    def save_diarization_settings(self):
        """Save diarization settings to QSettings"""
        enabled = self.diarization_box.enable_diarization.isChecked()
        model = self.diarization_box.diarization_model.currentText()

        self.settings.setValue("diarization_enabled", enabled)
        self.settings.setValue("diarization_model", model)

        min_speakers = self.diarization_box.min_speakers.text().strip()
        max_speakers = self.diarization_box.max_speakers.text().strip()

        if min_speakers:
            self.settings.setValue("min_speakers", min_speakers)
        if max_speakers:
            self.settings.setValue("max_speakers", max_speakers)

        self.settings.sync()

    def _configure_voxtral_box(self):
        """Configure Voxtral settings from saved settings"""
        mistral_api_key = os.environ.get("MISTRAL_API_KEY", "")
        if mistral_api_key:
            self.voxtral_box.mistral_api_key.setText("Using MISTRAL_API_KEY from environment")
            self.voxtral_box.mistral_api_key.setEnabled(False)
            env_key_note = QLabel("API key loaded from environment variable")
            env_key_note.setStyleSheet("color: green; font-style: italic;")
            env_key_layout = QHBoxLayout()
            env_key_layout.addWidget(env_key_note)
            env_key_layout.addStretch()
            self.voxtral_box.content_area.layout().addLayout(env_key_layout)
        else:
            self.has_env_mistral_key = False
            if self.settings.contains("mistral_api_key"):
                saved_key = self.settings.value("mistral_api_key")
                if saved_key:
                    self.voxtral_box.mistral_api_key.setText(saved_key)

        self.voxtral_box.mistral_api_key.textChanged.connect(self.save_voxtral_settings)

    def save_voxtral_settings(self):
        """Save Voxtral settings to QSettings"""
        api_key = self.voxtral_box.mistral_api_key.text().strip()
        if not self.has_env_mistral_key and api_key:
            self.settings.setValue("mistral_api_key", api_key)
        self.settings.sync()

    def _configure_advanced_options(self):
        """Configure advanced options with defaults"""
        import platform

        from config import get_default_backend, get_default_model_for_backend

        self.proxy_row_widget = self.advanced_options_box.proxy_row
        self.chunk_row_widget = self.advanced_options_box.chunk_row
        self.output_format_row_widget = self.advanced_options_box.output_format_row

        self.model_id = self.advanced_options_box.model_id
        self.backend_selection = self.advanced_options_box.backend_selection
        self.device_selection = self.advanced_options_box.device_selection
        self.language = self.advanced_options_box.language

        # Set up available backends — now including crispasr and sub-backends
        available_backends = [
            "crispasr",
            "crispasr-ffi",
            "faster-batched",
            "faster-sequenced",
            "whisper.cpp",
            "transformers",
            "OpenAI Whisper",
            "ctranslate2",
            "whisper-jax",
            "insanely-fast-whisper",
            "voxtral-local",
            "voxtral-api",
        ]

        # Add CrispASR sub-backends
        from config import CRISPASR_SUB_BACKENDS

        for sub in CRISPASR_SUB_BACKENDS:
            available_backends.append(f"crispasr:{sub}")

        if platform.system().lower() == "darwin":
            available_backends.insert(0, "mlx-whisper")

        self.backend_selection.addItems(available_backends)

        default_backend = get_default_backend()
        self.backend_selection.setCurrentText(default_backend)

        default_device = get_default_device()
        self.device_selection.setCurrentText(default_device)

        self.backend_selection.currentTextChanged.connect(self.update_model_options)

        self.update_model_options(default_backend)

        default_model = get_default_model_for_backend(default_backend)
        self.model_id.setCurrentText(default_model)

    def _apply_styling(self):
        """Apply application styling using the current theme."""
        from gui.themes import THEMES

        if not hasattr(self, "_current_theme"):
            # Load from QSettings on first call
            try:
                from config import get_settings

                settings = get_settings()
                self._current_theme = settings.value("theme", "dark")
            except Exception:
                self._current_theme = "dark"

        css = THEMES.get(self._current_theme, THEMES["dark"])
        self.setStyleSheet(css)

    def _toggle_theme(self):
        """Toggle between light and dark themes and persist."""
        current = getattr(self, "_current_theme", "dark")
        self._current_theme = "light" if current == "dark" else "dark"
        self._apply_styling()
        try:
            from config import get_settings

            get_settings().setValue("theme", self._current_theme)
        except Exception:
            pass

    def _show_log_viewer(self):
        """Show the log viewer in a dialog."""
        from gui.widgets.log_viewer import LogViewer

        if not hasattr(self, "_log_viewer_dialog"):
            from PyQt6.QtWidgets import QDialog
            from PyQt6.QtWidgets import QVBoxLayout as QVL

            dlg = QDialog(self)
            dlg.setWindowTitle("Susurrus Logs")
            dlg.resize(800, 500)
            lay = QVL(dlg)
            viewer = LogViewer(dlg)
            # Attach handler to root logger
            handler = viewer.get_handler()
            handler.setLevel(logging.DEBUG)
            logging.getLogger().addHandler(handler)
            lay.addWidget(viewer)
            self._log_viewer_dialog = dlg
        self._log_viewer_dialog.show()
        self._log_viewer_dialog.raise_()

    def create_menu_bar(self):
        """Create application menu bar"""
        menu_bar = QMenuBar(self)
        self.layout().setMenuBar(menu_bar)

        # File menu
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open Audio File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.select_audio_file)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Transcript...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_transcription)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Tools menu
        tools_menu = menu_bar.addMenu("&Tools")

        transcribe_action = QAction("&Transcribe", self)
        transcribe_action.setShortcut("F5")
        transcribe_action.triggered.connect(self.start_transcription)
        tools_menu.addAction(transcribe_action)

        abort_action = QAction("&Abort Transcription", self)
        abort_action.setShortcut("Esc")
        abort_action.triggered.connect(self.abort_transcription)
        tools_menu.addAction(abort_action)

        tools_menu.addSeparator()

        synthesize_action = QAction("&Synthesize (TTS)", self)
        synthesize_action.setShortcut("F6")
        synthesize_action.triggered.connect(self.start_synthesis)
        tools_menu.addAction(synthesize_action)

        translate_action = QAction("Trans&late", self)
        translate_action.setShortcut("F7")
        translate_action.triggered.connect(self.start_translation)
        tools_menu.addAction(translate_action)

        tools_menu.addSeparator()

        dependencies_action = QAction("Check &Dependencies...", self)
        dependencies_action.triggered.connect(self.show_dependencies_dialog)
        tools_menu.addAction(dependencies_action)

        install_action = QAction("&Install Dependencies...", self)
        install_action.triggered.connect(self.install_dependencies)
        tools_menu.addAction(install_action)

        cuda_diagnostics_action = QAction("CUDA &Diagnostics...", self)
        cuda_diagnostics_action.triggered.connect(self.show_cuda_diagnostics)
        tools_menu.addAction(cuda_diagnostics_action)

        yt_deps_action = QAction("Install &yt-dlp Dependencies...", self)
        yt_deps_action.triggered.connect(self.install_yt_dependencies)
        tools_menu.addAction(yt_deps_action)

        voxtral_deps_action = QAction("Install &Voxtral Dependencies...", self)
        voxtral_deps_action.triggered.connect(self.install_voxtral_dependencies)
        tools_menu.addAction(voxtral_deps_action)

        # View menu
        view_menu = menu_bar.addMenu("&View")

        toggle_theme_action = QAction("Toggle &Light/Dark Theme", self)
        toggle_theme_action.setShortcut("Ctrl+T")
        toggle_theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(toggle_theme_action)

        show_history_action = QAction("&History Tab", self)
        show_history_action.setShortcut("Ctrl+H")
        show_history_action.triggered.connect(self._switch_to_history)
        view_menu.addAction(show_history_action)

        show_logs_action = QAction("Show &Logs", self)
        show_logs_action.triggered.connect(self._show_log_viewer)
        view_menu.addAction(show_logs_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("&About Susurrus", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

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
        if not urls:
            return
        local_files = [u.toLocalFile() for u in urls if u.isLocalFile()]
        if not local_files:
            return
        # First file goes to the audio input
        self.audio_input_path.setText(local_files[0])
        # Additional files go to the batch queue
        if len(local_files) > 1 and hasattr(self, "batch_panel"):
            self.batch_panel.add_files(local_files[1:])

    # Action methods
    def select_audio_file(self):
        """Open file dialog with updated support for more audio formats"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.opus *.webm *.mp4 *.wma)",
        )
        if file_name:
            self.audio_input_path.setText(file_name)

    def _on_audio_file_changed(self, path):
        """Load waveform when audio file changes."""
        if path and os.path.isfile(path) and path.lower().endswith(".wav"):
            try:
                self.waveform_widget.load_wav(path)
                self.waveform_widget.setVisible(True)
            except Exception:
                self.waveform_widget.setVisible(False)
        else:
            self.waveform_widget.clear()
            self.waveform_widget.setVisible(False)

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

    # ---- Transcription ----

    def start_transcription(self):
        """Start transcription process"""
        args = self._collect_transcription_args()
        if not self._validate_transcription_args(args):
            return

        self.thread = TranscriptionThread(args)
        self.thread.progress_signal.connect(self.update_outputs)
        self.thread.progress_percent_signal.connect(self._update_progress_bar)
        self.thread.error_signal.connect(self.show_error)
        self.thread.finished.connect(self.on_transcription_finished)
        self.thread.transcription_replace_signal.connect(self.replace_transcription_output)

        if hasattr(self.thread, "diarization_signal"):
            self.thread.diarization_signal.connect(self.update_diarization_status)

        self.thread.start()

        self.transcribe_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # start indeterminate
        self.abort_button.setEnabled(True)
        self.transcription_output.clear()
        self.metrics_output.clear()
        self._transcription_segments = []
        self._transcription_result = None

    def _collect_transcription_args(self):
        """Collect transcription arguments from UI"""
        args = {
            "audio_input": self.audio_input_path.text().strip(),
            "audio_url": self.audio_url.text().strip(),
            "proxy_url": self.advanced_options_box.proxy_url.text().strip(),
            "proxy_username": self.advanced_options_box.proxy_username.text().strip(),
            "proxy_password": self.advanced_options_box.proxy_password.text().strip(),
            "model_id": self.model_id.currentText(),
            "start_time": self.advanced_options_box.start_time.text().strip(),
            "end_time": self.advanced_options_box.end_time.text().strip(),
            "word_timestamps": False,
            "language": self.language.text().strip() or "",
            "backend": self.backend_selection.currentText(),
            "device_arg": self.device_selection.currentText(),
            "pipeline_type": "default",
            "max_chunk_length": self.advanced_options_box.max_chunk_length.text().strip(),
            "output_format": self.advanced_options_box.output_format_selection.currentText(),
        }

        args["backend"] = args["backend"].strip().lower()
        args["device_arg"] = args["device_arg"].strip().lower()

        # CrispASR advanced settings (both binary and FFI)
        if args["backend"].startswith("crispasr"):
            crispasr_kwargs = self.crispasr_box.get_kwargs()
            args["crispasr_kwargs"] = crispasr_kwargs
            # If backend is crispasr:<sub> or crispasr-ffi:<sub>, extract the sub-backend
            if ":" in args["backend"]:
                sub = args["backend"].split(":", 1)[1]
                crispasr_kwargs.setdefault("crispasr_backend", sub)

        # Diarization
        diarization_enabled = self.diarization_box.enable_diarization.isChecked()
        args["diarization_enabled"] = diarization_enabled

        if diarization_enabled:
            hf_token = self.diarization_box.hf_token.text().strip()
            if not hf_token or hf_token == "Using HF_TOKEN from environment":
                hf_token = os.environ.get("HF_TOKEN", "")

            args["hf_token"] = hf_token
            args["min_speakers"] = self.diarization_box.min_speakers.text().strip() or None
            args["max_speakers"] = self.diarization_box.max_speakers.text().strip() or None
            args["diarization_model"] = self.diarization_box.diarization_model.currentText()

        # Voxtral-specific
        if args["backend"] == "voxtral-api":
            mistral_api_key = self.voxtral_box.mistral_api_key.text().strip()
            if (
                self.has_env_mistral_key
                or mistral_api_key == "Using MISTRAL_API_KEY from environment"
            ):
                mistral_api_key = os.environ.get("MISTRAL_API_KEY", "")
            args["mistral_api_key"] = mistral_api_key

        # ctranslate2-specific
        if args["backend"] == "ctranslate2":
            logging.info("ctranslate2 selected. Checking...")
            if args["device_arg"] == "mps":
                args["device_arg"] = "cpu"

            try:
                model_dir, original_model_id = CTranslate2ModelConverter.find_or_convert_model(
                    args["model_id"], parent_widget=self
                )
                if model_dir is None:
                    self.progress_bar.setVisible(False)
                    self.transcribe_button.setEnabled(True)
                    return None

                args["model_id"] = model_dir
                args["original_model_id"] = original_model_id
                args["preprocessor_path"] = model_dir
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

        if not args["audio_input"] and not args["audio_url"]:
            QMessageBox.warning(
                self, "No Audio Input", "Please provide either an audio file or URL."
            )
            self.transcribe_button.setEnabled(True)
            return False

        # CrispASR backend availability check (like CrisperWeaver)
        if args["backend"].startswith("crispasr") and ":" in args["backend"]:
            sub = args["backend"].split(":", 1)[1]
            try:
                from utils.crispasr_utils import probe_backends

                available = probe_backends()
                if available and sub not in available:
                    QMessageBox.warning(
                        self,
                        "Backend Not Available",
                        f"The '{sub}' backend is not compiled into the CrispASR binary.\n\n"
                        f"Available backends: {', '.join(available)}\n\n"
                        "Rebuild CrispASR with the required backend enabled.",
                    )
                    self.transcribe_button.setEnabled(True)
                    return False
            except Exception:
                pass  # Probe failed — let the backend report the error

        if args["audio_url"] and (
            "youtube.com" in args["audio_url"] or "youtu.be" in args["audio_url"]
        ):
            QMessageBox.critical(
                self,
                "Unsupported URL",
                "Downloading from YouTube is not supported due to Terms of Service restrictions.\n\n"
                "Please use a local file or a direct URL to a non-YouTube media file.",
            )
            self.transcribe_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            return False

        if args["backend"] == "voxtral-api":
            mistral_api_key = args.get("mistral_api_key", "")
            if not mistral_api_key:
                QMessageBox.critical(
                    self,
                    "Missing API Key",
                    "Mistral AI API key is required for voxtral-api backend.\n\n"
                    "Please enter your API key in the Voxtral API Settings section,\n"
                    "or set the MISTRAL_API_KEY environment variable.\n\n"
                    "Get your API key from: https://console.mistral.ai/",
                )
                self.transcribe_button.setEnabled(True)
                return False

        if args["backend"] == "voxtral-local":
            try:
                from backends.transcription import VoxtralLocal  # noqa: F401
            except ImportError:
                reply = QMessageBox.question(
                    self,
                    "Voxtral Dependencies Missing",
                    "The voxtral-local backend requires the development version of transformers.\n\n"
                    "Would you like to view installation instructions?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
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
                        "pip install mistral-common[audio] soundfile</pre>",
                    )

                self.transcribe_button.setEnabled(True)
                return False

        if args.get("diarization_enabled", False):
            if not is_diarization_available():
                QMessageBox.critical(
                    self,
                    "Diarization Not Available",
                    "Speaker diarization is not available. Please ensure you have:\n\n"
                    "1. Installed pyannote.audio\n"
                    "2. Set a valid Hugging Face token in the HF_TOKEN environment variable\n\n"
                    "If you still see this message, there may be a version conflict between packages.",
                )
                self.progress_bar.setVisible(False)
                self.transcribe_button.setEnabled(True)
                return False

            hf_token = args.get("hf_token", "")
            if not hf_token:
                QMessageBox.critical(
                    self,
                    "Missing Token",
                    "A Hugging Face token is required for speaker diarization.\n\n"
                    "Please enter your token or disable speaker diarization.",
                )
                self.progress_bar.setVisible(False)
                self.transcribe_button.setEnabled(True)
                return False

            if not self.settings.value("diarization_warning_shown", False):
                reply = QMessageBox.information(
                    self,
                    "Speaker Diarization Information",
                    "You are using speaker diarization for the first time.\n\n"
                    "Important notes:\n"
                    "- The first run will download the diarization model (approx. 1GB)\n"
                    "- Processing may take longer than standard transcription\n"
                    "- For language-specific content, consider using the matching language model\n\n"
                    "Do you want to continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )

                if reply == QMessageBox.StandardButton.No:
                    self.progress_bar.setVisible(False)
                    self.transcribe_button.setEnabled(True)
                    return False

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
        self._auto_save_history()

    def _update_progress_bar(self, fraction):
        """Update progress bar with deterministic 0-100% progress."""
        if self.progress_bar.maximum() == 0:
            # Switch from indeterminate to determinate on first real progress
            self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(int(fraction * 100))

    def update_outputs(self, metrics, transcription):
        if metrics:
            self.metrics_output.appendPlainText(metrics)
        if transcription:
            # Parse segment first
            self._parse_and_store_segment(transcription)

            # If we have structured segments, add as a segment row
            tr = getattr(self, "_transcription_result", None)
            if tr and tr.segments:
                seg = tr.segments[-1]  # last added
                idx = len(tr.segments) - 1
                speaker_display = tr.display_speaker(seg.speaker)
                self.transcription_output.add_segment(idx, seg, speaker_display)
            else:
                self.transcription_output.appendPlainText(transcription)

            self.save_button.setEnabled(True)
            self.transcription_text = self.transcription_output.toPlainText()

    def _parse_and_store_segment(self, line):
        """Parse a transcription output line and store as a Segment."""
        import re

        from utils.segment_model import TranscriptionResult

        if not hasattr(self, "_transcription_result"):
            self._transcription_result = TranscriptionResult()
        if not hasattr(self, "_transcription_segments"):
            self._transcription_segments = []

        stripped = line.strip()
        if not stripped:
            return

        # Try [HH:MM:SS.mmm --> HH:MM:SS.mmm] text format
        m = re.match(r"\[(\d+:\d+:\d+\.\d+)\s*-->\s*(\d+:\d+:\d+\.\d+)\]\s*(.*)", stripped)
        if m:
            start = self._parse_ts_for_segment(m.group(1))
            end = self._parse_ts_for_segment(m.group(2))
            text = m.group(3).strip()
            # Parse optional [Speaker N] prefix
            speaker = None
            spk_m = re.match(r"\[([^\]]+)\]\s*(.*)", text)
            if spk_m and spk_m.group(1).startswith("Speaker"):
                speaker = spk_m.group(1)
                text = spk_m.group(2)
            if text:
                self._transcription_result.add_segment(start, end, text, speaker=speaker)
                self._transcription_segments.append((start, end, text))
        else:
            self._transcription_result.add_segment(0.0, 0.0, stripped)
            self._transcription_segments.append((0.0, 0.0, stripped))

    def _on_segment_edited(self, index, new_text):
        """Handle inline segment edit from SegmentListWidget."""
        tr = getattr(self, "_transcription_result", None)
        if tr:
            tr.edit_segment(index, new_text)
            self.transcription_text = self.transcription_output.toPlainText()
            logging.info("Segment %d edited", index)

    @staticmethod
    def _parse_ts_for_segment(ts_str):
        """Parse HH:MM:SS.mmm to seconds."""
        parts = ts_str.split(":")
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

    def replace_transcription_output(self, text):
        self.transcription_output.clear()
        self.transcription_output.setPlainText(text)
        self.save_button.setEnabled(True)
        self.transcription_text = text

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

    # ---- TTS ----

    def start_synthesis(self):
        """Start TTS synthesis."""
        text = self.tts_widget.get_text()
        if not text:
            QMessageBox.warning(self, "No Text", "Please enter text or load a text file.")
            return

        args = {
            "tts_backend": self.tts_widget.tts_backend.currentText(),
            "text": text,
            "output_path": self.tts_widget.output_path.text().strip() or "tts_output.wav",
            "voice": self.tts_widget.voice_selection.currentText() or None,
            "model_id": self.tts_widget.model_id.currentText() or "auto",
            "device": self.tts_widget.device_selection.currentText().lower(),
            "language": self.tts_widget.language.text().strip() or None,
            "reference_audio": self.tts_widget.reference_audio.text().strip() or None,
            "i_have_rights": self.tts_widget.i_have_rights.isChecked(),
            "no_spoken_disclaimer": self.tts_widget.no_spoken_disclaimer.isChecked(),
            "no_watermark": self.tts_widget.no_watermark.isChecked(),
            "c2pa_cert": self.tts_widget.c2pa_cert.text().strip() or None,
            "c2pa_key": self.tts_widget.c2pa_key.text().strip() or None,
            "g2p_dict": (
                self.tts_widget.g2p_dict.currentText()
                if self.tts_widget.g2p_dict.currentText() != "(default)"
                else None
            ),
        }

        self.tts_widget.status_output.clear()
        self.tts_widget.status_output.appendPlainText("Starting synthesis...")
        self.tts_widget.synthesize_btn.setEnabled(False)

        self.tts_thread = TTSThread(args)
        self.tts_thread.progress_signal.connect(
            lambda msg: self.tts_widget.status_output.appendPlainText(msg)
        )
        self.tts_thread.error_signal.connect(self._on_tts_error)
        self.tts_thread.finished_signal.connect(self._on_tts_finished)
        self.tts_thread.start()

    def _on_tts_finished(self, output_path):
        self.tts_widget.synthesize_btn.setEnabled(True)
        self.tts_widget.play_btn.setEnabled(True)
        self.tts_widget.status_output.appendPlainText(f"Done! Audio saved to: {output_path}")
        self._tts_output_path = output_path

    def _on_tts_error(self, error_msg):
        self.tts_widget.synthesize_btn.setEnabled(True)
        self.tts_widget.status_output.appendPlainText(f"Error: {error_msg}")
        QMessageBox.critical(self, "TTS Error", error_msg)

    def _play_tts_output(self):
        """Play the last TTS output."""
        path = getattr(self, "_tts_output_path", None)
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "No Audio", "No TTS output available to play.")
            return

        try:
            from pydub import AudioSegment
            from pydub.playback import play as pydub_play

            audio = AudioSegment.from_file(path)
            pydub_play(audio)
        except Exception as e:
            QMessageBox.warning(
                self, "Playback Error", f"Could not play audio: {e}\n\nThe file is saved at: {path}"
            )

    # ---- Translation ----

    def start_translation(self):
        """Start translation."""
        text = self.translation_widget.source_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Text", "Please enter text to translate.")
            return

        args = {
            "backend": self.translation_widget.translation_backend.currentText(),
            "text": text,
            "source_lang": self.translation_widget.source_lang.currentText(),
            "target_lang": self.translation_widget.target_lang.currentText(),
            "model_id": self.translation_widget.model_id.currentText() or "auto",
        }

        self.translation_widget.result_text.clear()
        self.translation_widget.status_label.setText("Translating...")
        self.translation_widget.translate_btn.setEnabled(False)

        self.translation_thread = TranslationThread(args)
        self.translation_thread.progress_signal.connect(
            lambda msg: self.translation_widget.status_label.setText(msg)
        )
        self.translation_thread.error_signal.connect(self._on_translation_error)
        self.translation_thread.result_signal.connect(self._on_translation_finished)
        self.translation_thread.start()

    def _on_translation_finished(self, result):
        self.translation_widget.translate_btn.setEnabled(True)
        self.translation_widget.result_text.setPlainText(result)
        self.translation_widget.status_label.setText("Translation complete.")

    def _on_translation_error(self, error_msg):
        self.translation_widget.translate_btn.setEnabled(True)
        self.translation_widget.status_label.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Translation Error", error_msg)

    # ---- History ----

    def _auto_save_history(self):
        """Auto-save transcription to history on completion."""
        tr = getattr(self, "_transcription_result", None)
        segments = getattr(self, "_transcription_segments", [])
        text = getattr(self, "transcription_text", "")
        if not segments and not text:
            return
        try:
            from utils.history_service import HistoryEntry, HistoryService

            source = ""
            if hasattr(self, "audio_input_path"):
                source = self.audio_input_path.text()

            speaker_names = {}
            if tr and tr.speaker_names:
                speaker_names = tr.speaker_names

            entry = HistoryEntry(
                source_path=source,
                backend=getattr(self, "_last_backend", None),
                model=getattr(self, "_last_model", None),
                language=getattr(self, "_last_language", None),
                segments=segments,
                full_text=text,
                speaker_names=speaker_names,
            )
            HistoryService().save(entry)
            logging.info("Transcription auto-saved to history: %s", entry.id)
        except Exception as e:
            logging.warning("Failed to auto-save history: %s", e)

    # ---- Batch callbacks ----

    def _on_batch_job_done(self, job):
        """Auto-save a completed batch job to history and refresh panel."""
        try:
            from utils.history_service import HistoryEntry, HistoryService

            entry = HistoryEntry(
                source_path=job.file_path,
                backend=job.backend,
                model=job.model,
                language=job.language,
                segments=job.result_segments,
                full_text=job.result_text,
            )
            HistoryService().save(entry)
            logging.info("Batch job saved to history: %s", job.file_path)
        except Exception as e:
            logging.warning("Failed to save batch job to history: %s", e)
        if hasattr(self, "batch_panel"):
            self.batch_panel.update_display()

    def _on_batch_job_error(self, job):
        """Refresh batch panel on error."""
        logging.error("Batch job failed: %s — %s", job.file_path, job.error_message)
        if hasattr(self, "batch_panel"):
            self.batch_panel.update_display()

    # ---- Save ----

    def save_transcription(self):
        if not hasattr(self, "transcription_text") or not self.transcription_text:
            QMessageBox.warning(self, "Warning", "No transcription available to save.")
            return

        filter_str = (
            "Text Files (*.txt);;"
            "SRT Subtitles (*.srt);;"
            "WebVTT Subtitles (*.vtt);;"
            "JSON (*.json);;"
            "CSV (*.csv)"
        )
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Transcription", "", filter_str
        )
        if not save_path:
            return

        try:
            from utils.export_formats import EXPORT_FORMATS

            # Determine format from extension or selected filter
            ext = os.path.splitext(save_path)[1].lower()
            fmt_map = {v[0]: k for k, v in EXPORT_FORMATS.items()}
            fmt_name = fmt_map.get(ext, "TXT")

            segments = getattr(self, "_transcription_segments", [])
            if segments and fmt_name != "TXT":
                _, export_fn = EXPORT_FORMATS[fmt_name]
                if fmt_name == "JSON":
                    metadata = {}
                    if hasattr(self, "_last_backend"):
                        metadata["backend"] = self._last_backend
                    if hasattr(self, "_last_language"):
                        metadata["language"] = self._last_language
                    content = export_fn(segments, metadata=metadata)
                else:
                    content = export_fn(segments)
            else:
                # Fallback: save raw text
                content = self.transcription_text

            with open(save_path, "w", encoding="utf-8", errors="replace") as dst:
                dst.write(content)
            QMessageBox.information(self, "Success", f"Saved to: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")

    # Menu actions
    def show_cuda_diagnostics(self):
        dialog = CUDADiagnosticsDialog(self)
        dialog.exec()

    def show_dependencies_dialog(self):
        try:
            dependencies = check_dependencies()
        except Exception as e:
            logging.error(f"Error checking dependencies: {str(e)}")
            QMessageBox.warning(
                self,
                "Dependency Check Error",
                f"There was an error checking dependencies: {str(e)}",
            )
            dependencies = self.dependencies

        dialog = DependenciesDialog(dependencies, self)
        dialog.exec()

    def install_dependencies(self):
        dialog = InstallerDialog(self)
        dialog.exec()

    def install_yt_dependencies(self):
        dialog = InstallerDialog(self)
        dialog.install_yt()

    def install_voxtral_dependencies(self):
        dialog = InstallerDialog(self)
        dialog.install_voxtral()

    def show_about_dialog(self):
        QMessageBox.about(
            self,
            "About Susurrus",
            f"<h1>{APP_NAME}</h1>"
            f"<p>Audio Transcription, TTS, Translation & S2S Suite</p>"
            f"<p>Version {APP_VERSION}</p>"
            "<p>Features:</p>"
            "<ul>"
            "<li>38+ ASR backends via CrispASR (v0.8.7)</li>"
            "<li>27+ TTS engines (local and cloud)</li>"
            "<li>Multi-language translation (m2m100, MadLad, Gemma4)</li>"
            "<li>Speech-to-speech (lfm2-audio, mini-omni2)</li>"
            "<li>Speaker diarization (PyAnnote + CrispASR methods)</li>"
            "<li>Export: SRT, VTT, JSON, CSV, TXT</li>"
            "<li>Transcription history with search</li>"
            "<li>Batch processing queue</li>"
            "<li>Light/dark themes</li>"
            "<li>Waveform display, real-time progress</li>"
            "<li>Standalone alignment (--align-only)</li>"
            "</ul>"
            "<p>Shortcuts: F5=Transcribe, Ctrl+S=Save, Ctrl+T=Theme, Ctrl+H=History</p>",
        )

    def show_diarization_help(self):
        QMessageBox.information(
            self,
            "Speaker Diarization Help",
            "<h2>Speaker Diarization in Susurrus</h2>"
            "<p>Speaker diarization identifies different speakers in your audio recordings "
            "and creates transcriptions with speaker labels.</p>"
            "<h3>Methods</h3>"
            "<ul>"
            "<li><b>PyAnnote</b> — Neural model (requires HF token)</li>"
            "<li><b>CrispASR methods</b> — energy, xcorr, vad-turns, sherpa, ecapa</li>"
            "</ul>"
            "<h3>Requirements (PyAnnote)</h3>"
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
            "</ul>",
        )

    # Model management
    def update_model_options(self, backend):
        backend_lower = backend.lower()
        models = self.backend_model_map.get(backend_lower, [])
        self.model_id.clear()
        for model_tuple in models:
            model_id = model_tuple[0]
            self.model_id.addItem(model_id)

        # If no models found and it's a crispasr backend, add "auto"
        if not models and backend_lower.startswith("crispasr"):
            self.model_id.addItem("auto")

        # Show/hide chunking selection based on backend
        if backend_lower in ["openai whisper", "transformers", "voxtral-local"]:
            self.chunk_row_widget.setVisible(True)
        else:
            self.chunk_row_widget.setVisible(False)

        # Show/hide output format selection based on backend
        if backend_lower == "whisper.cpp":
            self.output_format_row_widget.setVisible(True)
        else:
            self.output_format_row_widget.setVisible(False)

        # Show/hide Voxtral settings box based on backend
        if backend_lower in ["voxtral-local", "voxtral-api"]:
            self.voxtral_box.setVisible(True)
            if backend_lower == "voxtral-api":
                self.voxtral_box.toggle_button.setChecked(True)
                self.voxtral_box.content_area.setVisible(True)
                self.voxtral_box.update_toggle_button_text()
        else:
            self.voxtral_box.setVisible(False)

        # Show/hide CrispASR advanced settings (both binary and FFI)
        if backend_lower.startswith("crispasr"):
            self.crispasr_box.setVisible(True)
        else:
            self.crispasr_box.setVisible(False)

        # Add CrispASR-FFI sub-backends
        from config import CRISPASR_SUB_BACKENDS

        if backend_lower == "crispasr-ffi":
            # Also add ffi sub-backends below the main ones
            for sub in CRISPASR_SUB_BACKENDS:
                ffi_name = f"crispasr-ffi:{sub}"
                if self.backend_selection.findText(ffi_name) < 0:
                    self.backend_selection.addItem(ffi_name)
