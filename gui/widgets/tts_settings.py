# gui/widgets/tts_settings.py
"""TTS settings widget for the Susurrus GUI."""

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config import CRISPASR_TTS_BACKENDS

# Python-native TTS backends (non-CrispASR) always offered in the GUI.
_NATIVE_TTS_BACKENDS = ["edge-tts", "piper", "kokoro-onnx", "chatterbox", "speecht5"]


class TTSSettingsWidget(QWidget):
    """Widget for TTS input and configuration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Text input
        layout.addWidget(QLabel("Text to synthesize:"))
        self.text_input = QPlainTextEdit()
        self.text_input.setPlaceholderText("Enter text here, or load from a file below...")
        self.text_input.setMaximumHeight(150)
        layout.addWidget(self.text_input)

        # File input row
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Or load text from file:"))
        self.text_file_path = QLineEdit()
        self.text_file_path.setPlaceholderText("TXT, MD, HTML, PDF, or EPUB")
        self.text_file_path.setReadOnly(True)
        file_row.addWidget(self.text_file_path)
        self.browse_file_btn = QPushButton("Browse")
        self.browse_file_btn.clicked.connect(self._browse_file)
        file_row.addWidget(self.browse_file_btn)
        layout.addLayout(file_row)

        # Backend and voice row
        bv_row = QHBoxLayout()
        bv_row.addWidget(QLabel("TTS Backend:"))
        self.tts_backend = QComboBox()
        # Native backends + every CrispASR TTS engine (synced with the release)
        self.tts_backend.addItems(
            _NATIVE_TTS_BACKENDS + [f"crispasr:{b}" for b in CRISPASR_TTS_BACKENDS]
        )
        self.tts_backend.currentTextChanged.connect(self._on_backend_changed)
        bv_row.addWidget(self.tts_backend)

        bv_row.addWidget(QLabel("Voice:"))
        self.voice_selection = QComboBox()
        self.voice_selection.setEditable(True)
        bv_row.addWidget(self.voice_selection)
        layout.addLayout(bv_row)

        # Model and device row
        md_row = QHBoxLayout()
        md_row.addWidget(QLabel("Model:"))
        self.model_id = QComboBox()
        self.model_id.setEditable(True)
        self.model_id.addItem("auto")
        md_row.addWidget(self.model_id)

        md_row.addWidget(QLabel("Device:"))
        self.device_selection = QComboBox()
        self.device_selection.addItems(["Auto", "CPU", "GPU", "MPS"])
        md_row.addWidget(self.device_selection)

        md_row.addWidget(QLabel("Language:"))
        self.language = QLineEdit()
        self.language.setPlaceholderText("de")
        self.language.setMaximumWidth(60)
        md_row.addWidget(self.language)
        layout.addLayout(md_row)

        # Voice cloning reference row
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reference audio:"))
        self.reference_audio = QLineEdit()
        self.reference_audio.setPlaceholderText("WAV file for voice cloning (optional)")
        ref_row.addWidget(self.reference_audio)
        self.browse_ref_btn = QPushButton("Browse")
        self.browse_ref_btn.clicked.connect(self._browse_reference)
        ref_row.addWidget(self.browse_ref_btn)
        layout.addLayout(ref_row)

        # Provenance / EU AI Act controls (CrispASR voice cloning)
        prov_row = QHBoxLayout()
        self.i_have_rights = QCheckBox("I have rights to clone this voice")
        self.i_have_rights.setToolTip(
            "Required for .wav voice cloning — attests consent of the cloned "
            "speaker or that it is your own voice."
        )
        prov_row.addWidget(self.i_have_rights)

        self.no_spoken_disclaimer = QCheckBox("No spoken disclaimer")
        self.no_spoken_disclaimer.setToolTip(
            "Skip the audible AI-disclosure prefix (watermark + C2PA provenance "
            "are still applied)."
        )
        prov_row.addWidget(self.no_spoken_disclaimer)

        prov_row.addWidget(QLabel("G2P dict:"))
        self.g2p_dict = QComboBox()
        self.g2p_dict.setEditable(True)
        self.g2p_dict.addItems(["(default)", "olaph", "open-dict"])
        prov_row.addWidget(self.g2p_dict)
        layout.addLayout(prov_row)

        # Provenance row 2: watermark + C2PA
        prov_row2 = QHBoxLayout()
        self.no_watermark = QCheckBox("Disable watermark")
        self.no_watermark.setToolTip(
            "Disable AudioSeal AI-content watermark — shifts marking "
            "responsibility to the operator (EU AI Act Art. 50)."
        )
        prov_row2.addWidget(self.no_watermark)

        prov_row2.addWidget(QLabel("C2PA cert:"))
        self.c2pa_cert = QLineEdit()
        self.c2pa_cert.setPlaceholderText("(bundled default)")
        self.c2pa_cert.setMaximumWidth(200)
        prov_row2.addWidget(self.c2pa_cert)
        self.browse_c2pa_cert_btn = QPushButton("...")
        self.browse_c2pa_cert_btn.setFixedWidth(30)
        self.browse_c2pa_cert_btn.clicked.connect(self._browse_c2pa_cert)
        prov_row2.addWidget(self.browse_c2pa_cert_btn)

        prov_row2.addWidget(QLabel("Key:"))
        self.c2pa_key = QLineEdit()
        self.c2pa_key.setPlaceholderText("(bundled default)")
        self.c2pa_key.setMaximumWidth(200)
        prov_row2.addWidget(self.c2pa_key)
        self.browse_c2pa_key_btn = QPushButton("...")
        self.browse_c2pa_key_btn.setFixedWidth(30)
        self.browse_c2pa_key_btn.clicked.connect(self._browse_c2pa_key)
        prov_row2.addWidget(self.browse_c2pa_key_btn)
        layout.addLayout(prov_row2)

        # Output row
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output file:"))
        self.output_path = QLineEdit()
        self.output_path.setText("tts_output.wav")
        out_row.addWidget(self.output_path)
        self.browse_output_btn = QPushButton("Browse")
        self.browse_output_btn.clicked.connect(self._browse_output)
        out_row.addWidget(self.browse_output_btn)
        layout.addLayout(out_row)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.synthesize_btn = QPushButton("Synthesize")
        self.synthesize_btn.setStyleSheet(
            "background-color: #2d7d2d; color: white; font-weight: bold; padding: 10px 20px;"
        )
        btn_row.addWidget(self.synthesize_btn)

        self.play_btn = QPushButton("Play")
        self.play_btn.setEnabled(False)
        btn_row.addWidget(self.play_btn)
        layout.addLayout(btn_row)

        # Status output
        layout.addWidget(QLabel("Status:"))
        self.status_output = QPlainTextEdit()
        self.status_output.setReadOnly(True)
        self.status_output.setMaximumHeight(100)
        layout.addWidget(self.status_output)

        # Initialize voices for default backend
        self._on_backend_changed(self.tts_backend.currentText())

    def _on_backend_changed(self, backend_name):
        """Update voice list when backend changes."""
        self.voice_selection.clear()

        try:
            from config import TTS_BACKEND_MAP

            entry = TTS_BACKEND_MAP.get(backend_name, {})
            voices = entry.get("voices", [])
            if voices:
                self.voice_selection.addItems(voices)
                default = entry.get("default_voice")
                if default:
                    self.voice_selection.setCurrentText(default)

            # Update models
            models = entry.get("models", [])
            self.model_id.clear()
            for model_tuple in models:
                self.model_id.addItem(model_tuple[0])
            if not models:
                self.model_id.addItem("auto")
        except ImportError:
            pass

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Text File",
            "",
            "Text Files (*.txt *.md *.html *.htm *.pdf *.epub)",
        )
        if path:
            self.text_file_path.setText(path)

    def _browse_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Audio",
            "",
            "Audio Files (*.wav *.mp3 *.flac)",
        )
        if path:
            self.reference_audio.setText(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Audio Output",
            "tts_output.wav",
            "Audio Files (*.wav *.mp3)",
        )
        if path:
            self.output_path.setText(path)

    def _browse_c2pa_cert(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select C2PA Certificate", "", "PEM Files (*.pem);;All Files (*)"
        )
        if path:
            self.c2pa_cert.setText(path)

    def _browse_c2pa_key(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select C2PA Private Key", "", "PEM Files (*.pem);;All Files (*)"
        )
        if path:
            self.c2pa_key.setText(path)

    def get_text(self):
        """Get the text to synthesize, either from the text area or loaded file."""
        text = self.text_input.toPlainText().strip()
        if text:
            return text

        file_path = self.text_file_path.text().strip()
        if file_path:
            try:
                from utils.text_extraction import extract_text

                return extract_text(file_path)
            except Exception as e:
                self.status_output.appendPlainText(f"Error reading file: {e}")
        return None
