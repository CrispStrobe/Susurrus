# gui/dialogs/voice_clone_wizard.py
"""3-step voice clone wizard: select audio → enter ref text → hand off to TTS."""

from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)


class VoiceCloneWizard(QDialog):
    """Guided voice cloning wizard.

    Steps:
        1. Select reference audio (.wav)
        2. Enter or paste the reference transcription text
        3. Confirm → sets TTS tab fields and switches to it
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Voice Clone Wizard")
        self.resize(500, 400)
        self._step = 0
        self.voice_path = ""
        self.ref_text = ""
        self._setup_ui()
        self._show_step(0)

    def _setup_ui(self):
        self._layout = QVBoxLayout(self)

        # Step 1: Select audio
        self._step1 = QVBoxLayout()
        self._step1_label = QLabel(
            "<h3>Step 1: Select Reference Audio</h3>"
            "<p>Choose a .wav file of the voice you want to clone.</p>"
        )
        self._step1.addWidget(self._step1_label)
        file_row = QHBoxLayout()
        self._audio_path = QLineEdit()
        self._audio_path.setPlaceholderText("Path to reference .wav file")
        file_row.addWidget(self._audio_path)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_audio)
        file_row.addWidget(browse_btn)
        self._step1.addLayout(file_row)
        self._step1_widget = self._wrap_layout(self._step1)
        self._layout.addWidget(self._step1_widget)

        # Step 2: Reference text
        self._step2 = QVBoxLayout()
        self._step2_label = QLabel(
            "<h3>Step 2: Reference Transcription</h3>"
            "<p>Enter what is spoken in the reference audio. "
            "This helps the TTS engine match the voice.</p>"
        )
        self._step2.addWidget(self._step2_label)
        self._ref_text_input = QPlainTextEdit()
        self._ref_text_input.setPlaceholderText("Enter the transcription of the reference audio...")
        self._ref_text_input.setMaximumHeight(120)
        self._step2.addWidget(self._ref_text_input)
        self._step2_widget = self._wrap_layout(self._step2)
        self._layout.addWidget(self._step2_widget)

        # Step 3: Confirm
        self._step3 = QVBoxLayout()
        self._step3_label = QLabel(
            "<h3>Step 3: Confirm</h3>"
            "<p>Voice cloning requires consent attestation (EU AI Act).</p>"
        )
        self._step3.addWidget(self._step3_label)
        self._consent_label = QLabel(
            "By proceeding, you attest that you have the rights to clone "
            "this voice (--i-have-rights will be set automatically)."
        )
        self._consent_label.setWordWrap(True)
        self._step3.addWidget(self._consent_label)
        self._summary_label = QLabel("")
        self._step3.addWidget(self._summary_label)
        self._step3_widget = self._wrap_layout(self._step3)
        self._layout.addWidget(self._step3_widget)

        # Navigation buttons
        nav_row = QHBoxLayout()
        self._back_btn = QPushButton("Back")
        self._back_btn.clicked.connect(self._go_back)
        nav_row.addWidget(self._back_btn)
        nav_row.addStretch()
        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(self._go_next)
        nav_row.addWidget(self._next_btn)
        self._layout.addLayout(nav_row)

    def _wrap_layout(self, layout):
        from PyQt6.QtWidgets import QWidget

        w = QWidget()
        w.setLayout(layout)
        return w

    def _show_step(self, step):
        self._step = step
        self._step1_widget.setVisible(step == 0)
        self._step2_widget.setVisible(step == 1)
        self._step3_widget.setVisible(step == 2)
        self._back_btn.setEnabled(step > 0)
        self._next_btn.setText("Clone Voice" if step == 2 else "Next")
        if step == 2:
            self._summary_label.setText(
                f"<b>Audio:</b> {self._audio_path.text()}<br>"
                f"<b>Ref text:</b> {self._ref_text_input.toPlainText()[:80]}..."
            )

    def _go_next(self):
        if self._step == 0:
            if not self._audio_path.text().strip():
                QMessageBox.warning(self, "Warning", "Please select a reference audio file.")
                return
            self._show_step(1)
        elif self._step == 1:
            self._show_step(2)
        elif self._step == 2:
            self.voice_path = self._audio_path.text().strip()
            self.ref_text = self._ref_text_input.toPlainText().strip()
            self.accept()

    def _go_back(self):
        if self._step > 0:
            self._show_step(self._step - 1)

    def _browse_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Audio",
            "",
            "Audio Files (*.wav *.mp3 *.flac);;All Files (*)",
        )
        if path:
            self._audio_path.setText(path)
