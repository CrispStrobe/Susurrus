# gui/dialogs/voice_clone_wizard.py
"""3-step voice clone wizard: select audio → enter ref text → hand off to TTS."""

from PyQt6.QtWidgets import (
    QCheckBox,
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

from utils.i18n import t


class VoiceCloneWizard(QDialog):
    """Guided voice cloning wizard.

    Steps:
        1. Select reference audio (.wav)
        2. Enter or paste the reference transcription text
        3. Confirm → sets TTS tab fields and switches to it
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("wizard.title"))
        self.resize(500, 400)
        self._step = 0
        self.voice_path = ""
        self.ref_text = ""
        self.consent_given = False
        self._setup_ui()
        self._show_step(0)

    def _setup_ui(self):
        self._layout = QVBoxLayout(self)

        # Step 1: Select audio
        self._step1 = QVBoxLayout()
        self._step1_label = QLabel(
            f"<h3>{t('wizard.step1_title')}</h3><p>{t('wizard.step1_body')}</p>"
        )
        self._step1.addWidget(self._step1_label)
        file_row = QHBoxLayout()
        self._audio_path = QLineEdit()
        self._audio_path.setPlaceholderText(t("ph.ref_wav_path"))
        file_row.addWidget(self._audio_path)
        browse_btn = QPushButton(t("btn.browse_ellipsis"))
        browse_btn.clicked.connect(self._browse_audio)
        file_row.addWidget(browse_btn)
        self._step1.addLayout(file_row)
        self._step1_widget = self._wrap_layout(self._step1)
        self._layout.addWidget(self._step1_widget)

        # Step 2: Reference text
        self._step2 = QVBoxLayout()
        self._step2_label = QLabel(
            f"<h3>{t('wizard.step2_title')}</h3><p>{t('wizard.step2_body')}</p>"
        )
        self._step2.addWidget(self._step2_label)
        self._ref_text_input = QPlainTextEdit()
        self._ref_text_input.setPlaceholderText(t("ph.ref_transcription"))
        self._ref_text_input.setMaximumHeight(120)
        self._step2.addWidget(self._ref_text_input)
        self._step2_widget = self._wrap_layout(self._step2)
        self._layout.addWidget(self._step2_widget)

        # Step 3: Confirm
        self._step3 = QVBoxLayout()
        self._step3_label = QLabel(
            f"<h3>{t('wizard.step3_title')}</h3><p>{t('wizard.step3_body')}</p>"
        )
        self._step3.addWidget(self._step3_label)
        self._consent_label = QLabel(t("consent.clone_detail"))
        self._consent_label.setWordWrap(True)
        self._step3.addWidget(self._consent_label)
        # An affirmative act, not a notice: the attestation is the user's to
        # make, so "Clone Voice" stays disabled until they make it.
        self._consent_check = QCheckBox(t("consent.clone_checkbox"))
        self._consent_check.toggled.connect(self._update_next_enabled)
        self._step3.addWidget(self._consent_check)
        self._summary_label = QLabel("")
        self._step3.addWidget(self._summary_label)
        self._step3_widget = self._wrap_layout(self._step3)
        self._layout.addWidget(self._step3_widget)

        # Navigation buttons
        nav_row = QHBoxLayout()
        self._back_btn = QPushButton(t("btn.back"))
        self._back_btn.clicked.connect(self._go_back)
        nav_row.addWidget(self._back_btn)
        nav_row.addStretch()
        self._next_btn = QPushButton(t("btn.next"))
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
        self._next_btn.setText(t("btn.clone_voice") if step == 2 else t("btn.next"))
        if step == 2:
            self._summary_label.setText(
                f"<b>{t('wizard.summary_audio')}</b> {self._audio_path.text()}<br>"
                f"<b>{t('wizard.summary_ref_text')}</b> {self._ref_text_input.toPlainText()[:80]}..."
            )
        self._update_next_enabled()

    def _update_next_enabled(self):
        """Gate the final button on the consent checkbox."""
        if self._step == 2:
            self._next_btn.setEnabled(self._consent_check.isChecked())
        else:
            self._next_btn.setEnabled(True)

    def _go_next(self):
        if self._step == 0:
            if not self._audio_path.text().strip():
                QMessageBox.warning(self, t("msg.warning.title"), t("msg.select_ref_audio.body"))
                return
            self._show_step(1)
        elif self._step == 1:
            self._show_step(2)
        elif self._step == 2:
            if not self._consent_check.isChecked():
                return
            self.voice_path = self._audio_path.text().strip()
            self.ref_text = self._ref_text_input.toPlainText().strip()
            self.consent_given = True
            self.accept()

    def _go_back(self):
        if self._step > 0:
            self._show_step(self._step - 1)

    def _browse_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            t("wizard.select_audio_dialog"),
            "",
            "Audio Files (*.wav *.mp3 *.flac);;All Files (*)",
        )
        if path:
            self._audio_path.setText(path)
