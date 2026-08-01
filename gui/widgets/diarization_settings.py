# diarization_settings.py
"""Diarization settings widget"""

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from utils.i18n import t

from .collapsible_box import CollapsibleBox


class DiarizationSettingsBox(CollapsibleBox):
    """Collapsible box for speaker diarization settings"""

    def __init__(self, parent=None):
        super().__init__("Speaker Diarization", parent)
        layout = QVBoxLayout()

        # Enable diarization checkbox
        self.enable_diarization = QCheckBox(t("chk.enable_diarization"))
        self.enable_diarization.setToolTip(t("tip.enable_diarization"))
        layout.addWidget(self.enable_diarization)

        # Hugging Face token input
        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel(t("label.hf_token")))
        self.hf_token = QLineEdit()
        self.hf_token.setPlaceholderText(t("ph.hf_token"))
        self.hf_token.setToolTip(t("tip.hf_token"))
        token_layout.addWidget(self.hf_token)

        # Token help button
        self.token_help_button = QPushButton("?")
        self.token_help_button.setMaximumWidth(30)
        self.token_help_button.clicked.connect(self.show_token_help)
        token_layout.addWidget(self.token_help_button)

        layout.addLayout(token_layout)

        # Diarization model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel(t("label.diarization_model")))
        self.diarization_model = QComboBox()

        # Add models - will be populated when DiarizationManager is available
        self.diarization_model.addItems(
            ["Default", "English", "Chinese", "German", "Spanish", "Japanese"]
        )

        model_layout.addWidget(self.diarization_model)

        # Model help button
        self.model_help_button = QPushButton("?")
        self.model_help_button.setMaximumWidth(30)
        self.model_help_button.clicked.connect(self.show_model_help)
        model_layout.addWidget(self.model_help_button)

        layout.addLayout(model_layout)

        # Min/Max speakers row
        speakers_layout = QHBoxLayout()
        speakers_layout.addWidget(QLabel(t("label.min_speakers")))
        self.min_speakers = QLineEdit()
        self.min_speakers.setPlaceholderText(t("ph.auto"))
        self.min_speakers.setMaximumWidth(60)
        speakers_layout.addWidget(self.min_speakers)

        speakers_layout.addWidget(QLabel(t("label.max_speakers_full")))
        self.max_speakers = QLineEdit()
        self.max_speakers.setPlaceholderText(t("ph.auto"))
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
            t("msg.hf_token_help.title"),
            "A Hugging Face API token is required for speaker diarization.\n\n"
            "1. Create a free account at https://huggingface.co\n"
            "2. Go to https://huggingface.co/settings/tokens\n"
            "3. Create a new token with 'read' access\n"
            "4. Copy and paste the token here\n\n"
            "Note: You need to accept the user agreement for the diarization models at "
            "https://huggingface.co/pyannote/speaker-diarization",
        )

    def show_model_help(self):
        """Show help dialog for diarization model selection"""
        QMessageBox.information(
            self,
            t("msg.diarization_model_help.title"),
            "Choose the appropriate diarization model for your audio:\n\n"
            "• Default: General purpose diarization model\n"
            "• English: Optimized for English conversations\n"
            "• Chinese: Optimized for Mandarin Chinese conversations\n"
            "• German: Optimized for German conversations\n"
            "• Spanish: Optimized for Spanish conversations\n"
            "• Japanese: Optimized for Japanese conversations\n\n"
            "Language-specific models may provide better results for their respective languages, "
            "especially for phone calls and naturalistic conversations.",
        )
