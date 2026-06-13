# gui/widgets/translation_settings.py
"""Translation settings widget for the Susurrus GUI."""

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config import CRISPASR_TRANSLATION_BACKENDS


class TranslationSettingsWidget(QWidget):
    """Widget for text translation configuration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Source text
        layout.addWidget(QLabel("Source text:"))
        self.source_text = QPlainTextEdit()
        self.source_text.setPlaceholderText("Enter text to translate...")
        self.source_text.setMaximumHeight(150)
        layout.addWidget(self.source_text)

        # Language and backend row
        config_row = QHBoxLayout()

        config_row.addWidget(QLabel("From:"))
        self.source_lang = QComboBox()
        self.source_lang.setEditable(True)
        self.source_lang.addItems(
            [
                "en",
                "de",
                "fr",
                "es",
                "it",
                "pt",
                "nl",
                "pl",
                "ru",
                "zh",
                "ja",
                "ko",
                "ar",
                "hi",
                "tr",
                "vi",
                "th",
                "sv",
                "da",
                "no",
                "fi",
                "cs",
                "ro",
                "hu",
                "el",
                "bg",
                "hr",
                "sk",
                "sl",
                "uk",
            ]
        )
        config_row.addWidget(self.source_lang)

        config_row.addWidget(QLabel("To:"))
        self.target_lang = QComboBox()
        self.target_lang.setEditable(True)
        self.target_lang.addItems(
            [
                "de",
                "en",
                "fr",
                "es",
                "it",
                "pt",
                "nl",
                "pl",
                "ru",
                "zh",
                "ja",
                "ko",
                "ar",
                "hi",
                "tr",
                "vi",
                "th",
                "sv",
                "da",
                "no",
                "fi",
                "cs",
                "ro",
                "hu",
                "el",
                "bg",
                "hr",
                "sk",
                "sl",
                "uk",
            ]
        )
        config_row.addWidget(self.target_lang)

        config_row.addWidget(QLabel("Backend:"))
        self.translation_backend = QComboBox()
        self.translation_backend.addItems([f"crispasr:{b}" for b in CRISPASR_TRANSLATION_BACKENDS])
        config_row.addWidget(self.translation_backend)

        config_row.addWidget(QLabel("Model:"))
        self.model_id = QComboBox()
        self.model_id.setEditable(True)
        self.model_id.addItems(["auto", "auto:q4_0", "auto:q8_0"])
        config_row.addWidget(self.model_id)
        layout.addLayout(config_row)

        # Translate button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.translate_btn = QPushButton("Translate")
        self.translate_btn.setStyleSheet(
            "background-color: #2d5d7d; color: white; font-weight: bold; padding: 10px 20px;"
        )
        btn_row.addWidget(self.translate_btn)
        layout.addLayout(btn_row)

        # Result text
        layout.addWidget(QLabel("Translation:"))
        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        layout.addWidget(self.result_text)

        # Status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888888; font-style: italic;")
        layout.addWidget(self.status_label)
