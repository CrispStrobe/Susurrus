# gui/widgets/voxtral_settings.py
"""Voxtral settings widget"""

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout

from utils.i18n import t

from .collapsible_box import CollapsibleBox


class VoxtralSettingsBox(CollapsibleBox):
    """Collapsible box for Voxtral API settings"""

    def __init__(self, parent=None):
        super().__init__("Voxtral API Settings", parent)
        layout = QVBoxLayout()

        # Info label
        info_label = QLabel(t("voxtral.description"))
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888888; font-style: italic;")
        layout.addWidget(info_label)

        # API key input
        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(QLabel(t("label.mistral_api_key")))
        self.mistral_api_key = QLineEdit()
        self.mistral_api_key.setPlaceholderText(t("ph.mistral_key"))
        self.mistral_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.mistral_api_key.setToolTip(t("tip.mistral_key"))
        api_key_layout.addWidget(self.mistral_api_key)

        # Show/Hide password button
        self.show_key_button = QPushButton("👁")
        self.show_key_button.setMaximumWidth(30)
        self.show_key_button.setCheckable(True)
        self.show_key_button.clicked.connect(self.toggle_api_key_visibility)
        api_key_layout.addWidget(self.show_key_button)

        # API key help button
        self.api_key_help_button = QPushButton("?")
        self.api_key_help_button.setMaximumWidth(30)
        self.api_key_help_button.clicked.connect(self.show_api_key_help)
        api_key_layout.addWidget(self.api_key_help_button)

        layout.addLayout(api_key_layout)

        # Supported languages info
        languages_label = QLabel(t("voxtral.languages"))
        languages_label.setWordWrap(True)
        languages_label.setStyleSheet("font-size: 11px; color: #999999;")
        layout.addWidget(languages_label)

        # Performance note
        note_label = QLabel(t("voxtral.note"))
        note_label.setWordWrap(True)
        note_label.setStyleSheet("font-size: 11px; color: #ff9900;")
        layout.addWidget(note_label)

        # Add layout to content area
        self.setContentLayout(layout)

    def toggle_api_key_visibility(self):
        """Toggle API key visibility"""
        if self.show_key_button.isChecked():
            self.mistral_api_key.setEchoMode(QLineEdit.EchoMode.Normal)
            self.show_key_button.setText("🙈")
        else:
            self.mistral_api_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.show_key_button.setText("👁")

    def show_api_key_help(self):
        """Show help dialog for Mistral API key"""
        QMessageBox.information(
            self,
            t("msg.mistral_key_help.title"),
            t("help.mistral_key"),
        )
